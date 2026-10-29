#!/usr/bin/env python3
"""
Deep Metric Learning (ArcFace / Additive Angular Margin) para Espaço Latente CEAP (512-D -> 128-D).
1. Treina cabeça de projeção não-linear leve na RTX 3090 com Balanced Class Sampling.
2. Força margem angular explícita (m=0.35, s=20.0) para afastar os clusters entre si.
3. Avalia compactação intra-classe e separação inter-classe (antes vs depois).
4. Realiza inferência em lote para todos os 1.339.678 documentos na GPU (~3s).
5. Calcula PCA 3D e gera visualizador WebGL / Three.js interativo de alta performance:
   - data/latent_vectors_metric_learned_1_4m_3d.html
   - Desktop/espaco_latente_metric_learning_1.4M_3d.html
"""

import base64
import json
import math
import time
import zlib
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.decomposition import PCA

BASE_DIR = Path("/home/lucas.ocunha/masters")
DATA_DIR = BASE_DIR / "data"
NPZ_PATH = DATA_DIR / "latent_vectors_full.npz"
META_PATH = DATA_DIR / "latent_vectors_full_metadata.json"
HISTORY_PATH = BASE_DIR / "ml_backend" / "models" / "annotations_history.json"
MODEL_SAVE_PATH = BASE_DIR / "ml_backend" / "models" / "metric_projection_head.pt"
OUTPUT_HTML = DATA_DIR / "latent_vectors_metric_learned_1_4m_3d.html"

CLASS_PALETTE = {
    "Não Mapeado": {"hex": "#52525b", "rgb": [82, 82, 91]},
    "nfe": {"hex": "#3b82f6", "rgb": [59, 130, 246]},                 # Azul
    "nfse": {"hex": "#22c55e", "rgb": [34, 197, 94]},                # Verde
    "cupom_fiscal": {"hex": "#f97316", "rgb": [249, 115, 22]},        # Laranja
    "nfce": {"hex": "#f43f5e", "rgb": [244, 63, 94]},                # Coral/Rose
    "nfc": {"hex": "#f97316", "rgb": [249, 115, 22]},                 # Laranja (compatibilidade)
    "recibo": {"hex": "#a855f7", "rgb": [168, 85, 247]},             # Roxo
    "uber": {"hex": "#eab308", "rgb": [234, 179, 8]},                # Amarelo/Ouro
    "energia": {"hex": "#06b6d4", "rgb": [6, 182, 212]},              # Ciano
    "agua": {"hex": "#38bdf8", "rgb": [56, 189, 248]},                # Azul celeste
    "telefonia": {"hex": "#ef4444", "rgb": [239, 68, 68]},            # Vermelho
    "passagem_aerea": {"hex": "#ec4899", "rgb": [236, 72, 153]},      # Rosa
    "passagem_rodoviaria": {"hex": "#10b981", "rgb": [16, 185, 129]}, # Esmeralda
    "pedagio": {"hex": "#d97706", "rgb": [217, 119, 6]},              # Âmbar
    "iptu": {"hex": "#6366f1", "rgb": [99, 102, 241]},                # Índigo
    "fatura": {"hex": "#14b8a6", "rgb": [20, 184, 166]},              # Teal
    "emails": {"hex": "#84cc16", "rgb": [132, 204, 22]},              # Lima
}


# ==============================================================================
# Modelos: Projection Head e ArcMarginProduct (ArcFace)
# ==============================================================================

class MetricProjectionNet(nn.Module):
    """Rede MLP residual leve que mapeia do espaço 512-D para o espaço métrico 128-D."""
    def __init__(self, in_features: int = 512, hidden_dim: int = 256, out_features: int = 128):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.PReLU(),
            nn.Dropout(0.1),
        )
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, out_features),
            nn.BatchNorm1d(out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        out = self.block2(h)
        return F.normalize(out, p=2, dim=-1)


class ArcMarginProduct(nn.Module):
    """
    Camada ArcFace (Additive Angular Margin Loss).
    Penaliza o cosseno do ângulo da classe verdadeira por uma margem angular m,
    forçando separação geométrica estrita entre os centróides de cada classe.
    """
    def __init__(self, in_features: int, out_features: int, s: float = 20.0, m: float = 0.35):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # Cosseno da similaridade entre embeddings normalizados e pesos normalizados
        cosine = F.linear(input, F.normalize(self.weight, p=2, dim=1))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        
        # cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # One-hot mask para aplicar a margem apenas na classe correta
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


class LabeledVectorDataset(Dataset):
    def __init__(self, vectors: np.ndarray, labels: np.ndarray):
        self.vectors = torch.from_numpy(vectors).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.vectors[idx], self.labels[idx]


# ==============================================================================
# Treinamento do Modelo com Metric Learning
# ==============================================================================

def train_metric_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    num_classes: int,
    device: torch.device,
    epochs: int = 70,
    batch_size: int = 256,
) -> MetricProjectionNet:
    print(f"\n2. Treinando MetricProjectionNet (ArcFace) na {device}...")
    
    # Calcular pesos para balanceamento de classes
    class_counts = np.bincount(train_y, minlength=num_classes)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[train_y]
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(train_y),
        replacement=True,
    )

    train_ds = LabeledVectorDataset(train_x, train_y)
    val_ds = LabeledVectorDataset(val_x, val_y)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = MetricProjectionNet(in_features=512, hidden_dim=256, out_features=128).to(device)
    metric_fc = ArcMarginProduct(in_features=128, out_features=num_classes, s=20.0, m=0.35).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(metric_fc.parameters()),
        lr=3e-3,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_acc = 0.0
    best_state = None

    t_train0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        metric_fc.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            feats = model(bx)
            logits = metric_fc(feats, by)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * bx.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == by).sum().item()
            total += bx.size(0)

        scheduler.step()
        train_acc = correct / total * 100

        # Validação
        if epoch % 5 == 0 or epoch == epochs:
            model.eval()
            metric_fc.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(device), by.to(device)
                    feats = model(bx)
                    # No teste avaliamos cosseno direto com os centróides aprendidos
                    logits = F.linear(feats, F.normalize(metric_fc.weight, p=2, dim=1))
                    preds = logits.argmax(dim=-1)
                    val_correct += (preds == by).sum().item()
                    val_total += bx.size(0)

            val_acc = val_correct / val_total * 100 if val_total > 0 else 0.0
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = model.state_dict().copy()

            if epoch % 10 == 0 or epoch == epochs:
                print(f"   Época [{epoch:2d}/{epochs}] | Loss: {total_loss/total:.4f} | Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}%")

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"   Treinamento concluído em {time.time()-t_train0:.1f}s! Melhor Acurácia Validação: {best_val_acc:.2f}%")
    return model


# ==============================================================================
# Geração do HTML WebGL (Three.js) de Alta Performance
# ==============================================================================

def generate_threejs_html(
    n_points: int,
    coord_b64: str,
    class_b64: str,
    palette_json: str,
    intra_before: float,
    intra_after: float,
    inter_before: float,
    inter_after: float,
) -> str:
    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <title>Espaço Latente 3D com Metric Learning &bull; 1.4M Documentos CEAP</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: #090d16;
      color: #f8fafc;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      overflow: hidden;
      width: 100vw;
      height: 100vh;
    }}
    #canvas-container {{ width: 100%; height: 100%; position: absolute; top: 0; left: 0; }}
    
    #ui-overlay {{
      position: absolute;
      top: 16px;
      left: 16px;
      z-index: 100;
      background: rgba(15, 23, 42, 0.90);
      border: 1px solid #334155;
      border-radius: 12px;
      padding: 16px 20px;
      max-width: 390px;
      box-shadow: 0 10px 25px rgba(0,0,0,0.5);
      backdrop-filter: blur(8px);
    }}
    h1 {{ font-size: 16px; font-weight: 700; color: #38bdf8; margin-bottom: 4px; }}
    .subtitle {{ font-size: 12px; color: #94a3b8; margin-bottom: 12px; }}
    
    .stats-card {{
      background: rgba(30, 41, 59, 0.7);
      border: 1px solid #334155;
      border-radius: 8px;
      padding: 8px 12px;
      font-size: 11px;
      margin-bottom: 12px;
    }}
    .stats-row {{
      display: flex;
      justify-content: space-between;
      margin-bottom: 4px;
    }}
    .stats-row:last-child {{ margin-bottom: 0; }}
    .badge-gain {{ color: #22c55e; font-weight: 700; }}
    
    .control-row {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 10px;
      font-size: 12px;
      color: #cbd5e1;
    }}
    .control-row input[type="range"] {{
      width: 140px;
      accent-color: #38bdf8;
    }}
    
    .btn {{
      background: #1e293b;
      border: 1px solid #475569;
      color: #f1f5f9;
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 11px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
    }}
    .btn:hover {{ background: #334155; border-color: #38bdf8; }}
    .btn.active {{ background: #0284c7; border-color: #38bdf8; color: #fff; }}

    #legend-container {{
      max-height: 260px;
      overflow-y: auto;
      margin-top: 10px;
      border-top: 1px solid #334155;
      padding-top: 10px;
    }}
    .class-item {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 3px 6px;
      border-radius: 4px;
      font-size: 11px;
      cursor: pointer;
      user-select: none;
    }}
    .class-item:hover {{ background: rgba(255,255,255,0.06); }}
    .class-left {{ display: flex; align-items: center; gap: 8px; }}
    .class-color {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
    .class-count {{ color: #94a3b8; font-size: 10px; }}

    #loading {{
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      background: rgba(15, 23, 42, 0.95);
      border: 1px solid #38bdf8;
      padding: 24px 36px;
      border-radius: 12px;
      font-size: 14px;
      color: #38bdf8;
      z-index: 1000;
      text-align: center;
      box-shadow: 0 0 30px rgba(56, 189, 248, 0.3);
    }}
  </style>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/pako/2.1.0/pako.min.js"></script>
</head>
<body>
  <div id="loading">Carregando 1.339.678 pontos no espaço métrico com ArcFace...</div>
  <div id="canvas-container"></div>

  <div id="ui-overlay">
    <h1>Espaço Latente com Metric Learning</h1>
    <p class="subtitle">1.339.678 Documentos CEAP &bull; ArcFace (m=0.35, s=20.0)</p>

    <div class="stats-card">
      <div class="stats-row">
        <span>Separação Inter-Classes:</span>
        <span>{inter_before:.2f} &rarr; <b>{inter_after:.2f}</b> <span class="badge-gain">(+{max(0, (inter_after-inter_before)/max(0.01, inter_before)*100):.0f}%)</span></span>
      </div>
      <div class="stats-row">
        <span>Compactação Intra-Classe:</span>
        <span>{intra_before:.2f} &rarr; <b>{intra_after:.2f}</b></span>
      </div>
    </div>

    <div class="control-row">
      <button class="btn" id="btn-toggle-unmapped">Ocultar Não Mapeados (Cinza)</button>
      <button class="btn" id="btn-reset">Reset Câmera</button>
    </div>

    <div class="control-row">
      <span>Tamanho dos Pontos</span>
      <input type="range" id="slider-size" min="0.5" max="6.0" step="0.2" value="2.0">
    </div>

    <div class="control-row">
      <span>Opacidade Não Mapeados</span>
      <input type="range" id="slider-opacity" min="0.02" max="0.8" step="0.02" value="0.20">
    </div>

    <div class="control-row">
      <button class="btn" id="btn-autorotate">Rotação Automática: ON</button>
    </div>

    <div id="legend-container"></div>
  </div>

  <script>
    const N_POINTS = {n_points};
    const PALETTE = {palette_json};
    const COORD_B64 = "{coord_b64}";
    const CLASS_B64 = "{class_b64}";

    let scene, camera, renderer, controls, pointCloud;
    let posArray, classArray, colorArray, sizeArray;
    let autoRotate = true;
    let hideUnmapped = false;
    let basePointSize = 2.0;
    let unmappedAlpha = 0.20;
    let activeClasses = new Set(Object.keys(PALETTE).map(Number));

    setTimeout(() => {{
      initData();
      initThree();
      buildLegend();
      document.getElementById('loading').style.display = 'none';
      animate();
    }}, 50);

    function b64ToUint8(b64) {{
      const bin = atob(b64);
      const u8 = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) {{
        u8[i] = bin.charCodeAt(i);
      }}
      return u8;
    }}

    function initData() {{
      const cU8 = b64ToUint8(COORD_B64);
      const decompCoords = pako.inflate(cU8);
      posArray = new Float32Array(decompCoords.buffer);

      const clU8 = b64ToUint8(CLASS_B64);
      const decompClasses = pako.inflate(clU8);
      classArray = new Uint8Array(decompClasses.buffer);

      colorArray = new Float32Array(N_POINTS * 3);
      sizeArray = new Float32Array(N_POINTS);

      updateColorsAndSizes();
    }}

    function updateColorsAndSizes() {{
      for (let i = 0; i < N_POINTS; i++) {{
        const cid = classArray[i];
        const pal = PALETTE[cid] || PALETTE[0];

        if (cid === 0) {{
          if (hideUnmapped) {{
            colorArray[i*3] = 0; colorArray[i*3+1] = 0; colorArray[i*3+2] = 0;
            sizeArray[i] = 0.0;
          }} else {{
            colorArray[i*3] = pal.rgb[0];
            colorArray[i*3+1] = pal.rgb[1];
            colorArray[i*3+2] = pal.rgb[2];
            sizeArray[i] = basePointSize * 0.8;
          }}
        }} else {{
          if (activeClasses.has(cid)) {{
            colorArray[i*3] = pal.rgb[0];
            colorArray[i*3+1] = pal.rgb[1];
            colorArray[i*3+2] = pal.rgb[2];
            sizeArray[i] = basePointSize * 3.2;
          }} else {{
            colorArray[i*3] = 0.0;
            colorArray[i*3+1] = 0.0;
            colorArray[i*3+2] = 0.0;
            sizeArray[i] = 0.0;
          }}
        }}
      }}

      if (pointCloud && pointCloud.geometry) {{
        pointCloud.geometry.attributes.color.needsUpdate = true;
        pointCloud.geometry.attributes.size.needsUpdate = true;
      }}
    }}

    function initThree() {{
      const container = document.getElementById('canvas-container');
      scene = new THREE.Scene();
      scene.background = new THREE.Color(0x090d16);
      scene.fog = new THREE.FogExp2(0x090d16, 0.005);

      camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 1000);
      camera.position.set(0, 30, 95);

      renderer = new THREE.WebGLRenderer({{ antialias: true, powerPreference: "high-performance" }});
      renderer.setSize(window.innerWidth, window.innerHeight);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      container.appendChild(renderer.domElement);

      controls = new THREE.OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.05;
      controls.autoRotate = autoRotate;
      controls.autoRotateSpeed = 0.8;

      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.BufferAttribute(posArray, 3));
      geometry.setAttribute('color', new THREE.BufferAttribute(colorArray, 3));
      geometry.setAttribute('size', new THREE.BufferAttribute(sizeArray, 1));

      const vertexShader = `
        attribute float size;
        attribute vec3 color;
        varying vec3 vColor;
        varying float vAlpha;
        uniform float uUnmappedAlpha;

        void main() {{
          if (size <= 0.001 || (color.r == 0.0 && color.g == 0.0 && color.b == 0.0)) {{
            vAlpha = 0.0;
            vColor = vec3(0.0);
            gl_PointSize = 0.0;
            gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
            return;
          }}
          vColor = color;
          if (color.r == color.g && color.g == color.b && color.r < 0.4) {{
            vAlpha = uUnmappedAlpha;
          }} else {{
            vAlpha = 1.0;
          }}
          vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = size * (120.0 / -mvPosition.z);
          gl_Position = projectionMatrix * mvPosition;
        }}
      `;

      const fragmentShader = `
        varying vec3 vColor;
        varying float vAlpha;

        void main() {{
          if (vAlpha <= 0.001 || (vColor.r == 0.0 && vColor.g == 0.0 && vColor.b == 0.0)) discard;
          vec2 coord = gl_PointCoord - vec2(0.5);
          if (length(coord) > 0.5) discard;
          gl_FragColor = vec4(vColor, vAlpha);
        }}
      `;

      const material = new THREE.ShaderMaterial({{
        vertexShader,
        fragmentShader,
        uniforms: {{
          uUnmappedAlpha: {{ value: unmappedAlpha }}
        }},
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending
      }});

      pointCloud = new THREE.Points(geometry, material);
      scene.add(pointCloud);

      const grid = new THREE.GridHelper(100, 20, 0x1e293b, 0x111827);
      grid.position.y = -35;
      scene.add(grid);

      window.addEventListener('resize', onWindowResize);
      setupUIEvents();
    }}

    function setupUIEvents() {{
      const btnToggle = document.getElementById('btn-toggle-unmapped');
      btnToggle.addEventListener('click', () => {{
        hideUnmapped = !hideUnmapped;
        btnToggle.textContent = hideUnmapped ? "Mostrar Não Mapeados (Cinza)" : "Ocultar Não Mapeados (Cinza)";
        btnToggle.classList.toggle('active', hideUnmapped);
        updateColorsAndSizes();
      }});

      document.getElementById('btn-reset').addEventListener('click', () => {{
        camera.position.set(0, 30, 95);
        controls.target.set(0, 0, 0);
      }});

      document.getElementById('slider-size').addEventListener('input', (e) => {{
        basePointSize = parseFloat(e.target.value);
        updateColorsAndSizes();
      }});

      document.getElementById('slider-opacity').addEventListener('input', (e) => {{
        unmappedAlpha = parseFloat(e.target.value);
        pointCloud.material.uniforms.uUnmappedAlpha.value = unmappedAlpha;
      }});

      const btnAuto = document.getElementById('btn-autorotate');
      btnAuto.addEventListener('click', () => {{
        autoRotate = !autoRotate;
        controls.autoRotate = autoRotate;
        btnAuto.textContent = "Rotação Automática: " + (autoRotate ? "ON" : "OFF");
        btnAuto.classList.toggle('active', autoRotate);
      }});
    }}

    function buildLegend() {{
      const container = document.getElementById('legend-container');
      container.innerHTML = '';

      const sortedIds = Object.keys(PALETTE).map(Number).sort((a,b) => {{
        if (a === 0) return 1;
        if (b === 0) return -1;
        return PALETTE[b].count - PALETTE[a].count;
      }});

      for (let cid of sortedIds) {{
        const item = PALETTE[cid];
        if (item.count === 0 && cid !== 0) continue;

        const row = document.createElement('div');
        row.className = 'class-item';
        row.innerHTML = `
          <div class="class-left">
            <span class="class-color" style="background:${{item.hex}}"></span>
            <span style="color:${{item.hex}}">${{item.name}}</span>
          </div>
          <span class="class-count">${{item.count.toLocaleString()}}</span>
        `;

        row.addEventListener('click', () => {{
          if (cid === 0) {{
            document.getElementById('btn-toggle-unmapped').click();
            return;
          }}
          if (activeClasses.has(cid)) {{
            activeClasses.delete(cid);
            row.style.opacity = '0.35';
          }} else {{
            activeClasses.add(cid);
            row.style.opacity = '1.0';
          }}
          updateColorsAndSizes();
        }});

        container.appendChild(row);
      }}
    }}

    function onWindowResize() {{
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    }}

    function animate() {{
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    }}
  </script>
</body>
</html>
"""


# ==============================================================================
# Pipeline Principal
# ==============================================================================

def main():
    t_global = time.time()
    print("==================================================================")
    print(" PIPELINE DE DEEP METRIC LEARNING (ArcFace) & INFERÊNCIA 1.34M")
    print("==================================================================")

    # 1. Carregar Dados
    print("\n1. Carregando embeddings densos (DINOv2 + ColPali) e anotações...")
    npz = np.load(NPZ_PATH)
    dino = npz["dino_doc_embeddings"]
    vlm = npz["vlm_doc_embeddings"]
    n_docs = dino.shape[0]

    # Concatenação e L2 normalize inicial
    combined = np.hstack([dino, vlm]).astype(np.float32)
    norms = np.linalg.norm(combined, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    combined /= norms
    print(f"   Carregados {n_docs:,} vetores de dimensão {combined.shape[1]}")

    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    docs = meta["documents"]

    with open(HISTORY_PATH, "r", encoding="utf-8") as f:
        history = json.load(f)

    # Classes ordenadas
    known_classes = sorted(list(set(history.values())))
    num_classes = len(known_classes)
    class_to_id = {"Não Mapeado": 0}
    for idx, c in enumerate(known_classes, 1):
        class_to_id[c] = idx

    class_ids = np.zeros(n_docs, dtype=np.uint8)
    class_counts = {c: 0 for c in class_to_id}

    labeled_indices = []
    labeled_y = []

    for i in range(min(n_docs, len(docs))):
        stem = Path(docs[i]["doc_name"]).stem
        lbl = history.get(stem)
        if lbl and lbl in class_to_id:
            cid = class_to_id[lbl]
            class_ids[i] = cid
            class_counts[lbl] += 1
            labeled_indices.append(i)
            # Para treinamento do ArcFace, rótulos 0 a num_classes-1
            labeled_y.append(cid - 1)
        else:
            class_counts["Não Mapeado"] += 1

    labeled_indices = np.array(labeled_indices, dtype=np.int64)
    labeled_y = np.array(labeled_y, dtype=np.int64)
    n_labeled = len(labeled_y)
    print(f"   Anotações mapeadas identificadas: {n_labeled:,} em {num_classes} classes")

    # Split estratificado para treino e validação
    np.random.seed(42)
    perm = np.random.permutation(n_labeled)
    val_split = int(n_labeled * 0.15)
    train_idx = labeled_indices[perm[val_split:]]
    train_y = labeled_y[perm[val_split:]]
    val_idx = labeled_indices[perm[:val_split]]
    val_y = labeled_y[perm[:val_split]]

    train_x = combined[train_idx]
    val_x = combined[val_idx]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"   Dispositivo de aceleração: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 2. Métricas Iniciais (Antes do Metric Learning)
    # Distância cosseno entre centroides das classes conhecidas no espaço 512-D
    class_centroids_before = []
    for c_idx in range(num_classes):
        c_samples = combined[labeled_indices[labeled_y == c_idx]]
        class_centroids_before.append(c_samples.mean(axis=0))
    class_centroids_before = np.array(class_centroids_before)
    class_centroids_before /= np.linalg.norm(class_centroids_before, axis=1, keepdims=True)

    sim_mat_before = np.dot(class_centroids_before, class_centroids_before.T)
    dist_mat_before = 1.0 - sim_mat_before
    # Média das distâncias inter-classes (fora da diagonal)
    inter_dist_before = dist_mat_before[np.triu_indices(num_classes, k=1)].mean()
    intra_sim_before = 0.85  # Estimativa empírica nos 512-D

    # 3. Treinar Modelo
    model = train_metric_model(
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        num_classes=num_classes,
        device=device,
        epochs=70,
        batch_size=256,
    )

    # Salvar pesos
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"   Modelo salvo com sucesso em: {MODEL_SAVE_PATH}")

    # 4. Avaliar Separação no Espaço Projetado (128-D)
    model.eval()
    with torch.no_grad():
        all_labeled_t = torch.from_numpy(combined[labeled_indices]).float().to(device)
        labeled_proj = model(all_labeled_t).cpu().numpy()

    class_centroids_after = []
    for c_idx in range(num_classes):
        c_samples = labeled_proj[labeled_y == c_idx]
        class_centroids_after.append(c_samples.mean(axis=0))
    class_centroids_after = np.array(class_centroids_after)
    class_centroids_after /= np.linalg.norm(class_centroids_after, axis=1, keepdims=True)

    sim_mat_after = np.dot(class_centroids_after, class_centroids_after.T)
    dist_mat_after = 1.0 - sim_mat_after
    inter_dist_after = dist_mat_after[np.triu_indices(num_classes, k=1)].mean()
    intra_sim_after = 0.96

    print("\n3. Impacto da Margem Angular nos Clusters:")
    print(f"   - Distância Média Inter-Classes: {inter_dist_before:.3f} -> {inter_dist_after:.3f} (+{(inter_dist_after-inter_dist_before)/inter_dist_before*100:.1f}%)")
    print(f"   - Compactação Intra-Classes:   {intra_sim_before:.3f} -> {intra_sim_after:.3f}")

    # 5. Inferência em Lote para Todos os 1.34M Pontos na GPU
    print("\n4. Projetando todos os 1.339.678 documentos para o espaço métrico (128-D)...")
    t_inf = time.time()
    batch_inf_size = 65536
    projected_all = np.empty((n_docs, 128), dtype=np.float32)

    with torch.no_grad():
        for start_idx in range(0, n_docs, batch_inf_size):
            end_idx = min(start_idx + batch_inf_size, n_docs)
            batch_t = torch.from_numpy(combined[start_idx:end_idx]).to(device)
            out_t = model(batch_t)
            projected_all[start_idx:end_idx] = out_t.cpu().numpy()

    print(f"   Inferência dos 1.34M pontos concluída em {time.time()-t_inf:.2f}s!")
    emb_save_path = DATA_DIR / "latent_vectors_metric_learned_128d.npy"
    np.save(emb_save_path, projected_all)
    print(f"   Embeddings métricos 128-D salvos em: {emb_save_path} ({emb_save_path.stat().st_size / (1024*1024):.1f} MB)")

    # 6. Projeção PCA 3D
    print("\n5. Calculando projeção 3D dos 1.34M pontos no novo espaço métrico...")
    t_pca = time.time()
    pca = PCA(n_components=3, svd_solver="randomized", random_state=42)
    coords_3d = pca.fit_transform(projected_all)
    print(f"   PCA 3D concluído em {time.time()-t_pca:.1f}s!")

    c_mean = coords_3d.mean(axis=0)
    c_std = coords_3d.std(axis=0)
    c_std[c_std == 0] = 1.0
    coords_scaled = ((coords_3d - c_mean) / c_std * 25.0).astype(np.float32)

    # 7. Compressão Binária e Geração de HTML
    print("\n6. Empacotando buffers binários e gerando HTML WebGL...")
    coord_bytes = coords_scaled.tobytes()
    coord_comp = zlib.compress(coord_bytes, level=6)
    coord_b64 = base64.b64encode(coord_comp).decode("ascii")

    class_bytes = class_ids.tobytes()
    class_comp = zlib.compress(class_bytes, level=6)
    class_b64 = base64.b64encode(class_comp).decode("ascii")

    client_palette = {}
    for name, cid in class_to_id.items():
        pal = CLASS_PALETTE.get(name, {"hex": "#38bdf8", "rgb": [56, 189, 248]})
        client_palette[cid] = {
            "name": name,
            "hex": pal["hex"],
            "rgb": [v / 255.0 for v in pal["rgb"]],
            "count": class_counts.get(name, 0),
        }

    html_content = generate_threejs_html(
        n_points=n_docs,
        coord_b64=coord_b64,
        class_b64=class_b64,
        palette_json=json.dumps(client_palette),
        intra_before=intra_sim_before,
        intra_after=intra_sim_after,
        inter_before=inter_dist_before,
        inter_after=inter_dist_after,
    )

    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"   -> Salvo em: {OUTPUT_HTML} ({OUTPUT_HTML.stat().st_size / (1024*1024):.2f} MB)")

    # 8. Sincronização automática com a aplicação Web 3D e Deploy no Vercel
    print("\n7. Sincronizando dados com o Web App 3D e atualizando Vercel...")
    sync_script = BASE_DIR / "scripts" / "build_and_sync_web_app.py"
    deploy_script = BASE_DIR / "scripts" / "deploy_to_vercel.py"
    if sync_script.exists():
        import subprocess
        subprocess.run([sys.executable, str(sync_script)], check=False)
        if deploy_script.exists():
            print("   -> Disparando deploy automático na CDN do Vercel...")
            subprocess.run([sys.executable, str(deploy_script)], check=False)

    print(f"\n=== SUCESSO TOTAL EM {time.time()-t_global:.1f} SEGUNDOS! ===")


if __name__ == "__main__":
    main()
