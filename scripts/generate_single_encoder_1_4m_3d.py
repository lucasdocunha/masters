#!/usr/bin/env python3
"""
Gera visualizações 3D interativas em WebGL (Three.js) para os 1.339.678 documentos do CEAP
isolando individualmente cada extrator de features:
1. DINOv2 (Vision Transformer 384-D)
2. ColPali (Vision-Language Model 128-D)

Salva em:
  - data/latent_vectors_dino_1_4m_3d.html & Desktop/espaco_latente_dino_1.4M_3d.html
  - data/latent_vectors_colpali_1_4m_3d.html & Desktop/espaco_latente_colpali_1.4M_3d.html
"""

import argparse
import base64
import json
import time
import zlib
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA

BASE_DIR = Path("/home/lucas.ocunha/masters")
DATA_DIR = BASE_DIR / "data"
NPZ_PATH = DATA_DIR / "latent_vectors_full.npz"
META_PATH = DATA_DIR / "latent_vectors_full_metadata.json"
HISTORY_PATH = BASE_DIR / "ml_backend" / "models" / "annotations_history.json"
DESKTOP_DIR = Path("/home/lucas.ocunha/Desktop")

CLASS_PALETTE = {
    "Não Mapeado": {"hex": "#52525b", "rgb": [82, 82, 91]},
    "nfe": {"hex": "#3b82f6", "rgb": [59, 130, 246]},                 # Azul
    "nfse": {"hex": "#22c55e", "rgb": [34, 197, 94]},                # Verde
    "nfc": {"hex": "#f97316", "rgb": [249, 115, 22]},                # Laranja
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
    "comprovante_termico_pedagio": {"hex": "#ff7849", "rgb": [255, 120, 73]},
}


def build_threejs_html(
    title: str,
    subtitle: str,
    badge_text: str,
    n_points: int,
    coord_b64: str,
    class_b64: str,
    palette_json: str,
) -> str:
    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
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
    
    /* Painel Superior de Controle */
    #ui-overlay {{
      position: absolute;
      top: 16px;
      left: 16px;
      z-index: 100;
      background: rgba(15, 23, 42, 0.88);
      border: 1px solid #334155;
      border-radius: 12px;
      padding: 16px 20px;
      max-width: 380px;
      box-shadow: 0 10px 25px rgba(0,0,0,0.5);
      backdrop-filter: blur(8px);
    }}
    h1 {{ font-size: 16px; font-weight: 700; color: #38bdf8; margin-bottom: 4px; }}
    .subtitle {{ font-size: 12px; color: #94a3b8; margin-bottom: 14px; }}
    
    .stats-badge {{
      display: inline-block;
      background: #0284c7;
      color: #fff;
      font-size: 11px;
      padding: 2px 8px;
      border-radius: 9999px;
      font-weight: 600;
      margin-bottom: 12px;
    }}
    
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

    /* Lista de Classes e Legenda */
    #legend-container {{
      max-height: 280px;
      overflow-y: auto;
      margin-top: 12px;
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

    /* Loading overlay */
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
  <div id="loading">Descompactando e carregando {n_points:,} pontos na GPU...</div>
  <div id="canvas-container"></div>

  <div id="ui-overlay">
    <h1>{title}</h1>
    <p class="subtitle">{subtitle}</p>
    <div class="stats-badge" id="stats-total">{badge_text}</div>

    <div class="control-row">
      <button class="btn" id="btn-toggle-unmapped">Ocultar Não Mapeados (Cinza)</button>
      <button class="btn" id="btn-reset">Reset Câmera</button>
    </div>

    <div class="control-row">
      <span>Tamanho dos Pontos</span>
      <input type="range" id="slider-size" min="0.5" max="6.0" step="0.2" value="1.8">
    </div>

    <div class="control-row">
      <span>Opacidade Não Mapeados</span>
      <input type="range" id="slider-opacity" min="0.02" max="0.8" step="0.02" value="0.22">
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
    let basePointSize = 1.8;
    let unmappedAlpha = 0.22;
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
            sizeArray[i] = basePointSize * 2.8;
          }} else {{
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
          if (vColor.r == 0.0 && vColor.g == 0.0 && vColor.b == 0.0) discard;
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


def process_encoder(encoder_name: str, vectors: np.ndarray, docs: list, history: dict):
    t_start = time.time()
    n_docs = vectors.shape[0]
    dim = vectors.shape[1]
    
    if encoder_name == "dino":
        title = "Espaço Latente 3D &bull; DINOv2 (384-D)"
        subtitle = "1.339.678 Documentos CEAP (DINOv2 ViT-S/14 Visual Layout)"
        badge_text = "DINOv2 &bull; 1.339.678 pontos (384-D)"
        out_name = "latent_vectors_dino_1_4m_3d.html"
        desktop_name = "espaco_latente_dino_1.4M_3d.html"
    elif encoder_name == "colpali":
        title = "Espaço Latente 3D &bull; ColPali VLM (128-D)"
        subtitle = "1.339.678 Documentos CEAP (ColPali SigLIP+PaliGemma Multimodal)"
        badge_text = "ColPali &bull; 1.339.678 pontos (128-D)"
        out_name = "latent_vectors_colpali_1_4m_3d.html"
        desktop_name = "espaco_latente_colpali_1.4M_3d.html"
    else:
        raise ValueError(f"Extrator desconhecido: {encoder_name}")

    print(f"\n=======================================================")
    print(f" PROCESSANDO EXTRATOR: {encoder_name.upper()} ({dim}-D)")
    print(f"=======================================================")

    print("1. Normalizando vetores L2...")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors_norm = vectors / norms

    print("2. Mapeando classes...")
    known_classes = sorted(list(set(history.values())))
    class_to_id = {"Não Mapeado": 0}
    for idx, c in enumerate(known_classes, 1):
        class_to_id[c] = idx

    class_ids = np.zeros(n_docs, dtype=np.uint8)
    class_counts = {c: 0 for c in class_to_id}

    for i in range(min(n_docs, len(docs))):
        stem = Path(docs[i]["doc_name"]).stem
        lbl = history.get(stem)
        if lbl and lbl in class_to_id:
            cid = class_to_id[lbl]
            class_ids[i] = cid
            class_counts[lbl] += 1
        else:
            class_counts["Não Mapeado"] += 1

    mapped_total = sum(v for k, v in class_counts.items() if k != "Não Mapeado")
    print(f"   Anotações mapeadas: {mapped_total:,} de {n_docs:,}")

    print(f"3. Calculando PCA 3D ({dim}-D -> 3-D)...")
    t_pca = time.time()
    pca = PCA(n_components=3, svd_solver="randomized", random_state=42)
    coords_3d = pca.fit_transform(vectors_norm)
    print(f"   PCA 3D concluído em {time.time()-t_pca:.1f}s!")

    c_mean = coords_3d.mean(axis=0)
    c_std = coords_3d.std(axis=0)
    c_std[c_std == 0] = 1.0
    coords_scaled = ((coords_3d - c_mean) / c_std * 25.0).astype(np.float32)

    print("4. Empacotando e comprimindo buffers binários...")
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

    html_content = build_threejs_html(
        title=title,
        subtitle=subtitle,
        badge_text=badge_text,
        n_points=n_docs,
        coord_b64=coord_b64,
        class_b64=class_b64,
        palette_json=json.dumps(client_palette),
    )

    out_file = DATA_DIR / out_name
    desktop_file = DESKTOP_DIR / desktop_name

    print(f"5. Gravando arquivos...")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"   -> Salvo em {out_file} ({out_file.stat().st_size / (1024*1024):.2f} MB)")

    with open(desktop_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"   -> Salvo em {desktop_file} ({desktop_file.stat().st_size / (1024*1024):.2f} MB)")

    print(f"   Concluído em {time.time()-t_start:.1f}s!")


def main():
    parser = argparse.ArgumentParser(description="Gera visualização 3D individual para DINOv2 ou ColPali.")
    parser.add_argument("--encoder", choices=["dino", "colpali", "all"], default="all", help="Extrator a processar")
    args = parser.parse_args()

    print("=== CARREGANDO DADOS PARA ESPAÇO LATENTE ISOLADO (1.34M DOCS) ===")
    t0 = time.time()
    npz = np.load(NPZ_PATH)
    
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    docs = meta["documents"]

    with open(HISTORY_PATH, "r", encoding="utf-8") as f:
        history = json.load(f)

    if args.encoder in ("dino", "all"):
        dino = npz["dino_doc_embeddings"].astype(np.float32)
        process_encoder("dino", dino, docs, history)
        del dino

    if args.encoder in ("colpali", "all"):
        vlm = npz["vlm_doc_embeddings"].astype(np.float32)
        process_encoder("colpali", vlm, docs, history)
        del vlm

    print(f"\n=== TODOS OS ARQUIVOS GERADOS COM SUCESSO EM {time.time()-t0:.1f}s! ===")


if __name__ == "__main__":
    main()
