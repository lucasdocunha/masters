#!/usr/bin/env python3
"""
Gera os arquivos de dados e metadados para a aplicação Web 3D na pasta web_app/data/:
1. metadata.json (paleta, classes, informações dos 4 modelos e estatísticas)
2. classes.bin.gz (IDs de classes de 0 a 15 para os 1.339.678 documentos)
3. coords_dino.bin.gz (Coordenadas 3D do DINOv2)
4. coords_colpali.bin.gz (Coordenadas 3D do ColPali)
5. coords_combined.bin.gz (Coordenadas 3D do DINOv2 + ColPali 512-D)
6. coords_metric.bin.gz (Coordenadas 3D do ArcFace Metric Learning 128-D)
"""

import os
import re
import json
import time
import gzip
import zlib
import base64
from pathlib import Path
import numpy as np

BASE_DIR = Path("/home/lucas.ocunha/masters")
DATA_DIR = BASE_DIR / "data"
WEB_APP_DIR = BASE_DIR / "web"
if not WEB_APP_DIR.exists():
    WEB_APP_DIR = BASE_DIR / "web_app"
WEB_DATA_DIR = WEB_APP_DIR / "data"
META_PATH = DATA_DIR / "latent_vectors_full_metadata.json"
HISTORY_PATH = BASE_DIR / "ml_backend" / "models" / "annotations_history.json"

CLASS_PALETTE = {
    "Não Mapeado": {"hex": "#52525b", "rgb": [82, 82, 91]},
    "nfe": {"hex": "#3b82f6", "rgb": [59, 130, 246]},                 # Azul
    "nfse": {"hex": "#22c55e", "rgb": [34, 197, 94]},                # Verde
    "cupom_fiscal": {"hex": "#f97316", "rgb": [249, 115, 22]},        # Laranja
    "nfce": {"hex": "#f43f5e", "rgb": [244, 63, 94]},                # Coral/Rose
    "recibo": {"hex": "#a855f7", "rgb": [168, 85, 247]},             # Roxo
    "uber": {"hex": "#eab308", "rgb": [234, 179, 8]},                # Amarelo/Ouro
    "energia": {"hex": "#06b6d4", "rgb": [6, 182, 212]},              # Ciano
    "agua": {"hex": "#38bdf8", "rgb": [56, 189, 248]},                # Azul celeste
    "telefonia": {"hex": "#ef4444", "rgb": [239, 68, 68]},            # Vermelho
    "passagem_aerea": {"hex": "#ec4899", "rgb": [236, 72, 153]},      # Rosa
    "passagem_rodoviaria": {"hex": "#10b981", "rgb": [16, 185, 129]}, # Esmeralda
    "pedagio": {"hex": "#b45309", "rgb": [180, 83, 9]},               # Âmbar escuro
    "sem_parar": {"hex": "#f59e0b", "rgb": [245, 158, 11]},           # Dourado
    "hotelaria": {"hex": "#e11d48", "rgb": [225, 29, 72]},            # Vinho
    "correios": {"hex": "#ca8a04", "rgb": [202, 138, 4]},             # Mostarda Correios
    "boleto": {"hex": "#6366f1", "rgb": [99, 102, 241]},              # Índigo
    "iptu": {"hex": "#8b5cf6", "rgb": [139, 92, 246]},                # Violeta
    "fatura": {"hex": "#14b8a6", "rgb": [20, 184, 166]},              # Teal
    "emails": {"hex": "#84cc16", "rgb": [132, 204, 22]},              # Lima
}

MODEL_METRICS = {
    "dino": {
        "id": "dino",
        "name": "DINOv2 (Visual Puro)",
        "badge": "Vision Transformer 384-D",
        "dims": 384,
        "type": "Visão Computacional",
        "description": "Clusterização puramente visual baseada em diagramação, logotipos e densidade gráfica de pixels.",
        "color": "#38bdf8",
        "file": "coords_dino.bin.gz",
        "inter_dist": "0.312",
        "intra_sim": "0.820"
    },
    "colpali": {
        "id": "colpali",
        "name": "ColPali (Documental Puro)",
        "badge": "Vision-Language Model 128-D",
        "dims": 128,
        "type": "Linguagem / OCR Semântico",
        "description": "Clusterização por vocabulário e padrões textuais típicos de cada tipo de documento fiscal.",
        "color": "#a855f7",
        "file": "coords_colpali.bin.gz",
        "inter_dist": "0.298",
        "intra_sim": "0.835"
    },
    "combined": {
        "id": "combined",
        "name": "Multimodal (DINO + ColPali)",
        "badge": "Fusão Densa 512-D",
        "dims": 512,
        "type": "Multimodal Não Supervisionado",
        "description": "Concatenação balanceada dos embeddings de visão e linguagem antes do fine-tuning métrico.",
        "color": "#22c55e",
        "file": "coords_combined.bin.gz",
        "inter_dist": "0.266",
        "intra_sim": "0.850"
    },
    "metric": {
        "id": "metric",
        "name": "Deep Metric Learning (ArcFace)",
        "badge": "ArcFace Margin 128-D",
        "dims": 128,
        "type": "Supervisionado / Angular Margin",
        "description": "Margem angular explícita (+299.9% de separação inter-classe, 99.85% de acurácia de validação).",
        "color": "#f43f5e",
        "file": "coords_metric.bin.gz",
        "inter_dist": "1.063",
        "intra_sim": "0.960"
    }
}


def extract_coords_from_html(html_path: Path) -> bytes:
    """Extrai os bytes brutos descompactados de coordenadas a partir do HTML gerado."""
    with open(html_path, "r", encoding="utf-8") as f:
        # Procurar COORD_B64 nas primeiras 300 linhas
        for _ in range(300):
            line = f.readline()
            if not line:
                break
            m = re.search(r"const COORD_B64\s*=\s*\"([A-Za-z0-9+/=]+)\";", line)
            if m:
                b64_str = m.group(1)
                return zlib.decompress(base64.b64decode(b64_str))
    raise ValueError(f"Não foi possível encontrar COORD_B64 em {html_path}")


def main():
    t0 = time.time()
    print("=== SINCRONIZADOR & BUILDER DO WEB APP 3D (1.34M DOCUMENTOS) ===")
    WEB_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Carregar metadados e histórico
    print("1. Lendo histórico de anotações e metadados...")
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    docs = meta["documents"]
    n_docs = len(docs)

    with open(HISTORY_PATH, "r", encoding="utf-8") as f:
        history = json.load(f)

    # Mapear classes
    known_classes = sorted(list(set(history.values())))
    class_to_id = {"Não Mapeado": 0}
    for idx, c in enumerate(known_classes, 1):
        class_to_id[c] = idx

    class_ids = np.zeros(n_docs, dtype=np.uint8)
    class_counts = {c: 0 for c in class_to_id}

    for i in range(n_docs):
        stem = Path(docs[i]["doc_name"]).stem
        lbl = history.get(stem)
        if lbl and lbl in class_to_id:
            cid = class_to_id[lbl]
            class_ids[i] = cid
            class_counts[lbl] += 1
        else:
            class_counts["Não Mapeado"] += 1

    print(f"   Anotações mapeadas: {sum(v for k, v in class_counts.items() if k != 'Não Mapeado'):,} em 15 classes")

    # 2. Gerar classes.bin.gz
    classes_raw = class_ids.tobytes()
    classes_gz = gzip.compress(classes_raw, compresslevel=6)
    classes_path = WEB_DATA_DIR / "classes.bin.gz"
    with open(classes_path, "wb") as f:
        f.write(classes_gz)
    print(f"   -> Salvo: {classes_path.name} ({len(classes_gz) / 1024:.1f} KB)")

    # 3. Gerar coordenadas para os 4 modelos
    model_sources = {
        "dino": DATA_DIR / "latent_vectors_dino_1_4m_3d.html",
        "colpali": DATA_DIR / "latent_vectors_colpali_1_4m_3d.html",
        "combined": DATA_DIR / "latent_vectors_full_3d.html",
        "metric": DATA_DIR / "latent_vectors_metric_learned_1_4m_3d.html",
    }

    print("\n2. Empacotando coordenadas 3D compactadas para os 4 modelos...")
    for model_key, html_path in model_sources.items():
        dst_path = WEB_DATA_DIR / MODEL_METRICS[model_key]["file"]
        print(f"   Extraindo {model_key:10s} de {html_path.name}...")
        raw_coords = extract_coords_from_html(html_path)
        assert len(raw_coords) == n_docs * 3 * 4, f"Tamanho incorreto para {model_key}: {len(raw_coords)}"
        
        gz_coords = gzip.compress(raw_coords, compresslevel=6)
        with open(dst_path, "wb") as f:
            f.write(gz_coords)
        print(f"   -> Salvo: {dst_path.name} ({len(gz_coords) / (1024*1024):.2f} MB)")

    # 4. Gerar metadata.json
    print("\n3. Gerando metadata.json...")
    client_palette = {}
    for name, cid in class_to_id.items():
        pal = CLASS_PALETTE.get(name, {"hex": "#38bdf8", "rgb": [56, 189, 248]})
        client_palette[cid] = {
            "name": name,
            "hex": pal["hex"],
            "rgb": [v / 255.0 for v in pal["rgb"]],
            "count": class_counts.get(name, 0),
        }

    dataset_stats = {
        "total_records": 3475760,
        "declared_pdfs": 1893880,
        "downloaded_pdfs": 1402334,
        "successful_docs": n_docs,
        "success_rate": f"{(n_docs / 1402334) * 100:.2f}%",
        "total_pages": len(meta.get("pages", [])) if "pages" in meta else 1966161,
        "avg_pages": 1.47,
        "single_page_docs": 1000539,
        "single_page_pct": "74.69%"
    }

    metadata = {
        "n_points": n_docs,
        "default_model": "metric",
        "models": MODEL_METRICS,
        "palette": client_palette,
        "dataset_stats": dataset_stats,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_annotated": sum(v for k, v in class_counts.items() if k != "Não Mapeado")
    }

    meta_out = WEB_DATA_DIR / "metadata.json"
    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"   -> Salvo: {meta_out.name} ({meta_out.stat().st_size / 1024:.1f} KB)")

    print(f"\n=== BUILD DO WEB APP CONCLUÍDO EM {time.time()-t0:.1f}s! ===")


if __name__ == "__main__":
    main()
