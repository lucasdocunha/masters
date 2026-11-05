#!/usr/bin/env python3
"""
Gera o dataset otimizado para o Plotly 3D WebGL (web/data/plotly_dataset.json.gz).
Extrai os 25.716 documentos do dataset de rotulação (21.427 anotados + 4.289 candidatos)
com suas coordenadas 3D correspondentes nos 4 espaços latentes:
- metric (ArcFace Deep Metric Learning 128-D)
- dino (DINOv2 Visual Puro 384-D)
- colpali (ColPali VLM Documental 128-D)
- combined (Multimodal DINO + ColPali 512-D)
"""

import json
import gzip
import time
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
WEB_DATA_DIR = BASE_DIR / "web" / "data"

def main():
    print("1. Carregando metadados e dataset de rotulação...")
    t0 = time.time()
    
    with open(WEB_DATA_DIR / "metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
        
    with gzip.open(WEB_DATA_DIR / "labeling_dataset.json.gz", "rb") as f:
        docs = json.loads(f.read().decode("utf-8"))
        
    print(f"   Carregados {len(docs):,} documentos em {time.time()-t0:.2f}s")
    coords_indices = np.array([d["coords_idx"] for d in docs], dtype=np.int32)
    
    models = ["metric", "dino", "colpali", "combined"]
    model_files = {
        "metric": WEB_DATA_DIR / "coords_metric.bin.gz",
        "dino": WEB_DATA_DIR / "coords_dino.bin.gz",
        "colpali": WEB_DATA_DIR / "coords_colpali.bin.gz",
        "combined": WEB_DATA_DIR / "coords_combined.bin.gz",
    }
    
    coords = {}
    print("\n2. Extraindo coordenadas 3D para os 4 modelos...")
    for m in models:
        t_m = time.time()
        with gzip.open(model_files[m], "rb") as f:
            all_coords = np.frombuffer(f.read(), dtype=np.float32).reshape(-1, 3)
        sub = all_coords[coords_indices].round(2)
        # Lista plana ou de trios [x, y, z]
        coords[m] = sub.tolist()
        print(f"   -> Modelo {m}: {len(coords[m]):,} pontos extraídos em {time.time()-t_m:.2f}s")
        
    payload = {
        "palette": meta.get("palette", {}),
        "models": meta.get("models", {}),
        "docs": docs,
        "coords": coords
    }
    
    output_path = WEB_DATA_DIR / "plotly_dataset.json.gz"
    print(f"\n3. Salvando e comprimindo em {output_path}...")
    json_bytes = json.dumps(payload, separators=(',', ':')).encode("utf-8")
    compressed = gzip.compress(json_bytes, compresslevel=6)
    output_path.write_bytes(compressed)
    
    size_mb = len(compressed) / (1024 * 1024)
    print(f"   Sucesso! Arquivo gerado: {size_mb:.2f} MB ({len(compressed):,} bytes).")
    print(f"   Tempo total: {time.time()-t0:.2f}s")

if __name__ == "__main__":
    main()
