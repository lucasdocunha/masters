#!/usr/bin/env python3
"""
Priorização de Amostras Mais Distantes do Cluster NFC (Active Learning)
Calcula o centroide dos documentos rotulados como NFC, mede a distância cosseno
de todos os documentos não rotulados de página única, renderiza as páginas em PNG
e insere no topo da fila do Label Studio em ordem decrescente de distância.
"""

import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import pymupdf
import requests

DATA_DIR = Path("/home/lucas.ocunha/masters/data")
PDF_DIR = Path("/media/ssd2/datasets/CEAP")
IMAGES_DIR = DATA_DIR / "rendered_pages"
HISTORY_FILE = Path("/home/lucas.ocunha/masters/ml_backend/models/annotations_history.json")
NPZ_FILE = DATA_DIR / "latent_vectors_sample_5pct.npz"
META_FILE = DATA_DIR / "latent_vectors_sample_5pct_metadata.json"
LOOKUP_FILE = DATA_DIR / "pdf_metadata_lookup.json"

LABEL_STUDIO_URL = "http://localhost:8085"
LABEL_STUDIO_EMAIL = "admin@ceap.com"
LABEL_STUDIO_PASS = "ceap-admin-2026"
PROJECT_ID = 1


def render_page(pdf_path: Path, output_png: Path, dpi: int = 150) -> bool:
    if output_png.exists() and output_png.stat().st_size > 0:
        return True
    try:
        doc = pymupdf.open(pdf_path)
        if len(doc) == 0:
            doc.close()
            return False
        page = doc[0]
        pix = page.get_pixmap(dpi=dpi)
        output_png.parent.mkdir(parents=True, exist_ok=True)
        pix.save(str(output_png))
        doc.close()
        return True
    except Exception as e:
        print(f"  [Aviso] Erro renderizando {pdf_path.name}: {e}")
        return False


def main(n_tasks: int = 500):
    print("=== Priorização de Documentos mais Distantes de NFC ===")
    
    # 1. Carregar embeddings e metadados
    print(f"Carregando embeddings de {NPZ_FILE}...")
    npz = np.load(NPZ_FILE)
    combined = np.hstack([npz["dino_doc_embeddings"], npz["vlm_doc_embeddings"]]).astype(np.float32)
    norms = np.linalg.norm(combined, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    combined /= norms

    print(f"Carregando metadados de {META_FILE}...")
    with open(META_FILE, "r", encoding="utf-8") as f:
        docs = json.load(f).get("documents", [])

    stem_to_idx = {Path(d.get("doc_name", "")).stem: idx for idx, d in enumerate(docs)}

    # 2. Carregar histórico de anotações
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        history = json.load(f)

    nfc_stems = [stem for stem, label in history.items() if label == "nfc" and stem in stem_to_idx]
    print(f"Encontrados {len(nfc_stems)} documentos NFC rotulados para formar o cluster.")

    if not nfc_stems:
        print("Erro: Nenhum NFC rotulado encontrado no histórico.")
        sys.exit(1)

    nfc_indices = [stem_to_idx[s] for s in nfc_stems]
    nfc_vectors = combined[nfc_indices]
    centroid_nfc = np.mean(nfc_vectors, axis=0)
    centroid_nfc /= np.linalg.norm(centroid_nfc)

    # 3. Calcular distâncias para todos os documentos de página única não rotulados
    print("Calculando distâncias cosseno em relação ao centroide NFC...")
    sims = np.dot(combined, centroid_nfc)
    dists = 1.0 - sims

    # Carregar lookup de fornecedor/subcota
    lookup = {}
    if LOOKUP_FILE.exists():
        print(f"Carregando lookup de {LOOKUP_FILE}...")
        with open(LOOKUP_FILE, "r", encoding="utf-8") as f:
            lookup = json.load(f)

    # Filtrar candidatos
    candidates = []
    for idx, d in enumerate(docs):
        if d.get("num_pages") != 1:
            continue
        doc_name = d.get("doc_name", "")
        stem = Path(doc_name).stem
        if stem in history:
            continue
        pdf_path = PDF_DIR / doc_name
        if not pdf_path.exists():
            continue
        candidates.append((idx, dists[idx], doc_name, stem, pdf_path))

    # Ordenar decrescente pela distância (os mais distantes primeiro!)
    candidates.sort(key=lambda x: x[1], reverse=True)
    print(f"Total de candidatos válidos de 1 página: {len(candidates):,}")

    selected = candidates[:n_tasks]
    print(f"Selecionados top {len(selected)} mais distantes.")
    print(f"Distância do 1º mais distante: {selected[0][1]:.4f}")
    print(f"Distância do último selecionado: {selected[-1][1]:.4f}")

    # 4. Renderizar páginas PNG
    print(f"Renderizando imagens PNG em {IMAGES_DIR}...")
    t0 = time.time()
    rendered_count = 0
    tasks_payload = []

    for rank, (idx, dist, doc_name, stem, pdf_path) in enumerate(selected, 1):
        out_png = IMAGES_DIR / f"{stem}.png"
        if not out_png.exists():
            if render_page(pdf_path, out_png):
                rendered_count += 1

        meta = lookup.get(doc_name, {})
        subcota = meta.get("subcota", "NÃO INFORMADA")
        fornecedor = meta.get("fornecedor", "NÃO INFORMADO")
        ano = meta.get("ano", docs[idx].get("year", "N/A"))

        tasks_payload.append({
            "data": {
                "image": f"/data/local-files/?d=images/{stem}.png",
                "pdf_url": f"/data/local-files/?d=pdfs/{doc_name}",
                "doc_name": doc_name,
                "subcota_oficial": subcota,
                "fornecedor_oficial": fornecedor,
                "ano": ano,
                "dist_nfc": f"{dist:.3f}",
                "prioridade_rank": rank
            }
        })

    print(f"Renderização concluída em {time.time()-t0:.2f}s ({rendered_count} novas imagens).")

    # 5. Enviar em lotes para o Label Studio via API
    print("Conectando ao Label Studio...")
    s = requests.Session()
    r = s.get(f"{LABEL_STUDIO_URL}/user/login/")
    csrftoken = s.cookies.get("csrftoken")
    login_res = s.post(
        f"{LABEL_STUDIO_URL}/user/login/",
        data={"email": LABEL_STUDIO_EMAIL, "password": LABEL_STUDIO_PASS, "csrfmiddlewaretoken": csrftoken},
        headers={"Referer": f"{LABEL_STUDIO_URL}/user/login/"}
    )
    if login_res.status_code != 200:
        print("Erro ao autenticar no Label Studio.")
        sys.exit(1)

    csrftoken = s.cookies.get("csrftoken")
    headers = {"X-CSRFToken": csrftoken, "Referer": f"{LABEL_STUDIO_URL}/"}

    # Inserir em lotes de 100
    batch_size = 100
    total_inserted = 0
    for i in range(0, len(tasks_payload), batch_size):
        batch = tasks_payload[i:i+batch_size]
        res = s.post(f"{LABEL_STUDIO_URL}/api/projects/{PROJECT_ID}/tasks/bulk", json=batch, headers=headers)
        if res.status_code == 201:
            total_inserted += len(batch)
            print(f"  Inseridas {total_inserted}/{len(tasks_payload)} tarefas...")
        else:
            print(f"  Aviso: falha no lote {i}: {res.status_code} - {res.text}")

    print(f"[OK] Total de {total_inserted} tarefas priorizadas inseridas com sucesso!")


if __name__ == "__main__":
    main(500)
