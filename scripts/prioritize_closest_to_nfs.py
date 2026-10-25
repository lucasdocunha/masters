#!/usr/bin/env python3
"""
Priorização de Amostras Mais Próximas ao Centroide das Notas Fiscais (NFs)
Calcula a similaridade de todos os documentos de página única não rotulados
em relação aos centroides das classes de Notas Fiscais (NFC, NFSe e NFe).
Remove tarefas não rotuladas antigas da fila, renderiza a 1ª página em PNG
e insere as 500 amostras mais representativas no topo da fila do Label Studio.
Em seguida, gera as predições do modelo ativo (v9) para todas as tarefas inseridas.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

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
    """Renderiza a primeira página do PDF para PNG se ainda não existir."""
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
    print("=== Reorganização da Fila do Label Studio: Amostras Mais Próximas às NFs ===")
    t0 = time.time()

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

    # Identificar documentos rotulados como Notas Fiscais
    nfc_stems = [s for s, l in history.items() if l == "nfc" and s in stem_to_idx]
    nfse_stems = [s for s, l in history.items() if l == "nfse" and s in stem_to_idx]
    all_nf_stems = [s for s, l in history.items() if l in ["nfc", "nfe", "nfse"] and s in stem_to_idx]

    print(f"Documentos anotados da família NF: {len(all_nf_stems)} (NFC={len(nfc_stems)}, NFSe={len(nfse_stems)})")

    if not all_nf_stems:
        print("Erro: Nenhum documento de NF encontrado no histórico.")
        sys.exit(1)

    # Calcular centroides normalizados
    c_all_nf = np.mean(combined[[stem_to_idx[s] for s in all_nf_stems]], axis=0)
    c_all_nf /= np.linalg.norm(c_all_nf)

    c_nfc = np.mean(combined[[stem_to_idx[s] for s in nfc_stems]], axis=0)
    c_nfc /= np.linalg.norm(c_nfc)

    c_nfse = np.mean(combined[[stem_to_idx[s] for s in nfse_stems]], axis=0)
    c_nfse /= np.linalg.norm(c_nfse)

    # 3. Similaridade máxima com qualquer um dos centroides de NFs
    print("Calculando similaridade máxima com a família de Notas Fiscais (NFC, NFSe, NFe)...")
    sim_nfc = np.dot(combined, c_nfc)
    sim_nfse = np.dot(combined, c_nfse)
    sim_all = np.dot(combined, c_all_nf)
    best_sim = np.maximum(np.maximum(sim_nfc, sim_nfse), sim_all)

    # Lookup de metadados
    lookup: dict[str, Any] = {}
    if LOOKUP_FILE.exists():
        with open(LOOKUP_FILE, "r", encoding="utf-8") as f:
            lookup = json.load(f)

    # Filtrar candidatos: página única, não anotados, PDF existente no disco
    labeled_stems = set(history.keys())
    candidates = []

    for idx, d in enumerate(docs):
        if d.get("num_pages") != 1:
            continue
        doc_name = d.get("doc_name", "")
        stem = Path(doc_name).stem
        if stem in labeled_stems:
            continue
        pdf_path = PDF_DIR / doc_name
        if not pdf_path.exists():
            continue

        score = float(best_sim[idx])
        # Identificar tipo mais próximo
        s_nfc = float(sim_nfc[idx])
        s_nfse = float(sim_nfse[idx])
        s_all = float(sim_all[idx])
        tipo = "NFC" if s_nfc >= s_nfse and s_nfc >= s_all else ("NFSe" if s_nfse >= s_all else "NFe/Geral")

        candidates.append((score, doc_name, stem, pdf_path, idx, tipo))

    # Ordenar decrescente pela similaridade (os mais próximos às NFs primeiro!)
    candidates.sort(key=lambda x: x[0], reverse=True)
    print(f"Total de candidatos válidos de 1 página: {len(candidates):,}")

    selected = candidates[:n_tasks]
    print(f"Selecionados top {len(selected)} mais próximos às NFs.")
    print(f"Maior similaridade: {selected[0][0]:.4f} ({selected[0][5]} - {selected[0][1]})")
    print(f"Menor similaridade no lote: {selected[-1][0]:.4f} ({selected[-1][5]} - {selected[-1][1]})")

    # 4. Renderizar páginas PNG
    print(f"Renderizando imagens PNG em {IMAGES_DIR}...")
    rendered_count = 0
    tasks_payload = []

    for rank, (score, doc_name, stem, pdf_path, idx, tipo) in enumerate(selected, 1):
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
                "sim_nf": f"{score:.3f}",
                "tipo_nf_sugerido": tipo,
                "prioridade_rank": rank
            }
        })

    print(f"Renderização concluída ({rendered_count} novas imagens geradas).")

    # 5. Limpar tarefas não rotuladas antigas no Label Studio (preservando todas as rotuladas)
    print("Limpando tarefas não rotuladas antigas no Label Studio...")
    os.system(
        """docker exec -w /label-studio/label_studio -e DJANGO_SETTINGS_MODULE=core.settings.label_studio """
        """ceap-label-studio python3 -c '"""
        """import django; django.setup(); """
        """from tasks.models import Task; """
        """deleted, _ = Task.objects.filter(project_id=1, annotations__isnull=True).delete(); """
        """print(f"[OK] Removidas {deleted} tarefas não rotuladas antigas.")'"""
    )

    # 6. Conectar ao Label Studio e enviar novo lote em ordem de prioridade
    print("Conectando ao Label Studio...")
    s = requests.Session()
    s.get(f"{LABEL_STUDIO_URL}/user/login/")
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

    # 7. Disparar predições do modelo ativo (v9) para a nova fila
    print("Disparando predições do modelo ativo para as novas tarefas...")
    pred_res = s.post(
        f"{LABEL_STUDIO_URL}/api/dm/actions?project={PROJECT_ID}&id=retrieve_tasks_predictions",
        json={"selectedItems": {"all": True, "included": []}},
        headers=headers,
        timeout=180
    )
    print(f"[OK] Sincronização de predições: status {pred_res.status_code}")

    # 8. Garantir model_version no projeto
    s.patch(f"{LABEL_STUDIO_URL}/api/projects/{PROJECT_ID}", json={"model_version": "v9"}, headers=headers)

    print(f"\n[SUCESSO] Fila reorganizada com sucesso em {time.time()-t0:.2f}s!")
    print("As amostras mais próximas às Notas Fiscais (NFC, NFSe e NFe) já estão prontas para rotulagem!")


if __name__ == "__main__":
    main(500)
