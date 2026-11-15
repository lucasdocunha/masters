#!/usr/bin/env python3
"""
Atualiza web/data/review/nfe.json.gz e web/data/review/index.json priorizando
os candidatos com termo DANFE mais similares ao novo centróide ArcFace,
removendo qualquer documento já rotulado.
"""

import csv
import gzip
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np

from dados.config import DATA_DIR, ROOT_DIR, WEB_DIR
from dados.labeling.review_queues import FIELDS, load_points

MANIFEST_20K = DATA_DIR / "validacao_batches" / "validacao_nfe_meta_20k" / ".manifesto_meta_20k.json"
COMPLETO_CSV = DATA_DIR / "ceap_documentos_completo.csv"
REVIEW_DIR = WEB_DIR / "data" / "review"
NFE_QUEUE_FILE = REVIEW_DIR / "nfe.json.gz"
INDEX_FILE = REVIEW_DIR / "index.json"
META_DIR = WEB_DIR / "data" / "latent" / "meta"
HISTORY_FILE = DATA_DIR / "models" / "annotations_history.json"


def main():
    t0 = time.time()
    print("=" * 70)
    print("ATUALIZANDO FILA WEB COM CANDIDATOS DANFE META 20K")
    print("=" * 70)

    # 1. Carregar histórico para garantir zero duplicatas
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        history = json.load(f)
    norm_hist = {k.replace(".pdf", ""): v for k, v in history.items()}
    nfe_rotulados = sum(1 for v in norm_hist.values() if v == "nfe")
    print(f"NF-e já rotuladas no histórico: {nfe_rotulados:,}")

    # 2. Carregar manifesto da meta 20k
    with open(MANIFEST_20K, "r", encoding="utf-8") as f:
        meta_20k = json.load(f)
    print(f"Candidatos DANFE meta 20k: {len(meta_20k):,}")

    target_ids = set(meta_20k.keys()) - set(norm_hist.keys())

    # 3. Metadados fiscais e URLs
    docs_info: Dict[str, dict] = {}
    with open(COMPLETO_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            d_id = row["doc_id"]
            if d_id in target_ids:
                try:
                    meta_obj = json.loads(row["metadados_json"])
                except Exception:
                    meta_obj = {}
                docs_info[d_id] = {
                    "url": row["link_documento"],
                    "ano": int(row["ano"]) if row["ano"].isdigit() else 2024,
                    "fornecedor": meta_obj.get("fornecedor", ""),
                    "subcota": meta_obj.get("subcota", "")
                }

    # 4. Mapear pontos no espaço latente
    points = load_points(META_DIR)
    doc_to_ponto = {}
    for i, p in enumerate(points):
        if p and p[0]:
            doc_to_ponto[str(p[0])] = i

    # 5. Ordenar por similaridade decrescente
    ordered_items = sorted(
        [v for k, v in meta_20k.items() if k in target_ids],
        key=lambda x: -float(x.get("sim", 0.0))
    )

    candidatos_rows = []
    for it in ordered_items:
        d_id = str(it["doc_id"])
        meta = docs_info.get(d_id, {})
        ponto = doc_to_ponto.get(d_id, -1)
        sim = round(float(it.get("sim", 1.0)), 4)
        forn = meta.get("fornecedor") or it.get("fornecedor", "")
        subcota = meta.get("subcota", "")
        ano = meta.get("ano") or 2024
        url = meta.get("url", "")
        segunda = "cupom_fiscal"
        margem = round(float(it.get("sim", 1.0) - 0.5), 4)

        row = [
            d_id,
            ponto,
            sim,
            "",
            segunda,
            margem,
            forn,
            subcota,
            ano,
            url
        ]
        candidatos_rows.append(row)

    # 6. Preservar auditoria existente
    existing_auditoria = []
    if NFE_QUEUE_FILE.exists():
        with gzip.open(NFE_QUEUE_FILE, "rt", encoding="utf-8") as f:
            old_data = json.load(f)
            existing_auditoria = old_data.get("auditoria", [])

    payload = {
        "classe": "nfe",
        "campos": FIELDS,
        "rotulados": nfe_rotulados,
        "candidatos": candidatos_rows,
        "auditoria": existing_auditoria
    }

    raw_bytes = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    compressed = gzip.compress(raw_bytes, 9, mtime=0)
    NFE_QUEUE_FILE.write_bytes(compressed)
    print(f"[OK] {NFE_QUEUE_FILE.name} gerado: {len(candidatos_rows):,} candidatos ({len(compressed)/1024:.1f} KB gzip)")

    # 7. Atualizar index.json
    if INDEX_FILE.exists():
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            idx_data = json.load(f)
        for c_entry in idx_data.get("classes", []):
            if c_entry.get("nome") == "nfe":
                c_entry["rotulados"] = nfe_rotulados
                c_entry["candidatos"] = len(candidatos_rows)
                c_entry["auditoria"] = len(existing_auditoria)
        idx_data["gerado"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        INDEX_FILE.write_text(json.dumps(idx_data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        print(f"[OK] {INDEX_FILE.name} atualizado.")

    print(f"Concluído em {time.time()-t0:.2f}s!")


if __name__ == "__main__":
    main()
