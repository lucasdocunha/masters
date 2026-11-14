#!/usr/bin/env python3
"""
Atualiza web/data/review/nfe.json.gz e web/data/review/index.json com os 16.654 candidatos limpos
de NF-e (Modelo 55 / DANFE), excluindo qualquer NFC-e e NFS-e.
Permite rotular diretamente pelo site em https://ceap-latent-space-3d.vercel.app/#/pasta/nfe
e https://ceap-latent-space-3d.vercel.app/#/rotular/nfe
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

MANIFEST_FILE = DATA_DIR / "validacao_batches" / "validacao_nfe_novos_candidatos" / ".candidatos_manifesto.json"
COMPLETO_CSV = DATA_DIR / "ceap_documentos_completo.csv"
REVIEW_DIR = WEB_DIR / "data" / "review"
NFE_QUEUE_FILE = REVIEW_DIR / "nfe.json.gz"
INDEX_FILE = REVIEW_DIR / "index.json"
META_DIR = WEB_DIR / "data" / "latent" / "meta"


def main():
    t0 = time.time()
    print("=" * 70)
    print("ATUALIZANDO FILA DE REVISÃO WEB DA NF-E (web/data/review/nfe.json.gz)")
    print("=" * 70)

    # 1. Carregar candidatos limpos
    with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    print(f"Total de candidatos no manifesto: {len(manifest):,}")

    target_ids = set(manifest.keys())

    # 2. Carregar metadados completos do CSV
    print("Carregando metadados fiscais e URLs do ceap_documentos_completo.csv...")
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

    print(f"Metadados encontrados: {len(docs_info):,} / {len(target_ids):,}")

    # 3. Mapear pontos no espaço latente 3D do site
    print("Mapeando índices de pontos do espaço latente...")
    points = load_points(META_DIR)
    doc_to_ponto = {}
    for i, p in enumerate(points):
        if p and p[0]:
            doc_to_ponto[str(p[0])] = i

    # 4. Ordenar candidatos: Tier 1 (SEFAZ/DANFE) primeiro, depois maior similaridade desc
    tier1 = []
    tier2 = []
    for d_id, c in manifest.items():
        if c.get("source") == "sefaz_danfe":
            tier1.append(c)
        else:
            tier2.append(c)

    tier1.sort(key=lambda x: -float(x.get("sim", 1.0)))
    tier2.sort(key=lambda x: -float(x.get("sim", 0.0)))
    ordered_cands = tier1 + tier2
    print(f"Candidatos ordenados: {len(tier1):,} SEFAZ/DANFE + {len(tier2):,} alta métrica = {len(ordered_cands):,} total")

    # 5. Montar linhas para o web app
    candidatos_rows = []
    for c in ordered_cands:
        d_id = str(c["doc_id"])
        meta = docs_info.get(d_id, {})
        ponto = doc_to_ponto.get(d_id, -1)
        sim = round(float(c.get("sim", 1.0)), 4)
        forn = meta.get("fornecedor") or c.get("fornecedor", "")
        subcota = meta.get("subcota", "")
        ano = meta.get("ano") or 2024
        url = meta.get("url", "")
        segunda = "cupom_fiscal"
        margem = round(float(c.get("diff", 0.45)), 4)

        row = [
            d_id,
            ponto,
            sim,
            "",          # atual (vazio pois é candidato novo)
            segunda,     # segunda classe mais próxima
            margem,      # margem
            forn,
            subcota,
            ano,
            url
        ]
        candidatos_rows.append(row)

    # 6. Carregar fila existente para preservar a auditoria
    existing_auditoria = []
    if NFE_QUEUE_FILE.exists():
        with gzip.open(NFE_QUEUE_FILE, "rt", encoding="utf-8") as f:
            old_data = json.load(f)
            existing_auditoria = old_data.get("auditoria", [])

    # Contagem de rotulados
    with open(DATA_DIR / "models" / "annotations_history.json", "r", encoding="utf-8") as f:
        history = json.load(f)
    nfe_rotulados = sum(1 for v in history.values() if v == "nfe")

    payload = {
        "classe": "nfe",
        "campos": FIELDS,
        "rotulados": nfe_rotulados,
        "candidatos": candidatos_rows,
        "auditoria": existing_auditoria
    }

    # Salvar gzip
    raw_bytes = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    compressed = gzip.compress(raw_bytes, 9, mtime=0)
    NFE_QUEUE_FILE.write_bytes(compressed)
    print(f"[OK] {NFE_QUEUE_FILE.name} gerado: {len(candidatos_rows):,} candidatos ({len(compressed)/1024:.1f} KB gzip)")

    # 7. Atualizar web/data/review/index.json
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

    print(f"Concluído com sucesso em {time.time()-t0:.2f}s!")


if __name__ == "__main__":
    main()
