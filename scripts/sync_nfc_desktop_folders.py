#!/usr/bin/env python3
"""
Sincroniza em tempo real as pastas do Desktop a partir do checkpoint do OCR massivo.
Permite ao usuário abrir e inspecionar os lotes no Desktop enquanto o OCR roda em background.
"""

import re
import json
from pathlib import Path
from collections import Counter

def sync_desktop():
    root_dir = Path("/home/lucas.ocunha/masters")
    checkpoint_file = root_dir / "data" / "nfc_27k_ocr_checkpoint.json"
    rendered_dir = root_dir / "data" / "rendered_pages"
    output_base = Path("/home/lucas.ocunha/Desktop/validacao_nfc_todos_27k")
    
    if not checkpoint_file.exists():
        print("Checkpoint ainda não encontrado.")
        return
        
    with open(checkpoint_file, "r", encoding="utf-8") as f:
        results = json.load(f)
        
    lookup_file = root_dir / "data" / "pdf_metadata_lookup.json"
    metadata = {}
    if lookup_file.exists():
        with open(lookup_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            
    output_base.mkdir(parents=True, exist_ok=True)
    
    categories = [
        "1_com_cupom_fiscal_ecf",
        "2_com_danfe_nfce",
        "3_com_cfe_sat",
        "4_outros_sem_marcador"
    ]
    for cat in categories:
        (output_base / cat).mkdir(parents=True, exist_ok=True)
        
    cat_items = {cat: [] for cat in categories}
    for doc_id, val in results.items():
        cat = val[0] if isinstance(val, (list, tuple)) else val
        if cat in cat_items:
            cat_items[cat].append(doc_id)
        else:
            cat_items["4_outros_sem_marcador"].append(doc_id)
            
    created_links = 0
    for cat, doc_ids in cat_items.items():
        cat_dir = output_base / cat
        total = len(doc_ids)
        
        if total <= 2500:
            for i, d in enumerate(doc_ids, 1):
                m = metadata.get(f"{d}.pdf", {})
                ano = m.get("ano", 0)
                raw_forn = m.get("fornecedor", "")[:22]
                forn = re.sub(r'[^a-zA-Z0-9]', '_', raw_forn).strip('_') or "FORNECEDOR"
                forn = re.sub(r'_+', '_', forn)
                src = rendered_dir / f"{d}.png"
                if src.exists():
                    link = cat_dir / f"rank_{i:04d}_[{ano}]_[{forn}]_{d}.png"
                    if not link.exists():
                        link.symlink_to(src)
                        created_links += 1
        else:
            batch_size = 2000
            for b_idx in range(0, total, batch_size):
                batch_docs = doc_ids[b_idx:b_idx + batch_size]
                sub_name = f"lote_{b_idx//batch_size + 1:02d}_{len(batch_docs)}"
                sub_dir = cat_dir / sub_name
                sub_dir.mkdir(parents=True, exist_ok=True)
                for i, d in enumerate(batch_docs, 1):
                    m = metadata.get(f"{d}.pdf", {})
                    ano = m.get("ano", 0)
                    raw_forn = m.get("fornecedor", "")[:22]
                    forn = re.sub(r'[^a-zA-Z0-9]', '_', raw_forn).strip('_') or "FORNECEDOR"
                    forn = re.sub(r'_+', '_', forn)
                    src = rendered_dir / f"{d}.png"
                    if src.exists():
                        link = sub_dir / f"rank_{i:04d}_[{ano}]_[{forn}]_{d}.png"
                        if not link.exists():
                            link.symlink_to(src)
                            created_links += 1

    readme = f"""=============================================================================
CLASSIFICAÇÃO EM TEMPO REAL DOS {len(results):,} CUPONS NFC PROCESSADOS
=============================================================================

1. 1_com_cupom_fiscal_ecf/ ({len(cat_items['1_com_cupom_fiscal_ecf']):,} docs):
   - Documentos confirmados com 'CUPOM FISCAL', 'COO', 'CCF', 'ECF' ou marcas térmicas.
   - Particionados em sub-lotes de 2.000 para navegação fluida no Desktop.

2. 2_com_danfe_nfce/ ({len(cat_items['2_com_danfe_nfce']):,} docs):
   - DANFE NFC-e (Modelo 65).

3. 3_com_cfe_sat/ ({len(cat_items['3_com_cfe_sat']):,} docs):
   - CF-e-SAT (Modelo 59).

4. 4_outros_sem_marcador/ ({len(cat_items['4_outros_sem_marcador']):,} docs):
   - Documentos desbotados ou sem marcadores explícitos de cupom.
"""
    (output_base / "LEIAME.txt").write_text(readme, encoding="utf-8")
    print(f"Sincronização concluída! {created_links} novos atalhos criados em {output_base}")
    print("Contagem atual:")
    for cat, doc_ids in cat_items.items():
        pct = (len(doc_ids) / len(results)) * 100 if len(results) > 0 else 0
        print(f"  {cat:30s}: {len(doc_ids):5d} ({pct:5.1f}%)")

if __name__ == "__main__":
    sync_desktop()
