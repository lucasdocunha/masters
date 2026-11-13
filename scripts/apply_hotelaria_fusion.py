#!/usr/bin/env python3
"""
Aplica a fusão da classe 'hotelaria':
- Lê o resumo da auditoria consolidada (data/evaluation/hotelaria_fusion_summary.json)
- Migra para 'nfse' os 685 confirmados
- Descarta os 26 recibos/condomínios avulsos identificados
- Atualiza annotations_history.json e data/rotulos/
- Atualiza o banco PostgreSQL
"""

import sys
import os
import json
import gzip
import psycopg2
from pathlib import Path

def main():
    root_dir = Path(__file__).resolve().parent.parent
    summary_path = root_dir / "data" / "evaluation" / "hotelaria_fusion_summary.json"
    if not summary_path.exists():
        print(f"Erro: {summary_path} não encontrado!")
        sys.exit(1)
        
    with open(summary_path) as f:
        summary = json.load(f)
        
    drop_ids = set(summary["drop_ids"])
    print(f"Total de drops cadastrados: {len(drop_ids)}")
    
    hist_path = root_dir / "data" / "models" / "annotations_history.json"
    with open(hist_path) as f:
        history = json.load(f)
        
    hotel_ids = [k for k, v in history.items() if v == "hotelaria"]
    print(f"Documentos atualmente marcados como 'hotelaria': {len(hotel_ids)}")
    
    nfse_migrated = []
    dropped = []
    
    for did in hotel_ids:
        if did in drop_ids:
            del history[did]
            dropped.append(did)
        else:
            history[did] = "nfse"
            nfse_migrated.append(did)
            
    print(f"Migrados para 'nfse': {len(nfse_migrated)}")
    print(f"Descartados (removidos das anotações): {len(dropped)}")
    
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"annotations_history.json salvo. Total de anotações ativas: {len(history):,}")
    
    # Atualizar data/rotulos/
    sys.path.insert(0, str(root_dir / "src"))
    from dados.labeling.labels_csv import read_label_rows, write_labels_dir
    rotulos_dir = root_dir / "data" / "rotulos"
    rows = read_label_rows(rotulos_dir)
    print(f"Linhas lidas em data/rotulos: {len(rows)}")
    
    for did in nfse_migrated:
        if did in rows:
            rows[did][-2] = "nfse"
    for did in dropped:
        if did in rows:
            del rows[did]
    for did, row in list(rows.items()):
        if row[-2] == "hotelaria":
            del rows[did]
            
    total_written = write_labels_dir(rotulos_dir, rows.values())
    print(f"data/rotulos/ gravado com sucesso: {total_written} linhas.")
    
    # Atualizar PostgreSQL
    try:
        conn = psycopg2.connect("postgresql://postgres:postgres@localhost:5432/ceap")
        cur = conn.cursor()
        
        int_nfse = [int(x) for x in nfse_migrated]
        int_drop = [int(x) for x in dropped]
        
        cur.execute("UPDATE documento SET classe_predita = 'nfse' WHERE ide_documento = ANY(%s)", (int_nfse,))
        print(f"DB: {cur.rowcount} documentos atualizados para 'nfse'.")
        
        cur.execute("UPDATE documento SET classe_predita = NULL WHERE ide_documento = ANY(%s) OR classe_predita = 'hotelaria'", (int_drop,))
        print(f"DB: {cur.rowcount} documentos redefinidos para NULL.")
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Aviso no banco: {e}")
        
    print("\nFusão de dados concluída!")

if __name__ == "__main__":
    main()
