#!/usr/bin/env python3
"""
Remove a classe fantasma 'emails' (1 único documento 8145516):
- Remove do annotations_history.json
- Remove de data/rotulos/
- Atualiza PostgreSQL
"""

import sys
import json
import psycopg2
from pathlib import Path

def main():
    root_dir = Path(__file__).resolve().parent.parent
    
    # 1. annotations_history.json
    hist_path = root_dir / "data" / "models" / "annotations_history.json"
    with open(hist_path) as f:
        history = json.load(f)
        
    emails_keys = [k for k, v in history.items() if v == "emails"]
    print(f"Chaves com 'emails' em annotations_history: {emails_keys}")
    for k in emails_keys:
        del history[k]
        
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"annotations_history.json atualizado. Total de anotações ativas: {len(history):,}")
    
    # 2. data/rotulos/
    sys.path.insert(0, str(root_dir / "src"))
    from dados.labeling.labels_csv import read_label_rows, write_labels_dir
    rotulos_dir = root_dir / "data" / "rotulos"
    rows = read_label_rows(rotulos_dir)
    print(f"Linhas em data/rotulos: {len(rows)}")
    
    deleted_rows = 0
    for k in emails_keys:
        if k in rows:
            del rows[k]
            deleted_rows += 1
            
    for did, row in list(rows.items()):
        if row[-2] == "emails":
            del rows[did]
            deleted_rows += 1
            
    total_written = write_labels_dir(rotulos_dir, rows.values())
    print(f"data/rotulos/ regravado: {total_written} linhas (removidos {deleted_rows}).")
    
    # 3. PostgreSQL
    try:
        conn = psycopg2.connect("postgresql://postgres:postgres@localhost:5432/ceap")
        cur = conn.cursor()
        cur.execute("UPDATE documento SET classe_predita = NULL WHERE ide_documento = 8145516 OR classe_predita = 'emails'")
        print(f"DB: {cur.rowcount} registros atualizados para NULL.")
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Aviso DB: {e}")
        
    print("\nClasse 'emails' removida com sucesso!")

if __name__ == "__main__":
    main()
