#!/home/lucas.ocunha/masters/.venv_ocr/bin/python3
"""
Confirma os documentos validados da pasta 1_danfe_nfe como classe 'nfe':
1. Audita os 397 arquivos mantidos pelo usuário em 1_danfe_nfe/ e registra os 62 descartados.
2. Registra os 397 documentos em annotations_history.json como 'nfe'.
3. Sincroniza a classe 'nfe' no PostgreSQL (tabela 'documento').
4. Salva registro dos 62 descartados em danfe_rejeitados_pelo_usuario.json.
5. Atualiza manifest.json e remove a pasta validada 1_danfe_nfe/.
"""

import os
import sys
import json
import shutil
from pathlib import Path
from collections import Counter
import psycopg

BASE_DIR = Path("/home/lucas.ocunha/masters")
DATA_DIR = BASE_DIR / "data"
VALIDATION_DIR = DATA_DIR / "validacao_batches" / "validacao_falsos_positivos_ocr"
DANFE_DIR = VALIDATION_DIR / "1_danfe_nfe"
HISTORY_FILE = BASE_DIR / "data" / "models" / "annotations_history.json"
YEAR_FILE = DATA_DIR / "pdf_to_year.json"
DB_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/ceap")


def main():
    print("=" * 72)
    print(" CONFIRMAÇÃO DAS DANFE (NF-e MOD. 55) VALIDADAS PELO USUÁRIO")
    print("=" * 72)

    if not DANFE_DIR.exists():
        print(f"[ERRO] Pasta {DANFE_DIR} não encontrada!")
        sys.exit(1)

    # 1. Mapear arquivos mantidos na pasta 1_danfe_nfe
    danfe_files = list(DANFE_DIR.glob("*.png"))
    danfe_stems = set(p.stem for p in danfe_files)
    print(f"\n1. Arquivos mantidos pelo usuário: {len(danfe_stems):,} imagens PNG.")

    # Carregar manifest.json para identificar os removidos
    manifest_p = VALIDATION_DIR / "manifest.json"
    removed_items = []
    if manifest_p.exists():
        with open(manifest_p, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        for item in manifest:
            if item.get("subpasta") == "1_danfe_nfe":
                doc_id = str(item.get("doc_id"))
                if doc_id not in danfe_stems:
                    removed_items.append(item)
                    item["status_validacao"] = "rejeitado_pelo_usuario"
                else:
                    item["status_validacao"] = "confirmado_classe_nfe"

        with open(manifest_p, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"   • Documentos descartados manualmente pelo usuário: {len(removed_items):,}")

    # Salvar log dos descartados
    rejeitados_p = VALIDATION_DIR / "danfe_rejeitados_pelo_usuario.json"
    with open(rejeitados_p, "w", encoding="utf-8") as f:
        json.dump(removed_items, f, indent=2, ensure_ascii=False)
    print(f"   ✓ Registro dos {len(removed_items):,} descartados salvo em {rejeitados_p.name}")

    # 2. Atualizar annotations_history.json com classe 'nfe'
    print(f"\n2. Atualizando histórico de anotações em {HISTORY_FILE.name}...")
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        history = json.load(f)

    prev_total = len(history)
    prev_nfe = sum(1 for v in history.values() if v == "nfe")

    added_count = 0
    for stem in danfe_stems:
        if stem not in history or history[stem] != "nfe":
            history[stem] = "nfe"
            added_count += 1

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    new_total = len(history)
    new_nfe = sum(1 for v in history.values() if v == "nfe")

    print(f"   ✓ NF-e anteriores no histórico: {prev_nfe:,}")
    print(f"   ✓ Novas NF-e adicionadas: {added_count:,}")
    print(f"   ✓ Novo total de NF-e (DANFE): {new_nfe:,}")
    print(f"   ✓ Novo total geral de anotações: {new_total:,}")

    # 3. Sincronizar com PostgreSQL
    print(f"\n3. Sincronizando com PostgreSQL ({DB_URL})...")
    with open(YEAR_FILE, "r", encoding="utf-8") as f:
        pdf_to_year = json.load(f)

    try:
        with psycopg.connect(DB_URL) as conn:
            with conn.cursor() as cur:
                rows_to_upsert = []
                for stem in danfe_stems:
                    doc_id = int(stem)
                    ano = pdf_to_year.get(f"{stem}.pdf", 2024)
                    caminho = f"/media/ssd2/datasets/CEAP/{stem}.pdf"
                    rows_to_upsert.append((doc_id, ano, caminho, "baixado", "nfe"))

                batch_size = 500
                for i in range(0, len(rows_to_upsert), batch_size):
                    chunk = rows_to_upsert[i:i + batch_size]
                    cur.executemany("""
                        INSERT INTO documento (ide_documento, ano, caminho_arquivo, status, classe_predita)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (ide_documento) DO UPDATE SET
                            status = 'baixado',
                            classe_predita = 'nfe';
                    """, chunk)
                conn.commit()
        print(f"   ✓ {len(rows_to_upsert):,} documentos atualizados na tabela 'documento' com classe_predita='nfe'.")
    except Exception as e:
        print(f"   [AVISO] Erro na sincronização com PostgreSQL: {e}")

    # 4. Remover pasta validada 1_danfe_nfe
    print(f"\n4. Removendo a pasta validada ({DANFE_DIR.name})...")
    shutil.rmtree(DANFE_DIR)
    print("   ✓ Pasta 1_danfe_nfe/ removida com sucesso após validação.")

    # 5. Situação da pasta validacao_falsos_positivos_ocr
    remaining_subdirs = [d.name for d in VALIDATION_DIR.iterdir() if d.is_dir()]
    print(f"\nPastas restantes em validacao_falsos_positivos_ocr/:")
    for sub in remaining_subdirs:
        cnt = len(list((VALIDATION_DIR / sub).glob("*.png")))
        print(f"  • {sub:20s}: {cnt:,} documentos")

    print("\n" + "=" * 72)
    print(" CONFIRMAÇÃO DE DANFE CONCLUÍDA COM SUCESSO!")
    print(f"Total de NF-e no Dataset: {new_nfe:,} documentos")
    print(f"Total Geral Rotulado   : {new_total:,} documentos")
    print("=" * 72)


if __name__ == "__main__":
    main()
