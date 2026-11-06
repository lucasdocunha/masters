#!/home/lucas.ocunha/masters/.venv_ocr/bin/python3
"""
Confirma os 199 Boletos Identificados pelo OCR como classe 'boleto' no histórico:
1. Registra os 199 documentos em annotations_history.json como 'boleto'.
2. Sincroniza a classe 'boleto' no PostgreSQL na tabela 'documento'.
3. Atualiza o manifest.json em validacao_falsos_positivos_ocr/.
4. Remove a pasta validada '2_boletos' da Área de Trabalho.
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
BOLETOS_DIR = VALIDATION_DIR / "2_boletos"
HISTORY_FILE = BASE_DIR / "ml_backend" / "models" / "annotations_history.json"
YEAR_FILE = DATA_DIR / "pdf_to_year.json"
DB_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/ceap")


def main():
    print("=" * 72)
    print(" CONFIRMAÇÃO DOS 199 DOCUMENTOS NA CLASSE 'BOLETO'")
    print("=" * 72)

    if not BOLETOS_DIR.exists():
        print(f"[ERRO] Pasta {BOLETOS_DIR} não encontrada!")
        sys.exit(1)

    # 1. Mapear arquivos PNG de boletos
    boleto_files = list(BOLETOS_DIR.glob("*.png"))
    print(f"\n1. Localizados {len(boleto_files):,} arquivos PNG em {BOLETOS_DIR.name}/")

    if not boleto_files:
        print("[ERRO] Nenhum arquivo PNG encontrado na pasta!")
        sys.exit(1)

    boleto_stems = [p.stem for p in boleto_files]

    # 2. Carregar e atualizar annotations_history.json
    print(f"\n2. Atualizando histórico de anotações em {HISTORY_FILE.name}...")
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        history = json.load(f)

    prev_total = len(history)
    prev_boletos = sum(1 for v in history.values() if v == "boleto")

    added_count = 0
    for stem in boleto_stems:
        if stem not in history or history[stem] != "boleto":
            history[stem] = "boleto"
            added_count += 1

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    new_total = len(history)
    new_boletos = sum(1 for v in history.values() if v == "boleto")

    print(f"   ✓ Boletos anteriores no histórico: {prev_boletos:,}")
    print(f"   ✓ Novos boletos adicionados: {added_count:,}")
    print(f"   ✓ Novo total de Boletos no dataset: {new_boletos:,}")
    print(f"   ✓ Novo total geral de anotações: {new_total:,}")

    # 3. Sincronizar com PostgreSQL
    print(f"\n3. Sincronizando com PostgreSQL ({DB_URL})...")
    with open(YEAR_FILE, "r", encoding="utf-8") as f:
        pdf_to_year = json.load(f)

    try:
        with psycopg.connect(DB_URL) as conn:
            with conn.cursor() as cur:
                rows_to_upsert = []
                for stem in boleto_stems:
                    doc_id = int(stem)
                    ano = pdf_to_year.get(f"{stem}.pdf", 2024)
                    caminho = f"/media/ssd2/datasets/CEAP/{stem}.pdf"
                    rows_to_upsert.append((doc_id, ano, caminho, "baixado", "boleto"))

                batch_size = 500
                for i in range(0, len(rows_to_upsert), batch_size):
                    chunk = rows_to_upsert[i:i + batch_size]
                    cur.executemany("""
                        INSERT INTO documento (ide_documento, ano, caminho_arquivo, status, classe_predita)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (ide_documento) DO UPDATE SET
                            status = 'baixado',
                            classe_predita = 'boleto';
                    """, chunk)
                conn.commit()
        print(f"   ✓ {len(rows_to_upsert):,} documentos atualizados na tabela 'documento' com classe_predita='boleto'.")
    except Exception as e:
        print(f"   [AVISO] Erro na sincronização com PostgreSQL: {e}")

    # 4. Atualizar manifest.json em validacao_falsos_positivos_ocr
    manifest_p = VALIDATION_DIR / "manifest.json"
    if manifest_p.exists():
        with open(manifest_p, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        boleto_set = set(boleto_stems)
        for item in manifest:
            if str(item.get("doc_id")) in boleto_set:
                item["status_validacao"] = "confirmado_classe_boleto"
        with open(manifest_p, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"   ✓ manifest.json atualizado com status 'confirmado_classe_boleto'.")

    # 5. Remover pasta validada 2_boletos
    print(f"\n4. Removendo a pasta validada ({BOLETOS_DIR.name})...")
    shutil.rmtree(BOLETOS_DIR)
    print("   ✓ Pasta 2_boletos/ removida com sucesso após validação.")

    # 6. Atualizar README.txt
    readme_p = VALIDATION_DIR / "README.txt"
    if readme_p.exists():
        remaining_subdirs = [d.name for d in VALIDATION_DIR.iterdir() if d.is_dir()]
        print(f"\nPastas restantes para validação em validacao_falsos_positivos_ocr/:")
        for sub in remaining_subdirs:
            cnt = len(list((VALIDATION_DIR / sub).glob("*.png")))
            print(f"  • {sub:20s}: {cnt:,} documentos")

    print("\n" + "=" * 72)
    print(" CONFIRMAÇÃO DE BOLETOS CONCLUÍDA COM SUCESSO!")
    print(f"Total de Boletos no Dataset: {new_boletos:,} documentos")
    print("=" * 72)


if __name__ == "__main__":
    main()
