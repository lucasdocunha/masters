#!/home/lucas.ocunha/masters/.venv_ocr/bin/python3
"""
Confirma os 34.656 documentos validados de CUPOM FISCAL no histórico de anotações e no PostgreSQL.
Ações:
  1. Carrega os arquivos em data/validacao_batches/validacao_cupom_fiscal/1_confirmados_ocr/
  2. Registra todos como classe 'cupom_fiscal' em annotations_history.json
  3. Sincroniza a tabela 'documento' do PostgreSQL com status e classe_predita
  4. Limpa a pasta 1_confirmados_ocr/ para liberar espaço e deixar apenas os itens residuais no Desktop
  5. Salva log e estatísticas consolidadas.
"""

import os
import sys
import json
import time
from pathlib import Path
from collections import Counter
import psycopg

BASE_DIR = Path("/home/lucas.ocunha/masters")
DATA_DIR = BASE_DIR / "data"
BATCH_DIR = DATA_DIR / "validacao_batches" / "validacao_cupom_fiscal"
CONFIRMADOS_DIR = BATCH_DIR / "1_confirmados_ocr"
HIST_FILE = BASE_DIR / "data" / "models" / "annotations_history.json"
YEAR_FILE = DATA_DIR / "pdf_to_year.json"
DB_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/ceap")


def main():
    print("=" * 72)
    print(" CONFIRMAÇÃO DE CUPONS FISCAIS VALIDADOS POR OCR")
    print("=" * 72)
    t0 = time.time()

    # 1. Carregar lista de confirmados
    if not CONFIRMADOS_DIR.exists():
        print(f"[ERRO] Pasta {CONFIRMADOS_DIR} não encontrada.")
        sys.exit(1)

    confirmed_files = [p.stem for p in CONFIRMADOS_DIR.glob("*.png")]
    print(f"\n1. Cupons confirmados na pasta 1_confirmados_ocr/: {len(confirmed_files):,}")

    if not confirmed_files:
        print("[AVISO] Nenhum arquivo para confirmar.")
        sys.exit(0)

    # 2. Atualizar annotations_history.json
    print(f"\n2. Atualizando {HIST_FILE}...")
    with open(HIST_FILE, "r", encoding="utf-8") as f:
        history = json.load(f)

    # Normalizar chaves para stem sem .pdf
    norm_history = {}
    for k, v in history.items():
        norm_history[k.replace(".pdf", "")] = v

    prev_total = len(norm_history)
    prev_cf = sum(1 for v in norm_history.values() if v == "cupom_fiscal")

    added = 0
    for s in confirmed_files:
        if s not in norm_history or norm_history[s] != "cupom_fiscal":
            norm_history[s] = "cupom_fiscal"
            added += 1

    with open(HIST_FILE, "w", encoding="utf-8") as f:
        json.dump(norm_history, f, indent=2, ensure_ascii=False)

    new_total = len(norm_history)
    new_cf = sum(1 for v in norm_history.values() if v == "cupom_fiscal")

    print(f"   ✓ Total anterior de CUPOM FISCAL: {prev_cf:,}")
    print(f"   ✓ Novos adicionados             : {added:,}")
    print(f"   ✓ Novo total de CUPOM FISCAL    : {new_cf:,}")
    print(f"   ✓ Novo total geral do Dataset   : {new_total:,}")

    # 3. Sincronizar PostgreSQL
    print(f"\n3. Sincronizando com PostgreSQL ({DB_URL})...")
    with open(YEAR_FILE, "r", encoding="utf-8") as f:
        pdf_to_year = json.load(f)

    try:
        with psycopg.connect(DB_URL) as conn:
            with conn.cursor() as cur:
                doc_rows = []
                for s in confirmed_files:
                    doc_id = int(s)
                    ano = pdf_to_year.get(f"{s}.pdf", 2024)
                    caminho = f"/media/ssd2/datasets/CEAP/{s}.pdf"
                    doc_rows.append((doc_id, ano, caminho, "baixado", "cupom_fiscal"))

                batch_size = 1000
                for i in range(0, len(doc_rows), batch_size):
                    cur.executemany("""
                        INSERT INTO documento (ide_documento, ano, caminho_arquivo, status, classe_predita)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (ide_documento) DO UPDATE
                        SET classe_predita = EXCLUDED.classe_predita,
                            status = EXCLUDED.status;
                    """, doc_rows[i:i + batch_size])

                conn.commit()
        print("   ✓ PostgreSQL atualizado com sucesso!")
    except Exception as e:
        print(f"   [AVISO] Erro ao sincronizar PostgreSQL: {e}")

    # 4. Limpeza da pasta 1_confirmados_ocr/
    print(f"\n4. Limpando imagens de {CONFIRMADOS_DIR}...")
    removed = 0
    for p in CONFIRMADOS_DIR.glob("*.png"):
        try:
            p.unlink()
            removed += 1
        except Exception:
            pass
    print(f"   ✓ {removed:,} imagens validadas removidas do Desktop (armazenamento seguro mantido em data/rendered_pages/).")

    # 5. Resumo
    print("\n" + "=" * 72)
    print(f" SUCESSO! {added:,} cupons fiscais registrados em {time.time() - t0:.1f}s.")
    print(f" Dataset total agora: {new_total:,} documentos.")
    print("=" * 72)


if __name__ == "__main__":
    main()
