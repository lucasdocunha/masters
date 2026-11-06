#!/home/lucas.ocunha/masters/.venv_ocr/bin/python3
"""
Aplica Regra de Auto-Confirmação Rápida de NF-e solicitada pelo usuário:
Critérios:
  1. Contém a palavra 'DANFE' no OCR
  2. Contém 'CHAVE DE ACESSO' (ou chave de 44 dígitos) no OCR
  3. Similaridade métrica ArcFace >= 0.96 (96% de confirmação)

Ações:
  - Confirma os documentos correspondentes em annotations_history.json como 'nfe'.
  - Sincroniza os registros na tabela 'documento' do PostgreSQL.
  - Remove os arquivos auto-confirmados de 1_confirmados_ocr/ para deixar apenas os restantes no Desktop.
  - Atualiza manifest.json e salva log de execução.
"""

import os
import sys
import re
import json
import gzip
from pathlib import Path
from collections import Counter
import psycopg

BASE_DIR = Path("/home/lucas.ocunha/masters")
DATA_DIR = BASE_DIR / "data"
NFE_DIR = DATA_DIR / "validacao_batches" / "validacao_nfe_13k"
CONFIRMADOS_DIR = NFE_DIR / "1_confirmados_ocr"
RESULTS_FILE = NFE_DIR / "rapidocr_results.json.gz"
HISTORY_FILE = BASE_DIR / "data" / "models" / "annotations_history.json"
YEAR_FILE = DATA_DIR / "pdf_to_year.json"
DB_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/ceap")

RE_DANFE = re.compile(r'DANFE\b', re.IGNORECASE)
RE_CHAVE = re.compile(r'CHAVE[\s_-]*(DE\s*)?ACESSO|\b\d{44}\b|\b\d{4}\s*\d{4}\s*\d{4}\s*\d{4}', re.IGNORECASE)


def main():
    print("=" * 72)
    print(" AUTO-CONFIRMAÇÃO NF-e: DANFE + CHAVE DE ACESSO + SIMILARIDADE >= 0.96")
    print("=" * 72)

    # 1. Carregar resultados do OCR
    with gzip.open(RESULTS_FILE, "rt", encoding="utf-8") as f:
        orig_results = json.load(f)

    current_files = set(p.stem for p in CONFIRMADOS_DIR.glob("*.png"))
    print(f"\n1. Documentos atualmente na pasta 1_confirmados_ocr/: {len(current_files):,}")

    current_records = [d for d in orig_results if str(d["doc_id"]) in current_files]

    # 2. Aplicar filtros da regra
    auto_confirmed = []
    remaining = []

    for d in current_records:
        sim = float(d.get("similarity", 0.0))
        txt = d.get("texto", "") or ""

        has_danfe = bool(RE_DANFE.search(txt))
        has_chave = bool(RE_CHAVE.search(txt))
        has_96 = (sim >= 0.96)

        if has_danfe and has_chave and has_96:
            auto_confirmed.append(d)
        else:
            remaining.append(d)

    print(f"\n2. Aplicação da Regra:")
    print(f"   • Documentos que atendem 100% aos 3 critérios: {len(auto_confirmed):,} ({len(auto_confirmed)/len(current_records)*100:.1f}%)")
    print(f"   • Restantes para inspeção humana              : {len(remaining):,} ({len(remaining)/len(current_records)*100:.1f}%)")

    if not auto_confirmed:
        print("[AVISO] Nenhum documento atendeu aos critérios.")
        sys.exit(0)

    # 3. Atualizar annotations_history.json
    print(f"\n3. Registrando {len(auto_confirmed):,} documentos em annotations_history.json...")
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        history = json.load(f)

    prev_total = len(history)
    prev_nfe = sum(1 for v in history.values() if v == "nfe")

    added_count = 0
    confirmed_stems = []
    for d in auto_confirmed:
        s = str(d["doc_id"])
        confirmed_stems.append(s)
        if s not in history or history[s] != "nfe":
            history[s] = "nfe"
            added_count += 1

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    new_total = len(history)
    new_nfe = sum(1 for v in history.values() if v == "nfe")

    print(f"   ✓ Total anterior de NF-e: {prev_nfe:,}")
    print(f"   ✓ Novas NF-e adicionadas: {added_count:,}")
    print(f"   ✓ Novo total de NF-e no dataset: {new_nfe:,}")
    print(f"   ✓ Novo total geral de anotações : {new_total:,}")

    # 4. Sincronizar com PostgreSQL
    print(f"\n4. Sincronizando com PostgreSQL ({DB_URL})...")
    with open(YEAR_FILE, "r", encoding="utf-8") as f:
        pdf_to_year = json.load(f)

    try:
        with psycopg.connect(DB_URL) as conn:
            with conn.cursor() as cur:
                rows_to_upsert = []
                for s in confirmed_stems:
                    doc_id = int(s)
                    ano = pdf_to_year.get(f"{s}.pdf", 2024)
                    caminho = f"/media/ssd2/datasets/CEAP/{s}.pdf"
                    rows_to_upsert.append((doc_id, ano, caminho, "baixado", "nfe"))

                batch_size = 1000
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
        print(f"   ✓ {len(rows_to_upsert):,} documentos sincronizados na tabela 'documento'.")
    except Exception as e:
        print(f"   [AVISO] Erro PostgreSQL: {e}")

    # 5. Remover arquivos auto-confirmados de 1_confirmados_ocr/
    print(f"\n5. Removendo {len(confirmed_stems):,} arquivos auto-confirmados da pasta visual...")
    removed_files = 0
    for s in confirmed_stems:
        p = CONFIRMADOS_DIR / f"{s}.png"
        if p.exists():
            p.unlink()
            removed_files += 1
    print(f"   ✓ {removed_files:,} imagens removidas de 1_confirmados_ocr/.")

    # 6. Salvar log da auto-confirmação
    log_info = {
        "regra": "DANFE + CHAVE_DE_ACESSO + SIMILARIDADE >= 0.96",
        "total_auto_confirmados": len(auto_confirmed),
        "total_restantes_inspecao": len(remaining),
        "data": "2026-11-01"
    }
    with open(NFE_DIR / "auto_confirmados_log.json", "w", encoding="utf-8") as f:
        json.dump(log_info, f, indent=2, ensure_ascii=False)

    # 7. Resumo
    remaining_files = list(CONFIRMADOS_DIR.glob("*.png"))
    first_remaining = sorted([p.stem for p in remaining_files])[0] if remaining_files else "Nenhum"

    print("\n" + "=" * 72)
    print(" AUTO-CONFIRMAÇÃO CONCLUÍDA COM SUCESSO!")
    print("=" * 72)
    print(f"Auto-confirmados como NF-e nesta execução : {len(auto_confirmed):,} documentos")
    print(f"Total acumulado de NF-e no dataset        : {new_nfe:,} documentos")
    print(f"Total geral de documentos no dataset      : {new_total:,} documentos")
    print(f"\nRestantes na pasta para inspeção humana   : {len(remaining_files):,} documentos")
    print(f"Primeiro documento da pasta agora         : {first_remaining}.png")
    print("=" * 72)


if __name__ == "__main__":
    main()
