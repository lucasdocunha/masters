#!/home/lucas.ocunha/masters/.venv_ocr/bin/python3
"""
Confirma os documentos do Tier 1 (98% a 100% de similaridade) validados pelo usuário:
1. Audita e confirma os 11.699 documentos restantes em annotations_history.json como 'nfse'.
2. Sincroniza todos os 11.699 no PostgreSQL (tabela 'documento' com status='baixado', classe_predita='nfse').
3. Remove a pasta validada '1_similares_98_a_100pct' para despoluir a visualização no Desktop.
4. Atualiza os resumos estatísticos em manifest_summary.json.
"""

import os
import sys
import json
import time
import shutil
from pathlib import Path
from collections import Counter
import psycopg

BASE_DIR = Path("/home/lucas.ocunha/masters")
DATA_DIR = BASE_DIR / "data"
VALIDATION_DIR = DATA_DIR / "validacao_batches" / "validacao_nfse_38k"
TIER1_DIR = VALIDATION_DIR / "1_similares_98_a_100pct"
HISTORY_FILE = BASE_DIR / "data" / "models" / "annotations_history.json"
YEAR_FILE = DATA_DIR / "pdf_to_year.json"
DB_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/ceap")


def main():
    print("=" * 72)
    print(" CONFIRMAÇÃO DO TIER 1 (98% A 100% DE SIMILARIDADE) COMO NFSe")
    print("=" * 72)

    if not TIER1_DIR.exists():
        print(f"[ERRO] Diretório {TIER1_DIR} não encontrado ou já foi processado!")
        sys.exit(1)

    # 1. Mapear arquivos presentes nos lotes do Tier 1
    lote_dirs = sorted([d for d in TIER1_DIR.iterdir() if d.is_dir() and d.name.startswith("lote_")])
    total_files = []
    print(f"\n1. Auditando lotes do Tier 1:")
    for ld in lote_dirs:
        files = list(ld.glob("*.png"))
        total_files.extend(files)
        print(f"   • {ld.name:15s}: {len(files):5,} documentos")

    print(f"\n   Total de documentos a confirmar: {len(total_files):,} arquivos PNG.")

    # 2. Carregar e atualizar annotations_history.json
    print(f"\n2. Atualizando {HISTORY_FILE}...")
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        history = json.load(f)

    prev_total = len(history)
    prev_nfse = sum(1 for v in history.values() if v == "nfse")

    new_confirmed_stems = []
    added_count = 0
    for p in total_files:
        stem = p.stem
        new_confirmed_stems.append(stem)
        if stem not in history or history[stem] != "nfse":
            history[stem] = "nfse"
            added_count += 1

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    new_total = len(history)
    new_nfse = sum(1 for v in history.values() if v == "nfse")

    print(f"   ✓ Documentos anotados anteriormente: {prev_total:,} (NFS-e: {prev_nfse:,})")
    print(f"   ✓ Novos documentos adicionados/atualizados: {added_count:,}")
    print(f"   ✓ Novo total no histórico: {new_total:,} (NFS-e confirmadas: {new_nfse:,})")

    # 3. Sincronizar com PostgreSQL
    print(f"\n3. Sincronizando com PostgreSQL ({DB_URL})...")
    with open(YEAR_FILE, "r", encoding="utf-8") as f:
        pdf_to_year = json.load(f)

    try:
        with psycopg.connect(DB_URL) as conn:
            with conn.cursor() as cur:
                rows_to_upsert = []
                for stem in new_confirmed_stems:
                    doc_id = int(stem)
                    ano = pdf_to_year.get(f"{stem}.pdf", 2024)
                    caminho = f"/media/ssd2/datasets/CEAP/{stem}.pdf"
                    rows_to_upsert.append((doc_id, ano, caminho, "baixado", "nfse"))

                batch_size = 2000
                for i in range(0, len(rows_to_upsert), batch_size):
                    chunk = rows_to_upsert[i:i + batch_size]
                    cur.executemany("""
                        INSERT INTO documento (ide_documento, ano, caminho_arquivo, status, classe_predita)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (ide_documento) DO UPDATE SET
                            status = 'baixado',
                            classe_predita = 'nfse';
                    """, chunk)
                conn.commit()
        print(f"   ✓ {len(rows_to_upsert):,} documentos sincronizados com sucesso na tabela 'documento'.")
    except Exception as e:
        print(f"   [AVISO] Erro na sincronização com PostgreSQL: {e}")

    # 4. Excluir a pasta validada do Tier 1
    print(f"\n4. Excluindo pasta validada do Tier 1 ({TIER1_DIR.name})...")
    shutil.rmtree(TIER1_DIR)
    print("   ✓ Pasta Tier 1 excluída com sucesso dos lotes ativos.")

    # 5. Atualizar manifest_summary.json
    summary_file = VALIDATION_DIR / "manifest_summary.json"
    if summary_file.exists():
        with open(summary_file, "r", encoding="utf-8") as f:
            summary = json.load(f)
        summary["tier_1_validado"] = {
            "status": "validado_e_confirmado",
            "confirmados_nfse": len(total_files),
            "data_confirmacao": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        remaining_pngs = len(list(VALIDATION_DIR.glob("*/*/*.png")))
        summary["total_candidatos_restantes_lotes"] = remaining_pngs
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    # 6. Relatório de Lotes Restantes
    remaining_tiers = sorted([d.name for d in VALIDATION_DIR.iterdir() if d.is_dir() and d.name.startswith(("2_", "3_", "4_"))])
    print("\n" + "=" * 72)
    print(" RESUMO DA OPERAÇÃO")
    print("=" * 72)
    print(f"Confirmados como NFS-e nesta rodada : {len(total_files):,} documentos")
    print(f"Total Acumulado de NFS-e no Dataset : {new_nfse:,} documentos")
    print(f"Total Geral de Classes Rotuladas     : {new_total:,} documentos")
    print(f"\nLotes restantes no Desktop para validação:")
    for t in remaining_tiers:
        td = VALIDATION_DIR / t
        t_lotes = list(td.glob("lote_*"))
        t_files = list(td.glob("lote_*/*.png"))
        print(f"  • {t}: {len(t_lotes)} lotes | {len(t_files):,} documentos")

    print(f"Total restante em validacao_nfse_38k: {remaining_pngs:,} documentos")
    print("=" * 72)


if __name__ == "__main__":
    main()
