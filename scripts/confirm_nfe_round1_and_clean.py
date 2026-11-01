#!/home/lucas.ocunha/masters/.venv_ocr/bin/python3
"""
Confirma os 1.618 documentos validados até 5693818.png como 'nfe':
1. Audita os 1.651 documentos até 5693818.png:
   - 1.618 mantidos -> confirmados em annotations_history.json como 'nfe' e no PostgreSQL.
   - 33 removidos -> registrados em nfe_removidos_33_pelo_usuario.json.
2. Remove os 1.618 validados de 1_confirmados_ocr/ para despoluir o Desktop.
3. Isola os 126 documentos restantes pertencentes aos mesmos fornecedores rejeitados para 3_falsos_positivos_detectados/.
4. Salva histórico e sincroniza banco de dados.
"""

import os
import sys
import json
import gzip
import shutil
from pathlib import Path
from collections import Counter
import psycopg

BASE_DIR = Path("/home/lucas.ocunha/masters")
DATA_DIR = BASE_DIR / "data"
NFE_DIR = DATA_DIR / "validacao_batches" / "validacao_nfe_13k"
CONFIRMADOS_DIR = NFE_DIR / "1_confirmados_ocr"
FALSOS_DIR = NFE_DIR / "3_falsos_positivos_detectados"
RESULTS_FILE = NFE_DIR / "rapidocr_results.json.gz"
HISTORY_FILE = BASE_DIR / "ml_backend" / "models" / "annotations_history.json"
YEAR_FILE = DATA_DIR / "pdf_to_year.json"
DB_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/ceap")

TARGET_STEM = "5693818"


def main():
    print("=" * 72)
    print(" CONFIRMAÇÃO DAS NF-E VALIDADAS ATÉ 5693818.png")
    print("=" * 72)

    # 1. Carregar lista original e metadados
    with gzip.open(RESULTS_FILE, "rt", encoding="utf-8") as f:
        orig_results = json.load(f)

    meta_by_id = {str(d["doc_id"]): d for d in orig_results}
    orig_nfe_ids = sorted([str(d["doc_id"]) for d in orig_results if d["classe"] == "nfe_confirmada"])

    if TARGET_STEM not in orig_nfe_ids:
        print(f"[ERRO] {TARGET_STEM} não encontrado na lista original!")
        sys.exit(1)

    idx_target = orig_nfe_ids.index(TARGET_STEM)
    expected_up_to_target = orig_nfe_ids[:idx_target + 1]
    remaining_expected = orig_nfe_ids[idx_target + 1:]

    current_files = set(p.stem for p in CONFIRMADOS_DIR.glob("*.png"))

    validated_kept = [s for s in expected_up_to_target if s in current_files]
    validated_removed = [s for s in expected_up_to_target if s not in current_files]

    print(f"\n1. Auditoria do intervalo (1 até {TARGET_STEM}):")
    print(f"   • Total no intervalo            : {len(expected_up_to_target):,} documentos")
    print(f"   • Mantidos e validados pelo usuário: {len(validated_kept):,} documentos")
    print(f"   • Descartados pelo usuário      : {len(validated_removed):,} documentos")

    # 2. Registrar os descartados e fornecedores rejeitados
    removed_records = [meta_by_id[s] for s in validated_removed if s in meta_by_id]
    with open(NFE_DIR / "nfe_removidos_33_pelo_usuario.json", "w", encoding="utf-8") as f:
        json.dump(removed_records, f, indent=2, ensure_ascii=False)

    rejected_suppliers = set(
        meta_by_id[s].get("fornecedor", "").strip().upper()
        for s in validated_removed if s in meta_by_id and meta_by_id[s].get("fornecedor")
    )
    print(f"   ✓ {len(rejected_suppliers)} fornecedores rejeitados catalogados:")
    for forn in sorted(rejected_suppliers):
        print(f"      - {forn}")

    # 3. Atualizar annotations_history.json
    print(f"\n2. Atualizando histórico de anotações em {HISTORY_FILE.name}...")
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        history = json.load(f)

    prev_total = len(history)
    prev_nfe = sum(1 for v in history.values() if v == "nfe")

    added_count = 0
    for stem in validated_kept:
        if stem not in history or history[stem] != "nfe":
            history[stem] = "nfe"
            added_count += 1

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    new_total = len(history)
    new_nfe = sum(1 for v in history.values() if v == "nfe")

    print(f"   ✓ NF-e anteriores: {prev_nfe:,}")
    print(f"   ✓ Novas NF-e confirmadas adicionadas: {added_count:,}")
    print(f"   ✓ Novo total de NF-e no dataset: {new_nfe:,}")
    print(f"   ✓ Novo total geral de anotações: {new_total:,}")

    # 4. Sincronizar com PostgreSQL
    print(f"\n3. Sincronizando {len(validated_kept):,} novos confirmados no PostgreSQL...")
    with open(YEAR_FILE, "r", encoding="utf-8") as f:
        pdf_to_year = json.load(f)

    try:
        with psycopg.connect(DB_URL) as conn:
            with conn.cursor() as cur:
                rows_to_upsert = []
                for stem in validated_kept:
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
        print(f"   ✓ {len(rows_to_upsert):,} documentos sincronizados no PostgreSQL.")
    except Exception as e:
        print(f"   [AVISO] Erro PostgreSQL: {e}")

    # 5. Remover os 1.618 validados da pasta 1_confirmados_ocr/
    print(f"\n4. Removendo os {len(validated_kept):,} validados de 1_confirmados_ocr/ para despoluir...")
    removed_from_folder = 0
    for stem in validated_kept:
        p = CONFIRMADOS_DIR / f"{stem}.png"
        if p.exists():
            p.unlink()
            removed_from_folder += 1
    print(f"   ✓ {removed_from_folder:,} arquivos validados removidos da visualização.")

    # 6. Isolar automaticamente os 126 documentos restantes dos mesmos fornecedores rejeitados
    print(f"\n5. Isolando documentos dos mesmos fornecedores rejeitados nos restantes...")
    FALSOS_DIR.mkdir(parents=True, exist_ok=True)
    isolated_count = 0
    isolated_records = []

    for stem in remaining_expected:
        p = CONFIRMADOS_DIR / f"{stem}.png"
        if not p.exists():
            continue
        meta = meta_by_id.get(stem, {})
        forn = meta.get("fornecedor", "").strip().upper()

        is_rejected = any(rf in forn for rf in rejected_suppliers if rf)
        if is_rejected:
            dest_p = FALSOS_DIR / f"{stem}.png"
            shutil.move(p, dest_p)
            isolated_count += 1
            isolated_records.append(meta)

    print(f"   ✓ {isolated_count:,} documentos suspeitos isolados automaticamente em 3_falsos_positivos_detectados/.")

    # 7. Resumo e próxima posição
    remaining_now = list(CONFIRMADOS_DIR.glob("*.png"))
    next_first = sorted([p.stem for p in remaining_now])[0] if remaining_now else "Nenhum"

    print("\n" + "=" * 72)
    print(" RESUMO DA ATUALIZAÇÃO")
    print("=" * 72)
    print(f"NF-e validadas nesta rodada       : {len(validated_kept):,} documentos")
    print(f"Novo total de NF-e no dataset     : {new_nfe:,} documentos")
    print(f"Total geral de documentos no dataset: {new_total:,} documentos")
    print(f"\nSituação da pasta 1_confirmados_ocr/:")
    print(f"  • Restantes para continuar validação: {len(remaining_now):,} documentos")
    print(f"  • Novo primeiro documento da pasta  : {next_first}.png")
    print(f"Situação da pasta 3_falsos_positivos_detectados/:")
    print(f"  • Total acumulado de falsos positivos: {len(list(FALSOS_DIR.glob('*.png'))):,} documentos")
    print("=" * 72)


if __name__ == "__main__":
    main()
