"""Gera/atualiza os rótulos versionados (data/rotulos/parte_NNNN.csv) e, se pedido, o CSV único.

As partes são a fonte da verdade dos rótulos: o site grava nelas direto pelo GitHub. Por isso, quando
elas já existem, este script só acrescenta metadados de documentos novos e nunca troca classes vindas do site.

Uso (a partir da raiz do repositório):
    uv run python scripts/export_dataset_csv.py                  # completa metadados faltantes nas partes
    uv run python scripts/export_dataset_csv.py --juntar         # também junta tudo em data/ceap_documentos_rotulados.csv
    uv run python scripts/export_dataset_csv.py --do-historico   # (re)cria as partes a partir de annotations_history.json
    uv run python scripts/export_dataset_csv.py --completo       # também gera data/ceap_documentos_completo.csv
                                                                 # (todos os PDFs em CEAP_PDF_DIR)
Os CSVs únicos não são versionados.
"""

import argparse
import csv
import json
import os

from dados.config import DATA_DIR, MODELS_DIR, PDF_DIR, RAW_DIR
from dados.labeling.labels_csv import (
    HEADERS, build_rows, doc_key, load_raw_metadata, make_row, normalize_history, read_label_rows, update_rows,
    write_labels_csv, write_labels_dir,
)

PARTS_DIR = DATA_DIR / "rotulos"
CSV_ROTULADOS = DATA_DIR / "ceap_documentos_rotulados.csv"
CSV_COMPLETO = DATA_DIR / "ceap_documentos_completo.csv"
HISTORY = MODELS_DIR / "annotations_history.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--do-historico", action="store_true", help="Recria as partes com as classes de annotations_history.json")
    parser.add_argument("--juntar", action="store_true", help="Junta as partes em data/ceap_documentos_rotulados.csv")
    parser.add_argument("--completo", action="store_true", help="Gera também o CSV com todos os PDFs do disco")
    args = parser.parse_args()

    if args.do_historico or not any(PARTS_DIR.glob("parte_*.csv")):
        labels = normalize_history(json.loads(HISTORY.read_text(encoding="utf-8")))
        print(f"{len(labels):,} rótulos em {HISTORY.name}; lendo metadados de {RAW_DIR}…")
        rows = build_rows(labels, load_raw_metadata(RAW_DIR, labels))
    else:
        existing = read_label_rows(PARTS_DIR)
        labels = {doc_id: row[-2] for doc_id, row in existing.items()}
        missing = [doc_id for doc_id, row in existing.items() if not row[1]]
        print(f"{len(labels):,} rótulos nas partes; {len(missing):,} sem metadados.")
        metadata = load_raw_metadata(RAW_DIR, missing) if missing else {}
        rows = update_rows({k: v for k, v in existing.items() if k not in metadata}, labels, metadata)

    count = write_labels_dir(PARTS_DIR, rows)
    parts = sorted(PARTS_DIR.glob("parte_*.csv"))
    largest = max(p.stat().st_size for p in parts) / 2**20
    print(f"{PARTS_DIR}: {count:,} linhas em {len(parts)} partes (maior: {largest:.2f} MB)")

    if args.juntar:
        write_labels_csv(CSV_ROTULADOS, rows)
        print(f"{CSV_ROTULADOS}: {CSV_ROTULADOS.stat().st_size / 2**20:.1f} MB")

    if args.completo:
        on_disk = {name.removesuffix(".pdf") for name in os.listdir(PDF_DIR) if name.endswith(".pdf")}
        by_id = {row[0]: row for row in rows}
        with open(CSV_COMPLETO, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(HEADERS)
            for doc_id in sorted(on_disk | set(by_id), key=doc_key):
                writer.writerow(by_id.get(doc_id) or make_row(doc_id, "Não Mapeado", None)[:-1] + ["nao_mapeado"])
        print(f"{CSV_COMPLETO}: {len(on_disk | set(by_id)):,} linhas")


if __name__ == "__main__":
    main()
