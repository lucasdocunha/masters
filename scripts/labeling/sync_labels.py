"""Mantém data/rotulos/ (fonte da verdade) e data/models/annotations_history.json iguais.

Uso (a partir da raiz do repositório):
    uv run python scripts/labeling/sync_labels.py                        # partes -> histórico (depois de git pull)
    uv run python scripts/labeling/sync_labels.py --importar-historico   # leva às partes o que um script mudou no histórico
    uv run python scripts/labeling/sync_labels.py --importar-historico --base 7c15f5c
    uv run python scripts/labeling/sync_labels.py --verificar            # erro se partes e histórico diferem

--importar-historico compara o histórico do disco com o do commit --base (padrão HEAD): só o que mudou
desde lá vai para as partes, então rótulos feitos pelo site não são revertidos. Depois, regenere as classes
do espaço latente com scripts/web/refresh_latent_classes.py.
"""

import argparse
import json
import subprocess
import sys

from dados.config import DATA_DIR, MODELS_DIR, RAW_DIR, ROOT_DIR, WEB_DIR
from dados.labeling.labels_csv import normalize_history, read_labels_csv
from dados.labeling.sync import apply_to_parts, diff_labels, export_history, import_history_changes, points_by_doc

PARTS = DATA_DIR / "rotulos"
HISTORY = MODELS_DIR / "annotations_history.json"
LOG = DATA_DIR / "labeling/correcoes_site.csv"


def history_at(ref: str) -> dict[str, str]:
    rel = HISTORY.relative_to(ROOT_DIR).as_posix()
    out = subprocess.run(["git", "show", f"{ref}:{rel}"], cwd=ROOT_DIR, capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def summary(changes: list[dict]) -> str:
    counts: dict[str, int] = {}
    for c in changes:
        key = f"{c['de'] or 'sem rótulo'} -> {c['para'] or 'sem rótulo'}"
        counts[key] = counts.get(key, 0) + 1
    return "\n".join(f"  {n:>6,}  {k}" for k, n in sorted(counts.items(), key=lambda kv: -kv[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--importar-historico", action="store_true", help="Aplica nas partes o que mudou no histórico desde --base")
    mode.add_argument("--verificar", action="store_true", help="Só confere se partes e histórico são iguais")
    parser.add_argument("--base", default="HEAD", help="Commit de referência do histórico (padrão: HEAD)")
    args = parser.parse_args()

    if args.verificar:
        diff = diff_labels(normalize_history(json.loads(HISTORY.read_text(encoding="utf-8"))), read_labels_csv(PARTS))
        if diff:
            print(f"Partes e histórico diferem em {len(diff):,} documentos (histórico -> partes):\n{summary(diff)}")
            sys.exit(1)
        print("Partes e histórico iguais.")
        return

    if args.importar_historico:
        changes = import_history_changes(history_at(args.base), json.loads(HISTORY.read_text(encoding="utf-8")))
        applied = apply_to_parts(PARTS, LOG, changes, points=points_by_doc(WEB_DIR / "data/latent/meta"), raw_dir=RAW_DIR)
        print(f"{len(changes):,} mudanças no histórico desde {args.base}; {len(applied):,} aplicadas nas partes:")
        if applied:
            print(summary(applied))

    exported = export_history(read_labels_csv(PARTS), HISTORY)
    print(f"Histórico atualizado a partir das partes: {len(exported):,} mudanças.")
    if exported:
        print(summary(exported))


if __name__ == "__main__":
    main()
