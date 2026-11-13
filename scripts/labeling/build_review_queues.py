"""Gera web/data/review/: filas de revisão por classe (similaridade ao centróide no espaço métrico 128-D).

Roda onde estão os embeddings (cluster). Depois, publique o site para as filas irem ao ar.

Uso (a partir da raiz do repositório):
    uv run python scripts/labeling/build_review_queues.py
    uv run python scripts/labeling/build_review_queues.py --vetores /caminho/latent_vectors_metric_learned_128d.npy --top 3000
    uv run python scripts/labeling/build_review_queues.py --coords-3d   # prévia sem os embeddings (coords 3D do site)
"""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from dados.config import DATA_DIR, WEB_DIR
from dados.labeling.labels_csv import read_labels_csv
from dados.labeling.review_queues import build_review_queues, load_points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vetores", type=Path, default=DATA_DIR / "latent_vectors_metric_learned_128d.npy")
    parser.add_argument("--rotulos", type=Path, default=DATA_DIR / "rotulos", help="Partes do CSV de rotulados (ou um CSV único)")
    parser.add_argument("--top", type=int, default=2000, help="Candidatos por classe")
    parser.add_argument("--coords-3d", action="store_true", help="Usa as coordenadas 3D do espaço métrico do site (prévia)")
    parser.add_argument("--auditoria", type=int, default=500, help="Rotulados menos similares por classe")
    args = parser.parse_args()

    latent = WEB_DIR / "data/latent"
    index = json.loads((latent / "index.json").read_text(encoding="utf-8"))
    labels = read_labels_csv(args.rotulos)
    names = [c["name"] for c in index["classes"] if c["id"] != 0]
    names += sorted(set(labels.values()) - set(names))

    if args.coords_3d:
        raw = gzip.decompress((latent / index["models"]["metric"]["file"]).read_bytes())
        vectors, space = np.frombuffer(raw, dtype=np.float32).reshape(-1, 3), "metric-3d"
    else:
        vectors, space = np.load(args.vetores, mmap_mode="r"), "metric-128d"
    summary = build_review_queues(vectors, load_points(latent / "meta"), labels, names, WEB_DIR / "data/review",
                                  top_n=args.top, audit_n=args.auditoria, space=space)
    print(f"Filas geradas em web/data/review/ (espaço {space}):")
    for name, counts in summary.items():
        print(f"  {name:20s} {counts['rotulados']:>7,} rotulados · {counts['candidatos']:,} candidatos · {counts['auditoria']:,} auditoria")


if __name__ == "__main__":
    main()
