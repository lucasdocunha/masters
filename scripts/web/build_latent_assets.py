"""Gera web/data/latent/ para a página do espaço latente.

Uso (a partir da raiz do repositório):
    uv run python scripts/web/build_latent_assets.py [--csv pontos.csv]
"""

import argparse
from pathlib import Path

from dados.config import ROOT_DIR
from dados.web.latent import build_latent_assets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, help="CSV com o documento de cada ponto, na ordem dos pontos ou com coluna coords_idx")
    args = parser.parse_args()
    stats = build_latent_assets(
        ROOT_DIR / "data/processed/web_legacy",
        ROOT_DIR / "web/data/plotly_dataset.json.gz",
        ROOT_DIR / "web/data/latent",
        csv_path=args.csv,
        raw_dir=ROOT_DIR / "data/raw",
        history_path=ROOT_DIR / "data/models/annotations_history.json",
    )
    print("Espaço latente gerado em web/data/latent/:")
    for key, value in stats.items():
        print(f"  {key:8s} {value:,}")


if __name__ == "__main__":
    main()
