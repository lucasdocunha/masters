"""Atualiza as classes do espaço latente (web/data/latent/classes.bin.gz e index.json) a partir de data/rotulos/.

Não mexe nas coordenadas. Rode depois de mudar rótulos fora do site (sync_labels.py) e antes de publicar.

Uso (a partir da raiz do repositório):
    uv run python scripts/web/refresh_latent_classes.py
"""

from dados.config import DATA_DIR, WEB_DIR
from dados.labeling.labels_csv import read_labels_csv
from dados.web.latent import refresh_latent_classes


def main() -> None:
    stats = refresh_latent_classes(WEB_DIR / "data/latent", read_labels_csv(DATA_DIR / "rotulos"))
    print(f"{stats['pontos']:,} pontos · {stats['rotulados']:,} rotulados · {stats['mudaram']:,} mudaram de classe")


if __name__ == "__main__":
    main()
