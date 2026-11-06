#!/usr/bin/env python3
import argparse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

DEFAULT_DEST = Path(__file__).resolve().parents[1] / "data" / "raw"
URL_TEMPLATE = "https://www.camara.leg.br/cotas/Ano-{year}.json.zip"


def download_url(url: str, dest_dir: Path = DEFAULT_DEST) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / url.split("/")[-1]

    print(f"Baixando: {url}")
    urllib.request.urlretrieve(url, target)

    if target.suffix == ".zip":
        with zipfile.ZipFile(target, "r") as z:
            z.extractall(dest_dir)
        target.unlink()
        print(f"Extraído: {target.stem}")

    return dest_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Baixa e descompacta arquivos da CEAP")
    parser.add_argument(
        "--path",
        "-p",
        type=Path,
        default=DEFAULT_DEST,
        help="Diretório de destino",
    )
    parser.add_argument(
        "urls",
        nargs="*",
        help="URLs específicas a baixar (padrão: 2014 até ano atual)",
    )
    args = parser.parse_args()

    urls = args.urls or [
        URL_TEMPLATE.format(year=y)
        for y in range(2014, datetime.now().year + 1)
    ]
    with ThreadPoolExecutor() as executor:
        list(executor.map(lambda u: download_url(u, args.path), urls))


if __name__ == "__main__":
    main()
