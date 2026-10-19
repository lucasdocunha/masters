#!/usr/bin/env python3
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

DEST_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
URL_TEMPLATE = "https://www.camara.leg.br/cotas/Ano-{year}.json.zip"


def download_url(url: str, dest_dir: Path = DEST_DIR) -> Path:
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
    urls = sys.argv[1:] or [
        URL_TEMPLATE.format(year=y)
        for y in range(2014, datetime.now().year + 1)
    ]
    with ThreadPoolExecutor() as executor:
        list(executor.map(download_url, urls))


if __name__ == "__main__":
    main()
