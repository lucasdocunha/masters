#!/usr/bin/env python3
import io
import re
import urllib.request
import zipfile
from pathlib import Path

PAGE_URL = "https://dadosabertos.camara.leg.br/swagger/api.html"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def get_json_zip_urls(start_year: int = 2014) -> list[str]:
    html = urllib.request.urlopen(PAGE_URL).read().decode("utf-8")
    urls = sorted(set(re.findall(r'https?://[^\s"\'<>]+/cotas/Ano-\d{4}\.json\.zip', html)))
    return [
        url
        for url in urls
        if int(re.search(r"Ano-(\d{4})", url).group(1)) >= start_year
    ]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    urls = get_json_zip_urls(2014)

    for url in urls:
        year = re.search(r"Ano-(\d{4})", url).group(1)
        print(f"Baixando e descompactando {year}...")
        with urllib.request.urlopen(url) as resp:
            with zipfile.ZipFile(io.BytesIO(resp.read())) as z:
                z.extractall(OUTPUT_DIR)
        print(f"Ano-{year}.json salvo em {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
