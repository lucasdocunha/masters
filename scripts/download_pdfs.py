#!/usr/bin/env python3
import argparse
import json
import shutil
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DEFAULT_DEST = Path(__file__).resolve().parents[1] / "data" / "raw" / "pdfs"


def get_pdf_urls(json_path: Path) -> list[str]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = data.get("dados", []) if isinstance(data, dict) else data
    return sorted(
        {
            item["urlDocumento"]
            for item in items
            if item.get("urlDocumento", "").lower().endswith(".pdf")
        }
    )


def download_pdf(url: str, dest_dir: Path) -> None:
    filename = dest_dir / url.split("/")[-1]
    if filename.exists() and filename.stat().st_size > 0:
        return
    if shutil.disk_usage(dest_dir).free < 200 * 1024 * 1024:
        raise OSError("Espaço em disco insuficiente (< 200MB livres).")
    try:
        urllib.request.urlretrieve(url, filename)
    except Exception as e:
        print(f"Erro ao baixar {url}: {e}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Baixa comprovantes PDF da CEAP")
    parser.add_argument(
        "--path",
        "-p",
        type=Path,
        default=DEFAULT_DEST,
        help="Diretório onde salvar os PDFs",
    )
    parser.add_argument(
        "json_files",
        nargs="*",
        type=Path,
        help="Arquivos Ano-*.json (padrão: busca no projeto)",
    )
    args = parser.parse_args()

    args.path.mkdir(parents=True, exist_ok=True)
    json_paths = args.json_files or sorted(
        Path(__file__).resolve().parents[1].glob("Ano-*.json")
    )

    for path in json_paths:
        urls = get_pdf_urls(path)
        print(f"[{path.name}] Baixando {len(urls)} PDFs em {args.path}...")
        with ThreadPoolExecutor(max_workers=10) as executor:
            executor.map(lambda u: download_pdf(u, args.path), urls)


if __name__ == "__main__":
    main()
