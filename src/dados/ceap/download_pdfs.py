#!/usr/bin/env python3
import argparse
import json
import re
import shutil
import sys
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dados.config import PDF_DIR, RAW_DIR

DEFAULT_DEST = PDF_DIR
HEADERS = {"User-Agent": "Mozilla/5.0"}


def get_default_json_files() -> list[Path]:
    valid = []
    for p in RAW_DIR.glob("Ano-*.json"):
        m = re.search(r"Ano-(\d{4})", p.name)
        if m and int(m.group(1)) >= 2014:
            valid.append(p)
    return sorted(set(valid))


def get_pdf_urls(json_path: Path) -> list[str]:
    print(f"Carregando {json_path.name}...", flush=True)
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


def download_pdf(url: str, dest_dir: Path) -> str:
    """Baixa um PDF e devolve "baixado", "existente" ou "falha"."""
    filename = dest_dir / url.split("/")[-1]
    if filename.exists() and filename.stat().st_size > 0:
        return "existente"
    if shutil.disk_usage(dest_dir).free < 200 * 1024 * 1024:
        raise OSError("Espaço em disco insuficiente (< 200MB livres).")

    tmp_file = filename.with_suffix(".tmp")
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp, open(tmp_file, "wb") as f:
            shutil.copyfileobj(resp, f)
        tmp_file.replace(filename)
        return "baixado"
    except Exception as e:
        if tmp_file.exists():
            tmp_file.unlink()
        print(f"\n[Falha] {url}: {e}", file=sys.stderr, flush=True)
        return "falha"


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
        "--workers",
        "-w",
        type=int,
        default=6,
        help="Quantidade de downloads concorrentes (padrão: 6)",
    )
    parser.add_argument(
        "json_files",
        nargs="*",
        type=Path,
        help="Arquivos Ano-*.json (padrão: >= 2014 em data/raw)",
    )
    args = parser.parse_args()

    args.path.mkdir(parents=True, exist_ok=True)
    json_paths = args.json_files or get_default_json_files()

    if not json_paths:
        print("Nenhum arquivo Ano-*.json encontrado (>= 2014). Baixe os JSONs primeiro.", flush=True)
        sys.exit(1)

    results = Counter()
    for path in json_paths:
        urls = get_pdf_urls(path)
        total = len(urls)
        print(f"[{path.name}] Iniciando download de {total} PDFs em {args.path}...", flush=True)

        if total == 0:
            continue

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(download_pdf, u, args.path) for u in urls]
            for i, future in enumerate(as_completed(futures), start=1):
                try:
                    results[future.result()] += 1
                except OSError as e:
                    print(f"\nInterrompido: {e}", file=sys.stderr, flush=True)
                    executor.shutdown(wait=False, cancel_futures=True)
                    sys.exit(1)
                if i % 10 == 0 or i == total:
                    pct = (i * 100) // total
                    print(f"\r[{path.name}] {i}/{total} ({pct}%) PDFs processados", end="", flush=True)
        print()

    print(f"Baixados: {results['baixado']} | Já existentes: {results['existente']} | Falhas: {results['falha']}", flush=True)


if __name__ == "__main__":
    main()
