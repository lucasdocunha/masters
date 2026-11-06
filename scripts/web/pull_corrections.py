"""Traz as correções de classe do site para o histórico de anotações.

Uso (a partir da raiz do repositório):
    uv run python scripts/web/pull_corrections.py            # só mostra o que mudaria
    uv run python scripts/web/pull_corrections.py --aplicar  # grava, com backup
"""

import argparse
import json
import shutil
import time
import urllib.request

from dados.config import ROOT_DIR
from dados.web.latent import merge_corrections

SITE = "https://ceap-latent-space-3d.vercel.app"
HISTORY = ROOT_DIR / "data/models/annotations_history.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=SITE, help="Endereço do site")
    parser.add_argument("--historico", type=str, default=str(HISTORY), help="Arquivo de anotações")
    parser.add_argument("--aplicar", action="store_true", help="Grava no histórico; sem isso, só mostra")
    args = parser.parse_args()

    with urllib.request.urlopen(f"{args.url}/api/corrections", timeout=60) as response:
        events = json.load(response)["corrections"]
    with open(args.historico, encoding="utf-8") as f:
        history = json.load(f)

    result = merge_corrections(history, events)
    print(f"{len(events)} eventos, {len(result['changes'])} mudanças:")
    for change in result["changes"]:
        print(f"  {change['id']}: {change['de']} -> {change['para']}")
    if not args.aplicar or not result["changes"]:
        return
    backup = f"{args.historico}.{time.strftime('%Y%m%d_%H%M%S')}.bak"
    shutil.copyfile(args.historico, backup)
    with open(args.historico, "w", encoding="utf-8") as f:
        json.dump(result["history"], f, ensure_ascii=False, indent=2)
    print(f"Histórico atualizado. Backup em {backup}")


if __name__ == "__main__":
    main()
