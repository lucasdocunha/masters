"""Gera os dados do site de inspeção em web/ (JSONs, miniaturas WebP e pré-visualizações).

Uso (a partir da raiz do repositório):
    uv run python scripts/web/build_site.py
"""

from dados.web.site import build_site


def main() -> None:
    stats = build_site()
    print("Site gerado em web/:")
    for k, v in stats.items():
        print(f"  {k:18s} {v:,}" if isinstance(v, int) else f"  {k:18s} {v}")


if __name__ == "__main__":
    main()
