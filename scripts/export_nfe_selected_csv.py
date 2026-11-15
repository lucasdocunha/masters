#!/usr/bin/env python3
"""
Gera os arquivos CSV detalhados em data/ com as Notas Fiscais Eletrônicas (NF-e) selecionadas:
1. data/ceap_nfe_selecionadas_novas.csv  (As 2.494 NF-e validadas nesta etapa: 1.568 desktop + 926 site)
2. data/ceap_nfe_todas_20k.csv           (A base completa de todas as 20.232 NF-e rotuladas)
3. data/ceap_nfe_selecionadas.csv       (Atalho/cópia com as selecionadas nesta etapa)
"""

import csv
import json
import re
import time
from pathlib import Path

from dados.config import DATA_DIR, DESKTOP_DIR, ROOT_DIR

HISTORY_FILE = DATA_DIR / "models" / "annotations_history.json"
COMPLETO_CSV = DATA_DIR / "ceap_documentos_completo.csv"
CORRECOES_SITE = DATA_DIR / "labeling" / "correcoes_site.csv"
DESKTOP_META_20K = DESKTOP_DIR / "validacao_nfe_meta_20k"

CSV_HEADER = [
    "doc_id",
    "classe",
    "origem_selecao",
    "fornecedor",
    "cnpj_cpf",
    "valor_liquido",
    "valor_documento",
    "data_emissao",
    "ano",
    "subcota",
    "nome_parlamentar",
    "sigla_uf",
    "sigla_partido",
    "link_documento",
    "caminho_arquivo",
]


def main():
    t0 = time.time()
    print("=" * 70)
    print("GERANDO CSVs DAS NF-E SELECIONADAS EM data/")
    print("=" * 70)

    # 1. Obter IDs validados no Desktop (1.568 arquivos mantidos)
    desktop_ids = set()
    if DESKTOP_META_20K.exists():
        for p in DESKTOP_META_20K.rglob("*.png"):
            m = re.search(r"_(\d+)\.png$", p.name)
            if m:
                desktop_ids.add(m.group(1))
    print(f"IDs confirmados nas pastas do Desktop: {len(desktop_ids):,}")

    # 2. Obter IDs validados no Site (últimos 926 do correcoes_site.csv)
    site_ids = set()
    if CORRECOES_SITE.exists():
        with open(CORRECOES_SITE, "r", encoding="utf-8") as f:
            site_rows = list(csv.reader(f))[-926:]
            for r in site_rows:
                if len(r) >= 5 and r[4] == "nfe":
                    site_ids.add(r[1])
    print(f"IDs validados no Site nesta etapa: {len(site_ids):,}")

    novas_selecionadas = desktop_ids.union(site_ids)
    print(f"Total combinado de NF-e selecionadas nesta etapa: {len(novas_selecionadas):,}")

    # 3. Obter todas as NF-e do histórico (20.232)
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        history = json.load(f)
    norm_hist = {k.replace(".pdf", ""): v for k, v in history.items()}
    all_nfe_ids = set(k for k, v in norm_hist.items() if v == "nfe")
    print(f"Total de NF-e no histórico completo: {len(all_nfe_ids):,}")

    # 4. Ler metadados do ceap_documentos_completo.csv
    print("\nLendo metadados completos de ceap_documentos_completo.csv...")
    docs_meta = {}
    with open(COMPLETO_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            d_id = row["doc_id"]
            if d_id in all_nfe_ids:
                try:
                    meta_obj = json.loads(row["metadados_json"])
                except Exception:
                    meta_obj = {}
                docs_meta[d_id] = {
                    "fornecedor": meta_obj.get("fornecedor", ""),
                    "cnpj_cpf": meta_obj.get("cnpjCPF", ""),
                    "valor_liquido": meta_obj.get("valorLiquido", "0.0"),
                    "valor_documento": meta_obj.get("valorDocumento", "0.0"),
                    "data_emissao": meta_obj.get("dataEmissao", ""),
                    "ano": row.get("ano", ""),
                    "subcota": meta_obj.get("subcota", ""),
                    "nome_parlamentar": meta_obj.get("nomeParlamentar", ""),
                    "sigla_uf": meta_obj.get("siglaUF", ""),
                    "sigla_partido": meta_obj.get("siglaPartido", ""),
                    "link_documento": row.get("link_documento", ""),
                    "caminho_arquivo": row.get("caminho_arquivo", f"/media/ssd2/datasets/CEAP/{d_id}.pdf")
                }

    print(f"Metadados carregados para {len(docs_meta):,} documentos.")

    def format_row(d_id: str, origem: str) -> list:
        m = docs_meta.get(d_id, {})
        return [
            d_id,
            "nfe",
            origem,
            m.get("fornecedor", ""),
            m.get("cnpj_cpf", ""),
            m.get("valor_liquido", "0.0"),
            m.get("valor_documento", "0.0"),
            m.get("data_emissao", ""),
            m.get("ano", ""),
            m.get("subcota", ""),
            m.get("nome_parlamentar", ""),
            m.get("sigla_uf", ""),
            m.get("sigla_partido", ""),
            m.get("link_documento", ""),
            m.get("caminho_arquivo", f"/media/ssd2/datasets/CEAP/{d_id}.pdf")
        ]

    # 6. Gerar CSV principal: data/ceap_nfe_selecionadas.csv com TODAS as 20.232 NF-e
    out_selecionadas = DATA_DIR / "ceap_nfe_selecionadas.csv"
    with open(out_selecionadas, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        for d_id in sorted(all_nfe_ids, key=lambda x: int(x) if x.isdigit() else x):
            if d_id in desktop_ids:
                origem = "validacao_desktop_meta_20k"
            elif d_id in site_ids:
                origem = "validacao_site"
            else:
                origem = "base_historica"
            writer.writerow(format_row(d_id, origem))

    print(f"[OK] {out_selecionadas.name} salvo com {len(all_nfe_ids):,} linhas ({out_selecionadas.stat().st_size / (1024*1024):.2f} MB)")

    # 7. Manter também data/ceap_nfe_todas_20k.csv como sinônimo explícito
    out_todas = DATA_DIR / "ceap_nfe_todas_20k.csv"
    if out_todas != out_selecionadas:
        import shutil
        shutil.copyfile(out_selecionadas, out_todas)
    print(f"[OK] {out_todas.name} sincronizado com {len(all_nfe_ids):,} linhas.")

    # 8. Guardar lote recente (2.494 docs) em arquivo específico
    out_novas = DATA_DIR / "ceap_nfe_selecionadas_lote_recente.csv"
    with open(out_novas, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        for d_id in sorted(novas_selecionadas, key=lambda x: int(x) if x.isdigit() else x):
            origem = "validacao_desktop_meta_20k" if d_id in desktop_ids else "validacao_site"
            writer.writerow(format_row(d_id, origem))
    print(f"[OK] {out_novas.name} salvo com {len(novas_selecionadas):,} linhas.")

    print(f"\nConcluído com sucesso em {time.time()-t0:.2f}s!")


if __name__ == "__main__":
    main()
