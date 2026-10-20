#!/usr/bin/env python3
"""
Seleciona uma amostra representativa de PDFs (10 por ano, de 2014 a 2026 = 130 documentos),
copia os arquivos PDF para 'data/pdfs_selecionados/' e gera os respectivos arquivos JSON
com as respostas esperadas (Ground Truth oficial do CEAP da Câmara dos Deputados).
"""

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any


def format_expected_record(item: dict[str, Any]) -> dict[str, Any]:
    """Formata o registro bruto do CEAP em um schema limpo e padronizado para validação."""
    doc_id = item.get("idDocumento")

    # Tratamento de valores monetários
    try:
        val_doc = float(item.get("valorDocumento", 0.0))
    except (ValueError, TypeError):
        val_doc = 0.0

    try:
        val_liq = float(item.get("valorLiquido", 0.0))
    except (ValueError, TypeError):
        val_liq = 0.0

    try:
        val_glosa = float(item.get("valorGlosa", 0.0))
    except (ValueError, TypeError):
        val_glosa = 0.0

    return {
        "idDocumento": doc_id,
        "ano": int(item.get("ano", 0)) if item.get("ano") else None,
        "mes": int(item.get("mes", 0)) if item.get("mes") else None,
        "fornecedor": (item.get("fornecedor") or "").strip(),
        "cnpjCPF": (item.get("cnpjCPF") or "").strip(),
        "numero": str(item.get("numero") or "").strip(),
        "tipoDocumento": str(item.get("tipoDocumento") or "").strip(),
        "dataEmissao": (item.get("dataEmissao") or "").strip(),
        "valorDocumento": val_doc,
        "valorLiquido": val_liq,
        "valorGlosa": val_glosa,
        "nomeParlamentar": (item.get("nomeParlamentar") or "").strip(),
        "cpfParlamentar": (item.get("cpf") or "").strip(),
        "siglaUF": (item.get("siglaUF") or "").strip(),
        "siglaPartido": (item.get("siglaPartido") or "").strip(),
        "descricaoDespesa": (item.get("descricao") or "").strip(),
        "urlDocumento": (item.get("urlDocumento") or "").strip(),
    }


def select_sample_for_year(
    json_path: Path,
    pdf_dir: Path,
    count: int = 10,
    rng: random.Random = random.Random(42),
) -> list[dict[str, Any]]:
    """Carrega o arquivo do ano e seleciona `count` documentos válidos e diversos."""
    if not json_path.exists():
        print(f"[Aviso] Arquivo não encontrado: {json_path}")
        return []

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data.get("dados", []) if isinstance(data, dict) else data

    # Filtrar apenas registros que possuem idDocumento e cujo PDF existe e não está vazio
    candidates = []
    seen_ids = set()

    for it in items:
        doc_id = it.get("idDocumento")
        if not doc_id or doc_id in seen_ids:
            continue
        pdf_path = pdf_dir / f"{doc_id}.pdf"
        if pdf_path.exists() and pdf_path.stat().st_size > 500:
            seen_ids.add(doc_id)
            candidates.append(it)

    if not candidates:
        print(f"[Aviso] Nenhum PDF válido encontrado para {json_path.name}")
        return []

    # Embaralhar para amostragem representativa
    rng.shuffle(candidates)

    # Selecionar priorizando diversidade de fornecedores
    selected = []
    seen_cnpj = set()

    for cand in candidates:
        cnpj = (cand.get("cnpjCPF") or "").strip()
        if cnpj and cnpj not in seen_cnpj:
            seen_cnpj.add(cnpj)
            selected.append(cand)
            if len(selected) == count:
                break

    # Se não atingiu o count apenas com fornecedores distintos, completa com os demais
    if len(selected) < count:
        for cand in candidates:
            if cand not in selected:
                selected.append(cand)
                if len(selected) == count:
                    break

    return selected


def run_sampling(
    raw_dir: Path,
    pdf_dir: Path,
    output_dir: Path,
    years: list[int],
    per_year: int = 10,
    seed: int = 42,
) -> dict[str, Any]:
    """Executa a seleção, cópia dos PDFs e geração dos arquivos JSON esperados."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    manifest = {
        "metadata": {
            "total_documents": 0,
            "per_year": per_year,
            "years": years,
            "seed": seed,
            "output_dir": str(output_dir),
        },
        "documents": [],
    }

    total_selected = 0
    print(f"=== INICIANDO SELEÇÃO DE {per_year} PDFs POR ANO ({years[0]}-{years[-1]}) ===")

    for y in sorted(years):
        json_file = raw_dir / f"Ano-{y}.json"
        items = select_sample_for_year(json_file, pdf_dir, count=per_year, rng=rng)

        print(f"[{y}] Selecionados {len(items)} documentos")

        for it in items:
            doc_id = it["idDocumento"]
            src_pdf = pdf_dir / f"{doc_id}.pdf"
            dst_pdf = output_dir / f"{doc_id}.pdf"

            # Copiar arquivo PDF
            shutil.copyfile(src_pdf, dst_pdf)

            # Gerar JSON com a resposta esperada (Ground Truth)
            expected = format_expected_record(it)

            expected_json_path = output_dir / f"{doc_id}_esperado.json"
            with open(expected_json_path, "w", encoding="utf-8") as f:
                json.dump(expected, f, ensure_ascii=False, indent=2)

            # Também salva como <doc_id>.json para compatibilidade
            compat_json_path = output_dir / f"{doc_id}.json"
            with open(compat_json_path, "w", encoding="utf-8") as f:
                json.dump(expected, f, ensure_ascii=False, indent=2)

            manifest["documents"].append({
                "idDocumento": doc_id,
                "ano": y,
                "pdf_path": str(dst_pdf),
                "expected_json": str(expected_json_path),
                "fornecedor": expected["fornecedor"],
                "cnpjCPF": expected["cnpjCPF"],
                "valorDocumento": expected["valorDocumento"],
                "valorLiquido": expected["valorLiquido"],
                "dataEmissao": expected["dataEmissao"],
                "numero": expected["numero"],
            })
            total_selected += 1

    manifest["metadata"]["total_documents"] = total_selected

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n[Sucesso] {total_selected} documentos selecionados e copiados para: {output_dir}")
    print(f"  - Manifest consolidado: {manifest_path}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Seleção de PDFs de amostra e geração de Ground Truth")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"), help="Pasta com Ano-*.json")
    parser.add_argument("--pdf-dir", type=Path, default=Path("/media/ssd2/datasets/CEAP"), help="Pasta com os PDFs originais")
    parser.add_argument("--output-dir", "-o", type=Path, default=Path("data/pdfs_selecionados"), help="Pasta de saída")
    parser.add_argument("--per-year", type=int, default=10, help="Quantidade de PDFs por ano")
    parser.add_argument("--years", nargs="*", type=int, default=list(range(2014, 2027)), help="Lista de anos a amostrar")
    parser.add_argument("--seed", type=int, default=42, help="Seed para reprodutibilidade")
    args = parser.parse_args()

    run_sampling(
        raw_dir=args.raw_dir,
        pdf_dir=args.pdf_dir,
        output_dir=args.output_dir,
        years=args.years,
        per_year=args.per_year,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
