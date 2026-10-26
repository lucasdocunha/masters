#!/usr/bin/env python3
"""
Gera o relatório comparativo completo entre os três pipelines:
1. Google Gemini 2.5 Flash Lite (API Nuvem)
2. Qwen2.5-VL-3B-Instruct (End-to-End Multimodal Local)
3. LightOnOCR-3-4B + Qwen2.5-VL-3B (Two-Stage Local)

Compara métricas globais, acurácia por ano, tempo médio de execução e divergências.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.append(str(Path(__file__).resolve().parent))
from evaluate_extraction import evaluate_single_doc


def evaluate_pipeline(expected_docs: Dict[str, Any], extracted_dir_or_pattern: Any, is_gemini: bool = False) -> tuple[List[Dict[str, Any]], float]:
    results = []
    times = []

    for doc_id, expected_data in expected_docs.items():
        if is_gemini:
            ext_path = Path("/home/lucas.ocunha/masters/data/pdfs_selecionados") / f"{doc_id}_extraido.json"
        else:
            ext_path = Path(extracted_dir_or_pattern) / f"{doc_id}.json"

        if not ext_path.exists():
            continue

        try:
            with open(ext_path, "r", encoding="utf-8") as f:
                extracted_data = json.load(f)
            
            res = evaluate_single_doc(expected_data, extracted_data)
            results.append(res)

            meta = extracted_data.get("_metadados_extracao", {})
            t = meta.get("tempo_segundos")
            if t:
                times.append(float(t))
        except Exception:
            pass

    avg_time = sum(times) / len(times) if times else 0.0
    return results, avg_time


def calc_stats(results: List[Dict[str, Any]], total: int):
    n = len(results)
    if n == 0:
        return {"cnpj": 0, "valor": 0, "data": 0, "numero": 0, "fornecedor": 0, "perfeito": 0, "n": 0}
    
    return {
        "cnpj": sum(1 for r in results if r["cnpj"]["match"]) / n * 100,
        "valor": sum(1 for r in results if r["valor"]["match"]) / n * 100,
        "data": sum(1 for r in results if r["data"]["match"]) / n * 100,
        "numero": sum(1 for r in results if r["numero"]["match"]) / n * 100,
        "fornecedor": sum(1 for r in results if r["fornecedor"]["match"]) / n * 100,
        "perfeito": sum(1 for r in results if r["match_perfeito"]) / n * 100,
        "n": n,
    }


def main():
    data_dir = Path("/home/lucas.ocunha/masters/data/pdfs_selecionados")
    manifest_path = data_dir / "manifest.json"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    expected_docs = {}
    for item in manifest.get("documents", []):
        d_id = str(item["idDocumento"])
        exp_path = data_dir / f"{d_id}_esperado.json"
        if exp_path.exists():
            with open(exp_path, "r", encoding="utf-8") as f:
                expected_docs[d_id] = json.load(f)

    total_base = len(expected_docs)
    print(f"=== COMPARATIVO GERAL DE PIPELINES (Total Ground Truth: {total_base} documentos) ===")

    # 1. Avaliar Gemini
    res_gemini, time_gemini = evaluate_pipeline(expected_docs, None, is_gemini=True)
    stats_gemini = calc_stats(res_gemini, total_base)

    # 2. Avaliar Opção B (Qwen2.5-VL)
    dir_qwenvl = data_dir / "extraidos_qwenvl"
    res_qwenvl, time_qwenvl = evaluate_pipeline(expected_docs, dir_qwenvl, is_gemini=False)
    stats_qwenvl = calc_stats(res_qwenvl, total_base)

    # 3. Avaliar Opção A (Two-Stage LightOn + Qwen)
    dir_twostage = data_dir / "extraidos_twostage"
    res_twostage, time_twostage = evaluate_pipeline(expected_docs, dir_twostage, is_gemini=False)
    stats_twostage = calc_stats(res_twostage, total_base)

    # Montar Relatório em Markdown
    lines = []
    lines.append("# Relatório Comparativo de Pipelines de Extração Fiscal\n")
    lines.append(f"Amostra total avaliada: **{total_base} documentos** do CEAP (anos 2014 a 2026).\n")
    lines.append("## 1. Comparativo Geral de Desempenho e Acurácia\n")
    lines.append("| Métrica / Campo Fiscal | Google Gemini 2.5 Flash Lite | Opção B: Qwen2.5-VL-3B (End-to-End) | Opção A: LightOnOCR + Qwen (Two-Stage) |")
    lines.append("| :--- | :---: | :---: | :---: |")
    lines.append(f"| **Documentos Processados** | **{stats_gemini['n']}/{total_base}** | **{stats_qwenvl['n']}/{total_base}** | **{stats_twostage['n']}/{total_base}** |")
    lines.append(f"| **Tempo Médio / Doc** | **{time_gemini:.2f}s** *(API)* | **{time_qwenvl:.2f}s** *(Local)* | **{time_twostage:.2f}s** *(Local)* |")
    lines.append(f"| **CNPJ / CPF do Emissor** | **{stats_gemini['cnpj']:.1f}%** | **{stats_qwenvl['cnpj']:.1f}%** | **{stats_twostage['cnpj']:.1f}%** |")
    lines.append(f"| **Valor Total do Documento** | **{stats_gemini['valor']:.1f}%** | **{stats_qwenvl['valor']:.1f}%** | **{stats_twostage['valor']:.1f}%** |")
    lines.append(f"| **Data de Emissão** | **{stats_gemini['data']:.1f}%** | **{stats_qwenvl['data']:.1f}%** | **{stats_twostage['data']:.1f}%** |")
    lines.append(f"| **Número da Nota / Cupom** | **{stats_gemini['numero']:.1f}%** | **{stats_qwenvl['numero']:.1f}%** | **{stats_twostage['numero']:.1f}%** |")
    lines.append(f"| **Nome / Razão Social** | **{stats_gemini['fornecedor']:.1f}%** | **{stats_qwenvl['fornecedor']:.1f}%** | **{stats_twostage['fornecedor']:.1f}%** |")
    lines.append(f"| **MATCH PERFEITO (Todos os 5)** | **{stats_gemini['perfeito']:.1f}%** | **{stats_qwenvl['perfeito']:.1f}%** | **{stats_twostage['perfeito']:.1f}%** |")

    report_md = "\n".join(lines)
    print("\n" + report_md + "\n")

    out_md = data_dir / "relatorio_comparativo_pipelines.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"Relatório salvo em: {out_md}")


if __name__ == "__main__":
    main()
