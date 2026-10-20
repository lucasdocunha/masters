#!/usr/bin/env python3
"""
Avaliação comparativa e validação dos dados extraídos pelo Google Gemini contra o Ground Truth oficial do CEAP.
Calcula métricas de acurácia por campo (CNPJ, Valor, Data, Número, Fornecedor), por ano e gera relatórios
completos em Markdown e JSON.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Optional


def clean_digits(val: Any) -> str:
    """Extrai apenas os dígitos de uma string ou número."""
    if val is None:
        return ""
    return re.sub(r"\D", "", str(val))


def parse_date_to_iso(date_str: Any) -> Optional[str]:
    """Tenta converter datas em formato BR (DD/MM/YYYY) ou ISO (YYYY-MM-DD...) para YYYY-MM-DD."""
    if not date_str:
        return None
    s = str(date_str).strip()

    # Formato ISO (YYYY-MM-DD)
    m_iso = re.search(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m_iso:
        return f"{m_iso.group(1)}-{m_iso.group(2)}-{m_iso.group(3)}"

    # Formato BR (DD/MM/YYYY)
    m_br = re.search(r"(\d{2})/(\d{2})/(\d{4})", s)
    if m_br:
        return f"{m_br.group(3)}-{m_br.group(2)}-{m_br.group(1)}"

    return None


def clean_doc_number(num_str: Any) -> str:
    """Remove caracteres especiais e zeros à esquerda de um número de documento."""
    if num_str is None:
        return ""
    # Pega apenas dígitos ou letras
    cleaned = re.sub(r"[^\w]", "", str(num_str)).upper()
    # Remove zeros à esquerda se for numérico
    if cleaned.isdigit():
        return str(int(cleaned))
    return cleaned


def strip_accents(text: str) -> str:
    """Remove acentos e diacríticos de uma string."""
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def compute_text_similarity(str1: str, str2: str) -> float:
    """Calcula a similaridade baseada em tokens (Jaccard) entre duas strings de texto."""
    if not str1 or not str2:
        return 0.0

    def tokenize(s: str) -> set[str]:
        # Remove acentos e converte para maiúsculo
        s = strip_accents(s.upper())
        # Remove sufixos empresariais comuns que variam nos cadastros
        s = re.sub(r"\b(LTDA|SA|S\/A|ME|EPP|EIRELI|COMERCIO|SERVICOS|DO BRASIL)\b", "", s)
        tokens = re.findall(r"\w+", s)
        return set(t for t in tokens if len(t) > 1)

    t1 = tokenize(str1)
    t2 = tokenize(str2)


    if not t1 or not t2:
        # Fallback para comparação direta limpa
        s1 = re.sub(r"\W", "", str1.upper())
        s2 = re.sub(r"\W", "", str2.upper())
        return 1.0 if s1 == s2 and s1 else 0.0

    intersection = len(t1 & t2)
    union = len(t1 | t2)
    return intersection / union if union > 0 else 0.0


def evaluate_single_doc(expected: dict[str, Any], extracted: dict[str, Any]) -> dict[str, Any]:
    """Compara os dados esperados com os extraídos e retorna o resultado da avaliação."""
    # 1. CNPJ/CPF
    cnpj_exp = clean_digits(expected.get("cnpjCPF"))
    cnpj_ext = clean_digits(extracted.get("fornecedor_cnpj_cpf"))
    # Se um dos dois estiver zerado ou com zeros parciais
    cnpj_match = False
    if cnpj_exp and cnpj_ext:
        cnpj_match = (cnpj_exp == cnpj_ext) or (cnpj_exp in cnpj_ext) or (cnpj_ext in cnpj_exp)

    # 2. Valor
    try:
        val_ext = float(extracted.get("valor_total")) if extracted.get("valor_total") is not None else None
    except (ValueError, TypeError):
        val_ext = None

    val_doc_exp = expected.get("valorDocumento")
    val_liq_exp = expected.get("valorLiquido")

    valor_match = False
    valor_diff = None
    if val_ext is not None:
        diff_doc = abs(val_ext - val_doc_exp) if val_doc_exp is not None else 999999
        diff_liq = abs(val_ext - val_liq_exp) if val_liq_exp is not None else 999999
        min_diff = min(diff_doc, diff_liq)
        valor_diff = round(min_diff, 2)
        valor_match = (min_diff <= 0.10)

    # 3. Data de Emissão
    date_exp = parse_date_to_iso(expected.get("dataEmissao"))
    date_ext = parse_date_to_iso(extracted.get("data_emissao"))
    data_match = (date_exp == date_ext) if (date_exp and date_ext) else False

    # 4. Número do Documento
    num_exp = clean_doc_number(expected.get("numero"))
    num_ext = clean_doc_number(extracted.get("numero_documento"))
    numero_match = False
    if num_exp and num_ext:
        numero_match = (num_exp == num_ext) or (num_exp in num_ext) or (num_ext in num_exp)

    # 5. Fornecedor
    forn_exp = expected.get("fornecedor") or ""
    forn_ext = extracted.get("fornecedor_nome") or ""
    sim_forn = compute_text_similarity(forn_exp, forn_ext)
    fornecedor_match = (sim_forn >= 0.50) or (clean_digits(forn_exp) == clean_digits(forn_ext) and clean_digits(forn_exp) != "")

    all_matched = (cnpj_match and valor_match and data_match and numero_match and fornecedor_match)

    return {
        "idDocumento": expected.get("idDocumento"),
        "ano": expected.get("ano"),
        "cnpj": {
            "esperado": expected.get("cnpjCPF"),
            "extraido": extracted.get("fornecedor_cnpj_cpf"),
            "match": cnpj_match,
        },
        "valor": {
            "esperado_documento": val_doc_exp,
            "esperado_liquido": val_liq_exp,
            "extraido": val_ext,
            "diferenca": valor_diff,
            "match": valor_match,
        },
        "data": {
            "esperado": date_exp,
            "extraido": date_ext,
            "match": data_match,
        },
        "numero": {
            "esperado": expected.get("numero"),
            "extraido": extracted.get("numero_documento"),
            "match": numero_match,
        },
        "fornecedor": {
            "esperado": forn_exp,
            "extraido": forn_ext,
            "similaridade": round(sim_forn, 2),
            "match": fornecedor_match,
        },
        "match_perfeito": all_matched,
    }


def generate_markdown_report(results: list[dict[str, Any]], total_available: int) -> str:
    """Gera um relatório completo em formato Markdown com métricas consolidadas."""
    n = len(results)
    if n == 0:
        return "# Relatório de Avaliação - Nenhum documento extraído encontrado."

    cnpj_hits = sum(1 for r in results if r["cnpj"]["match"])
    valor_hits = sum(1 for r in results if r["valor"]["match"])
    data_hits = sum(1 for r in results if r["data"]["match"])
    num_hits = sum(1 for r in results if r["numero"]["match"])
    forn_hits = sum(1 for r in results if r["fornecedor"]["match"])
    perfect_hits = sum(1 for r in results if r["match_perfeito"])

    pct = lambda x: f"{(x / n) * 100:.1f}%"

    md = []
    md.append("# Relatório de Avaliação da Extração com Google Gemini")
    md.append(f"\nDocumentos avaliados: **{n}** de **{total_available}** disponíveis na amostra.\n")

    md.append("## 1. Métricas Globais de Acurácia\n")
    md.append("| Campo Fiscal | Acertos | Total Avaliado | Acurácia (%) |")
    md.append("| :--- | :--- | :--- | :--- |")
    md.append(f"| **CNPJ / CPF do Fornecedor** | {cnpj_hits} | {n} | **{pct(cnpj_hits)}** |")
    md.append(f"| **Valor Total do Documento** | {valor_hits} | {n} | **{pct(valor_hits)}** |")
    md.append(f"| **Data de Emissão** | {data_hits} | {n} | **{pct(data_hits)}** |")
    md.append(f"| **Número da Nota / Documento** | {num_hits} | {n} | **{pct(num_hits)}** |")
    md.append(f"| **Nome / Razão Social do Fornecedor** | {forn_hits} | {n} | **{pct(forn_hits)}** |")
    md.append(f"| **Match Perfeito (Todos os 5 campos)** | {perfect_hits} | {n} | **{pct(perfect_hits)}** |")

    # Acurácia por Ano
    md.append("\n## 2. Acurácia por Ano\n")
    md.append("| Ano | Qtd Docs | CNPJ (%) | Valor (%) | Data (%) | Número (%) | Fornecedor (%) | Perfeito (%) |")
    md.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

    by_year = {}
    for r in results:
        y = r.get("ano", "N/A")
        by_year.setdefault(y, []).append(r)

    for y in sorted(by_year.keys()):
        y_list = by_year[y]
        yn = len(y_list)
        y_cnpj = sum(1 for r in y_list if r["cnpj"]["match"])
        y_val = sum(1 for r in y_list if r["valor"]["match"])
        y_data = sum(1 for r in y_list if r["data"]["match"])
        y_num = sum(1 for r in y_list if r["numero"]["match"])
        y_forn = sum(1 for r in y_list if r["fornecedor"]["match"])
        y_perf = sum(1 for r in y_list if r["match_perfeito"])

        y_pct = lambda x: f"{(x / yn) * 100:.0f}%"
        md.append(f"| {y} | {yn} | {y_pct(y_cnpj)} | {y_pct(y_val)} | {y_pct(y_data)} | {y_pct(y_num)} | {y_pct(y_forn)} | {y_pct(y_perf)} |")

    # Amostra de Discrepâncias
    divergences = [r for r in results if not r["match_perfeito"]]
    if divergences:
        md.append(f"\n## 3. Análise de Discrepâncias ({len(divergences)} casos com divergência em pelo menos 1 campo)\n")
        md.append("| Doc ID | Ano | Campo c/ Falha | Esperado | Extraído |")
        md.append("| :--- | :--- | :--- | :--- | :--- |")

        for r in divergences[:20]:  # Mostrar até os primeiros 20
            d_id = r["idDocumento"]
            y = r["ano"]
            falhas = []
            if not r["cnpj"]["match"]:
                falhas.append(("CNPJ", r["cnpj"]["esperado"], r["cnpj"]["extraido"]))
            if not r["valor"]["match"]:
                falhas.append(("Valor", f"Doc: {r['valor']['esperado_documento']} / Liq: {r['valor']['esperado_liquido']}", str(r["valor"]["extraido"])))
            if not r["data"]["match"]:
                falhas.append(("Data", str(r["data"]["esperado"]), str(r["data"]["extraido"])))
            if not r["numero"]["match"]:
                falhas.append(("Número", str(r["numero"]["esperado"]), str(r["numero"]["extraido"])))
            if not r["fornecedor"]["match"]:
                falhas.append(("Fornecedor", str(r["fornecedor"]["esperado"]), str(r["fornecedor"]["extraido"])))

            for campo, exp, ext in falhas:
                md.append(f"| `{d_id}` | {y} | **{campo}** | `{exp}` | `{ext}` |")

    return "\n".join(md)


def run_evaluation(
    eval_dir: Path,
    output_report: Optional[Path] = None,
    output_json: Optional[Path] = None,
) -> dict[str, Any]:
    """Varre eval_dir, compara arquivos _esperado.json com _extraido.json e salva relatórios."""
    eval_dir = Path(eval_dir)
    expected_files = list(eval_dir.glob("*_esperado.json"))
    if not expected_files:
        expected_files = [p for p in eval_dir.glob("*.json") if not p.name.endswith(("_extraido.json", "manifest.json", "relatorio_avaliacao.json"))]

    print(f"=== INICIANDO AVALIAÇÃO DE EXTRAÇÃO ({len(expected_files)} documentos esperados) ===")

    results = []
    missing_extractions = []

    for exp_path in sorted(expected_files):
        doc_id = exp_path.name.replace("_esperado.json", "").replace(".json", "")
        ext_path = eval_dir / f"{doc_id}_extraido.json"

        if not ext_path.exists():
            missing_extractions.append(doc_id)
            continue

        with open(exp_path, "r", encoding="utf-8") as f:
            expected_data = json.load(f)

        with open(ext_path, "r", encoding="utf-8") as f:
            extracted_data = json.load(f)

        res = evaluate_single_doc(expected_data, extracted_data)
        results.append(res)

    print(f"Documentos extraídos encontrados e validados: {len(results)}/{len(expected_files)}")
    if missing_extractions:
        print(f"[Aviso] {len(missing_extractions)} documentos ainda não possuem arquivo '_extraido.json'")

    md_report = generate_markdown_report(results, len(expected_files))

    rep_path = output_report or (eval_dir / "relatorio_avaliacao.md")
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write(md_report)

    json_path = output_json or (eval_dir / "relatorio_avaliacao.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_amostra": len(expected_files),
                "total_extraidos": len(results),
                "documentos": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\n[Sucesso] Relatórios de avaliação gerados:")
    print(f"  - Markdown: {rep_path}")
    print(f"  - JSON:     {json_path}")

    # Exibir resumo no console
    if results:
        pct = lambda x: f"{(x / len(results)) * 100:.1f}%"
        print(f"\nResumo da Acurácia:")
        print(f"  • CNPJ/CPF:   {pct(sum(1 for r in results if r['cnpj']['match']))}")
        print(f"  • Valor:      {pct(sum(1 for r in results if r['valor']['match']))}")
        print(f"  • Data:       {pct(sum(1 for r in results if r['data']['match']))}")
        print(f"  • Número:     {pct(sum(1 for r in results if r['numero']['match']))}")
        print(f"  • Fornecedor: {pct(sum(1 for r in results if r['fornecedor']['match']))}")
        print(f"  • Perfeito:   {pct(sum(1 for r in results if r['match_perfeito']))}")

    return {"results": results, "report_path": str(rep_path)}


def main():
    parser = argparse.ArgumentParser(description="Validação da extração de documentos contra Ground Truth do CEAP")
    parser.add_argument("--eval-dir", "-d", type=Path, default=Path("data/pdfs_selecionados"), help="Pasta com os pares de arquivos")
    parser.add_argument("--output-report", type=Path, help="Caminho do relatório .md")
    parser.add_argument("--output-json", type=Path, help="Caminho do relatório .json")
    args = parser.parse_args()

    run_evaluation(
        eval_dir=args.eval_dir,
        output_report=args.output_report,
        output_json=args.output_json,
    )


if __name__ == "__main__":
    main()
