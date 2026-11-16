#!/usr/bin/env python3
"""
Script de Análise Comparativa dos Modelos VLM (S5-nano, Qwen3.5-4B, Qwen3.5-9B)
nas 20.232 Notas Fiscais Eletrônicas (NF-e) do CEAP.

Gera:
1. data/evaluation/relatorio_comparativo_modelos_vlm.json
2. data/evaluation/relatorio_comparativo_modelos_vlm.md
"""

import csv
import json
import re
import time
from collections import Counter
from pathlib import Path

from dados.config import DATA_DIR, ROOT_DIR

MODELS_DIR = DATA_DIR / "evaluation" / "resultados_modelos"
FILES = {
    "S5-nano": MODELS_DIR / "resultados_S5-nano.jsonl",
    "Qwen3.5-4B": MODELS_DIR / "resultados_Qwen3.5-4B.jsonl",
    "Qwen3.5-9B": MODELS_DIR / "resultados_Qwen3.5-9B.jsonl",
}

OUT_JSON = DATA_DIR / "evaluation" / "relatorio_comparativo_modelos_vlm.json"
OUT_MD = DATA_DIR / "evaluation" / "relatorio_comparativo_modelos_vlm.md"
CEAP_20K_CSV = DATA_DIR / "ceap_nfe_todas_20k.csv"

FIELDS = [
    "numero_nota",
    "chave_de_acesso",
    "data_emissao",
    "valor_total_nota",
    "cnpj_emitente",
    "cnpj_destinatario",
]


def load_jsonl(path: Path) -> dict:
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            for k, v in item.items():
                data[k] = v
    return data


def sefaz_mod11(key_str: str) -> bool:
    if not key_str or len(key_str) != 44 or not key_str.isdigit():
        return False
    digits = [int(c) for c in key_str]
    weights = [
        4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4,
        3, 2, 9, 8, 7, 6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2,
    ]
    s = sum(d * w for d, w in zip(digits[:43], weights))
    rem = s % 11
    dv = 0 if rem in (0, 1) else 11 - rem
    return dv == digits[43]


def norm_val(field: str, val):
    if val is None or val == "" or val == []:
        return None
    if field == "valor_total_nota":
        try:
            return round(float(val), 2)
        except Exception:
            s = str(val).replace(".", "").replace(",", ".")
            try:
                return round(float(s), 2)
            except Exception:
                return None
    elif field in ("cnpj_emitente", "cnpj_destinatario", "chave_de_acesso"):
        digits = re.sub(r"\D", "", str(val))
        return digits if digits else None
    elif field == "numero_nota":
        digits = re.sub(r"\D", "", str(val))
        return digits.lstrip("0") if digits else None
    elif field == "data_emissao":
        s = str(val).strip()
        m = re.match(r"^(\d{2})/(\d{2})/(\d{4})", s)
        if m:
            return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
        return s[:10] if len(s) >= 10 else None
    return str(val).strip()


def main():
    t0 = time.time()
    print("=" * 70)
    print("ANALISANDO RESULTADOS DOS MODELOS VLM")
    print("=" * 70)

    # 1. Carregar predições
    models_data = {}
    for name, path in FILES.items():
        print(f"Carregando {name} de {path.name}...")
        models_data[name] = load_jsonl(path)
        print(f"  -> {len(models_data[name]):,} documentos carregados.")

    # 2. Carregar Ground Truth CEAP
    print(f"\nCarregando Ground Truth de {CEAP_20K_CSV.name}...")
    gt = {}
    with open(CEAP_20K_CSV, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            gt[r["link_documento"]] = {
                "doc_id": r["doc_id"],
                "cnpj_emitente": re.sub(r"\D", "", r["cnpj_cpf"]),
                "valor_total_nota": round(float(r["valor_documento"]), 2) if r["valor_documento"] else None,
                "data_emissao": r["data_emissao"][:10] if r["data_emissao"] else None,
                "fornecedor": r["fornecedor"],
            }
    print(f"  -> {len(gt):,} documentos no Ground Truth.")

    # 3. Interseção e união
    all_urls = set()
    for d in models_data.values():
        all_urls.update(d.keys())

    common_urls = sorted(
        list(set(models_data["S5-nano"].keys()) & set(models_data["Qwen3.5-4B"].keys()) & set(models_data["Qwen3.5-9B"].keys()))
    )
    n_common = len(common_urls)
    total_docs = len(all_urls)

    print(f"\nUnião Total: {total_docs:,} | Docs em comum nos 3: {n_common:,} ({n_common/total_docs*100:.2f}%)")

    # 4. Estatísticas de Cobertura e Preenchimento
    cobertura = {}
    for name, d in models_data.items():
        raw_count = sum(1 for v in d.values() if isinstance(v, dict) and "raw_response" in v)
        field_fill = {}
        for f in FIELDS:
            cnt = sum(1 for u in common_urls if isinstance(d[u], dict) and d[u].get(f) not in (None, "", []))
            field_fill[f] = {
                "count": cnt,
                "percentage": round(cnt / n_common * 100, 2),
            }
        cobertura[name] = {
            "total_docs": len(d),
            "percentage_base": round(len(d) / total_docs * 100, 2),
            "raw_response_count": raw_count,
            "raw_response_pct": round(raw_count / len(d) * 100, 2),
            "preenchimento_campos": field_fill,
        }

    # 5. Validação Chave de Acesso SEFAZ Modulo 11
    chave_stats = {}
    for name, d in models_data.items():
        total_k = 0
        len44 = 0
        valid_mod11 = 0
        for u in common_urls:
            val = d[u].get("chave_de_acesso") if isinstance(d[u], dict) else None
            if val:
                clean = re.sub(r"\D", "", str(val))
                total_k += 1
                if len(clean) == 44:
                    len44 += 1
                    if sefaz_mod11(clean):
                        valid_mod11 += 1
        chave_stats[name] = {
            "total_chaves_extraidas": total_k,
            "exatamente_44_digitos": len44,
            "exatamente_44_digitos_pct": round(len44 / n_common * 100, 2),
            "validas_sefaz_mod11": valid_mod11,
            "validas_sefaz_mod11_pct": round(valid_mod11 / n_common * 100, 2),
        }

    # 6. Matriz de Concordância entre Modelos
    concordancia = {}
    s5 = models_data["S5-nano"]
    q4 = models_data["Qwen3.5-4B"]
    q9 = models_data["Qwen3.5-9B"]

    for f in FIELDS:
        consenso_3_norm = 0
        consenso_3_exact = 0
        maioria_2_norm = 0
        divergencia_norm = 0
        q4_q9_norm = 0
        s5_q9_norm = 0
        s5_q4_norm = 0

        for u in common_urls:
            v_s5 = s5[u].get(f) if isinstance(s5[u], dict) else None
            v_q4 = q4[u].get(f) if isinstance(q4[u], dict) else None
            v_q9 = q9[u].get(f) if isinstance(q9[u], dict) else None

            nv_s5 = norm_val(f, v_s5)
            nv_q4 = norm_val(f, v_q4)
            nv_q9 = norm_val(f, v_q9)

            # Exato
            if v_s5 is not None and v_s5 == v_q4 == v_q9:
                consenso_3_exact += 1

            # Normalizado
            vals = [v for v in [nv_s5, nv_q4, nv_q9] if v is not None]
            if not vals:
                continue

            c = Counter(vals)
            top_cnt = c.most_common(1)[0][1]
            if top_cnt == 3:
                consenso_3_norm += 1
                maioria_2_norm += 1
            elif top_cnt == 2:
                maioria_2_norm += 1
            else:
                divergencia_norm += 1

            if nv_q4 is not None and nv_q4 == nv_q9:
                q4_q9_norm += 1
            if nv_s5 is not None and nv_s5 == nv_q9:
                s5_q9_norm += 1
            if nv_s5 is not None and nv_s5 == nv_q4:
                s5_q4_norm += 1

        concordancia[f] = {
            "consenso_3_3_norm": consenso_3_norm,
            "consenso_3_3_norm_pct": round(consenso_3_norm / n_common * 100, 2),
            "consenso_3_3_exato": consenso_3_exact,
            "consenso_3_3_exato_pct": round(consenso_3_exact / n_common * 100, 2),
            "maioria_2_3_norm": maioria_2_norm,
            "maioria_2_3_norm_pct": round(maioria_2_norm / n_common * 100, 2),
            "divergencia_total_norm": divergencia_norm,
            "divergencia_total_norm_pct": round(divergencia_norm / n_common * 100, 2),
            "qwen4b_eq_qwen9b_norm": q4_q9_norm,
            "qwen4b_eq_qwen9b_norm_pct": round(q4_q9_norm / n_common * 100, 2),
            "s5nano_eq_qwen9b_norm": s5_q9_norm,
            "s5nano_eq_qwen9b_norm_pct": round(s5_q9_norm / n_common * 100, 2),
            "s5nano_eq_qwen4b_norm": s5_q4_norm,
            "s5nano_eq_qwen4b_norm_pct": round(s5_q4_norm / n_common * 100, 2),
        }

    # 7. Acurácia contra Ground Truth Oficial CEAP
    acuracia_ceap = {}
    for name, d in models_data.items():
        acuracia_ceap[name] = {}
        for field in ["valor_total_nota", "data_emissao", "cnpj_emitente"]:
            correct = 0
            total_eval = 0
            for u in common_urls:
                expected = gt.get(u, {}).get(field)
                if expected is None:
                    continue
                total_eval += 1
                pred = norm_val(field, d[u].get(field) if isinstance(d[u], dict) else None)
                if pred == expected:
                    correct += 1
            acuracia_ceap[name][field] = {
                "corretos": correct,
                "total": total_eval,
                "acuracia_pct": round(correct / total_eval * 100, 2),
            }

    # Ensemble (Maioria 2/3) contra CEAP
    acuracia_ceap["Ensemble_Maioria_2_3"] = {}
    for field in ["valor_total_nota", "data_emissao", "cnpj_emitente"]:
        correct = 0
        total_eval = 0
        for u in common_urls:
            expected = gt.get(u, {}).get(field)
            if expected is None:
                continue
            total_eval += 1
            v1 = norm_val(field, s5[u].get(field) if isinstance(s5[u], dict) else None)
            v2 = norm_val(field, q4[u].get(field) if isinstance(q4[u], dict) else None)
            v3 = norm_val(field, q9[u].get(field) if isinstance(q9[u], dict) else None)
            vals = [v for v in [v1, v2, v3] if v is not None]
            if not vals:
                pred = None
            else:
                c = Counter(vals)
                top = c.most_common(1)[0]
                pred = top[0] if top[1] >= 2 else v3  # fallback Qwen-9B
            if pred == expected:
                correct += 1
        acuracia_ceap["Ensemble_Maioria_2_3"][field] = {
            "corretos": correct,
            "total": total_eval,
            "acuracia_pct": round(correct / total_eval * 100, 2),
        }

    # 8. Salvar JSON consolidado
    summary_data = {
        "metadata": {
            "data_analise": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_documentos_base": total_docs,
            "documentos_comuns_intersecao": n_common,
            "modelos_avaliados": list(FILES.keys()),
        },
        "cobertura": cobertura,
        "validacao_chave_sefaz_mod11": chave_stats,
        "concordancia_entre_modelos": concordancia,
        "acuracia_contra_ground_truth_ceap": acuracia_ceap,
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Salvo {OUT_JSON.name} ({OUT_JSON.stat().st_size / 1024:.1f} KB)")

    # 9. Salvar Markdown com relatório detalhado
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("# Relatório Comparativo de Modelos VLM: S5-nano vs Qwen3.5-4B vs Qwen3.5-9B\n\n")
        f.write(f"**Data da Análise:** {summary_data['metadata']['data_analise']}  \n")
        f.write(f"**Base Analisada:** {total_docs:,} Notas Fiscais Eletrônicas (NF-e) do CEAP  \n")
        f.write(f"**Documentos em Comum (3 Modelos):** {n_common:,} ({n_common/total_docs*100:.2f}%)  \n\n")

        f.write("## 1. Cobertura Geral e Taxas de Preenchimento\n\n")
        f.write("| Modelo | Docs Processados | Cobertura | Falhas de Parse (Raw) |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        for m, c in cobertura.items():
            f.write(f"| **{m}** | {c['total_docs']:,} | {c['percentage_base']}% | {c['raw_response_count']:,} ({c['raw_response_pct']}%) |\n")
        f.write("\n")

        f.write("### Preenchimento por Campo (Docs Comuns: 20.071)\n\n")
        f.write("| Campo | S5-nano | Qwen3.5-4B | Qwen3.5-9B |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        for fld in FIELDS:
            s5_p = cobertura["S5-nano"]["preenchimento_campos"][fld]["percentage"]
            q4_p = cobertura["Qwen3.5-4B"]["preenchimento_campos"][fld]["percentage"]
            q9_p = cobertura["Qwen3.5-9B"]["preenchimento_campos"][fld]["percentage"]
            f.write(f"| `{fld}` | {s5_p}% | {q4_p}% | {q9_p}% |\n")
        f.write("\n")

        f.write("## 2. Matriz de Concordância entre os Modelos\n\n")
        f.write("| Campo | Consenso (3/3) | Maioria (≥2/3) | Qwen-4B == Qwen-9B | S5-nano == Qwen-9B | S5-nano == Qwen-4B |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for fld in FIELDS:
            co = concordancia[fld]
            f.write(f"| **`{fld}`** | **{co['consenso_3_3_norm_pct']}%** | **{co['maioria_2_3_norm_pct']}%** | {co['qwen4b_eq_qwen9b_norm_pct']}% | {co['s5nano_eq_qwen9b_norm_pct']}% | {co['s5nano_eq_qwen4b_norm_pct']}% |\n")
        f.write("\n")

        f.write("## 3. Validação de Chave de Acesso (SEFAZ Módulo 11)\n\n")
        f.write("| Modelo | Chaves Extraídas | Com exatos 44 dígitos | Válidas no Módulo 11 SEFAZ |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        for m, ch in chave_stats.items():
            f.write(f"| **{m}** | {ch['total_chaves_extraidas']:,} | {ch['exatamente_44_digitos']:,} ({ch['exatamente_44_digitos_pct']}%) | **{ch['validas_sefaz_mod11']:,} ({ch['validas_sefaz_mod11_pct']}%)** |\n")
        f.write("\n")

        f.write("## 4. Acurácia Real contra o Ground Truth Oficial da Câmara (CEAP)\n\n")
        f.write("| Campo | S5-nano | Qwen3.5-4B | Qwen3.5-9B | **Ensemble (Maioria 2/3)** |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        for fld in ["valor_total_nota", "data_emissao", "cnpj_emitente"]:
            s5_a = acuracia_ceap["S5-nano"][fld]["acuracia_pct"]
            q4_a = acuracia_ceap["Qwen3.5-4B"][fld]["acuracia_pct"]
            q9_a = acuracia_ceap["Qwen3.5-9B"][fld]["acuracia_pct"]
            ens_a = acuracia_ceap["Ensemble_Maioria_2_3"][fld]["acuracia_pct"]
            f.write(f"| **`{fld}`** | {s5_a}% | {q4_a}% | {q9_a}% | **{ens_a}%** |\n")
        f.write("\n")

    print(f"[OK] Salvo {OUT_MD.name} ({OUT_MD.stat().st_size / 1024:.1f} KB)")
    print(f"\nFinalizado em {time.time()-t0:.2f}s!")


if __name__ == "__main__":
    main()
