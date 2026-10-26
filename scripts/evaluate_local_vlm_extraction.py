#!/usr/bin/env python3
"""
Avaliação comparativa de extração estruturada de dados fiscais utilizando:
- Opção A (Two-Stage): LightOnOCR-3-4B (Visual -> Markdown) + Qwen2.5-VL-3B-Instruct (Markdown -> JSON)
- Opção B (End-to-End): Qwen2.5-VL-3B-Instruct direto (Visual -> JSON)

Mede acurácia (CNPJ, Valor, Data, Número, Fornecedor) contra o Ground Truth oficial do CEAP
e compara com o baseline do Google Gemini (gemini-3.5-flash-lite).
"""

import argparse
import io
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pymupdf
import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3_5ForConditionalGeneration,
)

# Importar métricas oficiais de avaliação do projeto
sys.path.append(str(Path(__file__).resolve().parent))
from evaluate_extraction import evaluate_single_doc, generate_markdown_report

MODEL_LIGHTON = "lightonai/LightOnOCR-3-4B"
MODEL_QWEN_VL = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_DATA_DIR = Path("/home/lucas.ocunha/masters/data/pdfs_selecionados")
DEFAULT_MANIFEST = DEFAULT_DATA_DIR / "manifest.json"

PROMPT_SISTEMA_FISCAL = """Você é um especialista em auditoria fiscal e análise de documentos fiscais brasileiros (NF-e, NFC-e, NFS-e, DANFE, cupons fiscais, bilhetes de passagem aérea/terrestre e recibos).
Analise o documento fiscal e extraia com máxima fidelidade os campos solicitados no schema JSON:
- fornecedor_nome: Razão social ou nome fantasia do estabelecimento emissor
- fornecedor_cnpj_cpf: CNPJ ou CPF do emissor (formatado ou apenas dígitos)
- numero_documento: Número da nota fiscal, DANFE, cupom fiscal, bilhete ou recibo
- data_emissao: Data de emissão no formato padronizado YYYY-MM-DD
- valor_total: Valor total expresso no documento fiscal (numérico float)
- descricao_resumida: Breve resumo dos produtos ou serviços descritos
- observacoes: Observações relevantes (ou null)
Se algum dado estiver completamente ilegível ou ausente, retorne null no respectivo campo.
Responda EXCLUSIVAMENTE em formato JSON válido com as chaves indicadas."""


def parse_json_from_response(text: str) -> Dict[str, Any]:
    """Extrai e valida um dicionário JSON a partir da resposta de texto de um modelo."""
    text = text.strip()
    # Remover blocos markdown se existirem (```json ... ```)
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    # Procurar primeiro '{' e último '}'
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    return {
        "fornecedor_nome": None,
        "fornecedor_cnpj_cpf": None,
        "numero_documento": None,
        "data_emissao": None,
        "valor_total": None,
        "descricao_resumida": None,
        "observacoes": f"Erro ao decodificar JSON: {text[:120]}",
    }


def render_pdf_first_page(pdf_path: Path, dpi: int = 150) -> Image.Image:
    """Renderiza a primeira página de um PDF em uma imagem PIL."""
    doc = pymupdf.open(pdf_path)
    page = doc[0]
    pix = page.get_pixmap(dpi=dpi)
    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    doc.close()
    return img


def run_twostage_extraction(
    pdf_path: Path,
    lighton_model: Qwen3_5ForConditionalGeneration,
    lighton_proc: AutoProcessor,
    qwen_model: Qwen2_5_VLForConditionalGeneration,
    qwen_proc: AutoProcessor,
    device: str,
) -> tuple[Dict[str, Any], float]:
    """Estágio 1 (LightOnOCR -> Markdown) + Estágio 2 (Qwen LLM -> JSON)."""
    t0 = time.time()
    img = render_pdf_first_page(pdf_path)

    # 1. LightOnOCR transcrição Markdown
    conv_ocr = [{"role": "user", "content": [{"type": "image", "image": img}]}]
    prompt_ocr = lighton_proc.apply_chat_template(conv_ocr, add_generation_prompt=True, tokenize=False)
    inputs_ocr = lighton_proc(text=[prompt_ocr], images=[img], return_tensors="pt").to(device)

    with torch.inference_mode():
        out_ocr = lighton_model.generate(**inputs_ocr, max_new_tokens=1024, do_sample=False)
    gen_ocr = out_ocr[0, inputs_ocr["input_ids"].shape[1] :]
    markdown_text = lighton_proc.decode(gen_ocr, skip_special_tokens=True).strip()

    # 2. Qwen2.5 LLM estruturação para JSON
    conv_llm = [
        {"role": "system", "content": PROMPT_SISTEMA_FISCAL},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Texto do documento fiscal transcrito:\n\n{markdown_text}\n\nExtraia os dados em JSON conforme instruído:",
                }
            ],
        },
    ]
    prompt_llm = qwen_proc.apply_chat_template(conv_llm, add_generation_prompt=True, tokenize=False)
    inputs_llm = qwen_proc(text=[prompt_llm], return_tensors="pt").to(device)

    with torch.inference_mode():
        out_llm = qwen_model.generate(**inputs_llm, max_new_tokens=384, do_sample=False)
    gen_llm = out_llm[0, inputs_llm["input_ids"].shape[1] :]
    json_text = qwen_proc.decode(gen_llm, skip_special_tokens=True).strip()

    data = parse_json_from_response(json_text)
    elapsed = time.time() - t0
    return data, elapsed


def run_qwenvl_direct_extraction(
    pdf_path: Path,
    qwen_model: Qwen2_5_VLForConditionalGeneration,
    qwen_proc: AutoProcessor,
    device: str,
) -> tuple[Dict[str, Any], float]:
    """Estágio Único End-to-End: Imagem do PDF -> JSON diretamente."""
    t0 = time.time()
    img = render_pdf_first_page(pdf_path)

    conversation = [
        {"role": "system", "content": PROMPT_SISTEMA_FISCAL},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "Extraia as informações deste documento fiscal conforme o schema estruturado JSON."},
            ],
        },
    ]
    prompt = qwen_proc.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = qwen_proc(text=[prompt], images=[img], return_tensors="pt").to(device)

    with torch.inference_mode():
        output_ids = qwen_model.generate(**inputs, max_new_tokens=384, do_sample=False)
    gen_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    res_text = qwen_proc.decode(gen_ids, skip_special_tokens=True).strip()

    data = parse_json_from_response(res_text)
    elapsed = time.time() - t0
    return data, elapsed


def print_summary_table(title: str, results: List[Dict[str, Any]], total: int, avg_time: float):
    """Exibe no terminal a tabela comparativa de acurácia idêntica ao relatório do Gemini."""
    cnpj_hits = sum(1 for r in results if r["cnpj"]["match"])
    val_hits = sum(1 for r in results if r["valor"]["match"])
    data_hits = sum(1 for r in results if r["data"]["match"])
    num_hits = sum(1 for r in results if r["numero"]["match"])
    forn_hits = sum(1 for r in results if r["fornecedor"]["match"])
    perf_hits = sum(1 for r in results if r["match_perfeito"])

    pct = lambda x: f"{(x / total) * 100:.1f}%" if total > 0 else "0.0%"

    print("\n" + "=" * 65)
    print(f" {title.upper()} (Amostra: {total} docs | Média: {avg_time:.2f}s/doc)")
    print("=" * 65)
    print(f"{'Campo Fiscal':<35} | {'Acertos':<8} | {'Total':<6} | {'Acurácia':<8}")
    print("-" * 65)
    print(f"{'CNPJ / CPF do Fornecedor':<35} | {cnpj_hits:<8} | {total:<6} | {pct(cnpj_hits):<8}")
    print(f"{'Valor Total do Documento':<35} | {val_hits:<8} | {total:<6} | {pct(val_hits):<8}")
    print(f"{'Data de Emissão':<35} | {data_hits:<8} | {total:<6} | {pct(data_hits):<8}")
    print(f"{'Número da Nota / Documento':<35} | {num_hits:<8} | {total:<6} | {pct(num_hits):<8}")
    print(f"{'Nome / Razão Social':<35} | {forn_hits:<8} | {total:<6} | {pct(forn_hits):<8}")
    print("-" * 65)
    print(f"{'MATCH PERFEITO (Todos os 5)':<35} | {perf_hits:<8} | {total:<6} | {pct(perf_hits):<8}")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Avaliação de Modelos VLM Locais vs Gemini")
    parser.add_argument("--method", choices=["twostage", "qwenvl", "both"], default="both", help="Abordagem a avaliar")
    parser.add_argument("--limit", type=int, default=20, help="Quantidade de documentos da amostra a testar (default: 20)")
    parser.add_argument("--gpu-id", type=int, default=0, help="ID da GPU local (0 ou 1)")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    print(f"=== INICIANDO BENCHMARK VLM LOCAL vs GEMINI ({device}) ===")
    print(f"Método(s): {args.method} | Limite: {args.limit} documentos")

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    docs = manifest.get("documents", [])[: args.limit]

    # Carregar modelo Qwen2.5-VL-3B-Instruct (usado tanto no end-to-end quanto no estágio 2 do two-stage)
    print(f"\nCarregando {MODEL_QWEN_VL} em {device}...")
    qwen_proc = AutoProcessor.from_pretrained(MODEL_QWEN_VL)
    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_QWEN_VL,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    print(f"Qwen2.5-VL-3B-Instruct pronto!")

    # Carregar LightOnOCR apenas se necessário
    lighton_model = None
    lighton_proc = None
    if args.method in ("twostage", "both"):
        print(f"\nCarregando {MODEL_LIGHTON} em {device}...")
        lighton_proc = AutoProcessor.from_pretrained(MODEL_LIGHTON)
        lighton_model = Qwen3_5ForConditionalGeneration.from_pretrained(
            MODEL_LIGHTON,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        print("LightOnOCR-3-4B pronto!")

    # 1. Execução Opção B (End-to-End Qwen2.5-VL)
    if args.method in ("qwenvl", "both"):
        print("\n>>> EXECUTANDO OPÇÃO B: QWEN2.5-VL-3B-INSTRUCT (END-TO-END) <<<")
        results_qwenvl = []
        times_qwenvl = []

        for item in tqdm(docs, desc="Opção B (Qwen-VL)"):
            doc_id = item["idDocumento"]
            pdf_path = DEFAULT_DATA_DIR / f"{doc_id}.pdf"
            exp_path = DEFAULT_DATA_DIR / f"{doc_id}_esperado.json"
            if not pdf_path.exists() or not exp_path.exists():
                continue

            with open(exp_path, "r", encoding="utf-8") as f:
                expected = json.load(f)

            try:
                extracted, elapsed = run_qwenvl_direct_extraction(pdf_path, qwen_model, qwen_proc, device)
                times_qwenvl.append(elapsed)
                res = evaluate_single_doc(expected, extracted)
                results_qwenvl.append(res)
            except Exception as e:
                print(f" [Erro doc {doc_id}]: {e}")

        avg_t_qwenvl = sum(times_qwenvl) / len(times_qwenvl) if times_qwenvl else 0
        print_summary_table("Opção B: Qwen2.5-VL-3B-Instruct (End-to-End)", results_qwenvl, len(results_qwenvl), avg_t_qwenvl)

    # 2. Execução Opção A (Two-Stage LightOnOCR + Qwen)
    if args.method in ("twostage", "both"):
        print("\n>>> EXECUTANDO OPÇÃO A: LIGHTONOCR-3-4B + QWEN (TWO-STAGE) <<<")
        results_twostage = []
        times_twostage = []

        for item in tqdm(docs, desc="Opção A (Two-Stage)"):
            doc_id = item["idDocumento"]
            pdf_path = DEFAULT_DATA_DIR / f"{doc_id}.pdf"
            exp_path = DEFAULT_DATA_DIR / f"{doc_id}_esperado.json"
            if not pdf_path.exists() or not exp_path.exists():
                continue

            with open(exp_path, "r", encoding="utf-8") as f:
                expected = json.load(f)

            try:
                extracted, elapsed = run_twostage_extraction(
                    pdf_path, lighton_model, lighton_proc, qwen_model, qwen_proc, device
                )
                times_twostage.append(elapsed)
                res = evaluate_single_doc(expected, extracted)
                results_twostage.append(res)
            except Exception as e:
                print(f" [Erro doc {doc_id}]: {e}")

        avg_t_twostage = sum(times_twostage) / len(times_twostage) if times_twostage else 0
        print_summary_table("Opção A: LightOnOCR + Qwen (Two-Stage)", results_twostage, len(results_twostage), avg_t_twostage)

    # 3. Baseline do Gemini na mesma amostra
    print("\n>>> BASELINE GEMINI (Amostra correspondente) <<<")
    results_gemini = []
    for item in docs:
        doc_id = item["idDocumento"]
        exp_path = DEFAULT_DATA_DIR / f"{doc_id}_esperado.json"
        gemini_path = DEFAULT_DATA_DIR / f"{doc_id}_extraido.json"
        if exp_path.exists() and gemini_path.exists():
            with open(exp_path, "r", encoding="utf-8") as f:
                exp = json.load(f)
            with open(gemini_path, "r", encoding="utf-8") as f:
                ext = json.load(f)
            results_gemini.append(evaluate_single_doc(exp, ext))

    print_summary_table("Baseline: Google Gemini 2.5 Flash Lite", results_gemini, len(results_gemini), 2.20)


if __name__ == "__main__":
    main()
