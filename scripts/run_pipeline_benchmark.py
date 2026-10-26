#!/usr/bin/env python3
"""
Execução em Lote de Extração Fiscal para Avaliação de Modelos Locais vs Gemini:
- Opção B (qwenvl): Qwen2.5-VL-3B-Instruct (End-to-End Visual -> JSON)
- Opção A (twostage): LightOnOCR-3-4B (Visual -> Markdown) + Qwen2.5-VL-3B-Instruct (Markdown -> JSON)

Suporta paralelização multi-GPU (SLURM), persistência incremental e retomada automática.
"""

import argparse
import io
import json
import os
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

MODEL_LIGHTON = "lightonai/LightOnOCR-3-4B"
MODEL_QWEN_VL = "Qwen/Qwen2.5-VL-3B-Instruct"

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
    """Decodifica com segurança o JSON gerado pelo modelo."""
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

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
        "observacoes": f"Erro parsing JSON: {text[:100]}",
    }


def render_pdf_first_page(pdf_path: Path, dpi: int = 150) -> Image.Image:
    """Renderiza a primeira página do PDF para imagem PIL."""
    doc = pymupdf.open(pdf_path)
    page = doc[0]
    pix = page.get_pixmap(dpi=dpi)
    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    doc.close()
    return img


def run_worker(
    method: str,
    gpu_id: int,
    num_gpus: int,
    input_dir: Path,
    manifest_path: Path,
    limit: Optional[int] = None,
):
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    print(f"=== INICIANDO WORKER [{method.upper()}] (GPU {gpu_id}/{num_gpus} - {device}) ===")

    out_subdir = input_dir / f"extraidos_{method}"
    out_subdir.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    all_docs = manifest.get("documents", [])
    if limit:
        all_docs = all_docs[:limit]

    # Particionamento por GPU
    worker_docs = [d for idx, d in enumerate(all_docs) if (idx % num_gpus) == gpu_id]
    print(f"Documentos atribuídos a este worker: {len(worker_docs):,} (de {len(all_docs):,} totais)")

    # 1. Carregar Modelos
    qwen_model = None
    qwen_proc = None
    lighton_model = None
    lighton_proc = None

    t0 = time.time()
    print(f"[Worker-{gpu_id}] Carregando {MODEL_QWEN_VL} em {device}...")
    qwen_proc = AutoProcessor.from_pretrained(MODEL_QWEN_VL)
    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_QWEN_VL,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )

    if method == "twostage":
        print(f"[Worker-{gpu_id}] Carregando {MODEL_LIGHTON} em {device}...")
        lighton_proc = AutoProcessor.from_pretrained(MODEL_LIGHTON)
        lighton_model = Qwen3_5ForConditionalGeneration.from_pretrained(
            MODEL_LIGHTON,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )

    print(f"[Worker-{gpu_id}] Modelos carregados com sucesso em {time.time()-t0:.2f}s!")

    # 2. Loop de Processamento
    processed_count = 0
    skipped_count = 0
    pbar = tqdm(worker_docs, desc=f"GPU-{gpu_id} [{method}]", unit="doc")

    for item in pbar:
        doc_id = item["idDocumento"]
        pdf_path = input_dir / f"{doc_id}.pdf"
        out_json_path = out_subdir / f"{doc_id}.json"

        if out_json_path.exists():
            skipped_count += 1
            continue

        if not pdf_path.exists():
            continue

        t_start = time.time()
        try:
            img = render_pdf_first_page(pdf_path)

            if method == "qwenvl":
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
                    out_ids = qwen_model.generate(**inputs, max_new_tokens=384, do_sample=False)
                gen_ids = out_ids[0, inputs["input_ids"].shape[1] :]
                res_text = qwen_proc.decode(gen_ids, skip_special_tokens=True).strip()
                data = parse_json_from_response(res_text)

            elif method == "twostage":
                # Estágio 1: LightOnOCR para Markdown
                conv_ocr = [{"role": "user", "content": [{"type": "image", "image": img}]}]
                prompt_ocr = lighton_proc.apply_chat_template(conv_ocr, add_generation_prompt=True, tokenize=False)
                inputs_ocr = lighton_proc(text=[prompt_ocr], images=[img], return_tensors="pt").to(device)

                with torch.inference_mode():
                    out_ocr = lighton_model.generate(**inputs_ocr, max_new_tokens=1024, do_sample=False)
                gen_ocr = out_ocr[0, inputs_ocr["input_ids"].shape[1] :]
                md_text = lighton_proc.decode(gen_ocr, skip_special_tokens=True).strip()

                # Estágio 2: Qwen LLM para JSON
                conv_llm = [
                    {"role": "system", "content": PROMPT_SISTEMA_FISCAL},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Texto do documento fiscal transcrito:\n\n{md_text}\n\nExtraia os dados em JSON conforme instruído:"}
                        ],
                    },
                ]
                prompt_llm = qwen_proc.apply_chat_template(conv_llm, add_generation_prompt=True, tokenize=False)
                inputs_llm = qwen_proc(text=[prompt_llm], return_tensors="pt").to(device)

                with torch.inference_mode():
                    out_llm = qwen_model.generate(**inputs_llm, max_new_tokens=384, do_sample=False)
                gen_llm = out_llm[0, inputs_llm["input_ids"].shape[1] :]
                res_text = qwen_proc.decode(gen_llm, skip_special_tokens=True).strip()
                data = parse_json_from_response(res_text)

            elapsed = round(time.time() - t_start, 2)
            data["_metadados_extracao"] = {
                "modelo": "Qwen2.5-VL-3B-Instruct" if method == "qwenvl" else "LightOnOCR-3-4B + Qwen2.5-VL-3B",
                "metodo": method,
                "gpu_id": gpu_id,
                "tempo_segundos": elapsed,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "idDocumento": doc_id,
            }

            with open(out_json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            processed_count += 1
            pbar.set_postfix({"feitos": processed_count, "pulados": skipped_count})

        except Exception as e:
            print(f" [Worker-{gpu_id}] Erro no doc {doc_id}: {e}")

    print(f"\n[Worker-{gpu_id} CONCLUÍDO] Total processados: {processed_count:,} | Pulados: {skipped_count:,}")


def main():
    parser = argparse.ArgumentParser(description="Runner de Benchmark de Pipelines VLM Locais")
    parser.add_argument("--method", choices=["qwenvl", "twostage"], required=True, help="Método de extração")
    parser.add_argument("--gpu-id", type=int, default=0, help="ID da GPU local")
    parser.add_argument("--num-gpus", type=int, default=1, help="Total de GPUs em paralelo")
    parser.add_argument("--limit", type=int, default=None, help="Limitar total de documentos")
    parser.add_argument("--input-dir", type=Path, default=Path("/home/lucas.ocunha/masters/data/pdfs_selecionados"))
    parser.add_argument("--manifest", type=Path, default=Path("/home/lucas.ocunha/masters/data/pdfs_selecionados/manifest.json"))
    args = parser.parse_args()

    run_worker(
        method=args.method,
        gpu_id=args.gpu_id,
        num_gpus=args.num_gpus,
        input_dir=args.input_dir,
        manifest_path=args.manifest,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
