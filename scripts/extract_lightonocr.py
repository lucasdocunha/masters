#!/usr/bin/env python3
"""
Pipeline de Extração de Texto com LightOnOCR-3-4B e Persistência no PostgreSQL.
Suporta múltiplas páginas por documento, aceleração dual-GPU via SLURM
e persistência incremental com commit a cada lote.
"""

import argparse
import io
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, List, Optional

import psycopg
import pymupdf
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

MODEL_ID = "lightonai/LightOnOCR-3-4B"
DEFAULT_DB_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/ceap")
DEFAULT_PDF_DIR = Path("/media/ssd2/datasets/CEAP")

stop_requested = False


def sig_handler(signum, frame):
    global stop_requested
    print(f"\n[AVISO] Sinal {signum} recebido. Concluindo o lote atual antes de encerrar...")
    stop_requested = True


signal.signal(signal.SIGINT, sig_handler)
signal.signal(signal.SIGTERM, sig_handler)


def get_pending_documents(
    conn: psycopg.Connection,
    gpu_id: int,
    num_gpus: int,
    years: Optional[List[int]] = None,
    sample_only: bool = False,
    limit: Optional[int] = None,
) -> List[dict]:
    """Consulta documentos pendentes que ainda não possuem texto completo extraído."""
    with conn.cursor() as cur:
        query = """
            SELECT d.ide_documento, d.ano, d.caminho_arquivo, d.num_paginas
            FROM documento d
            LEFT JOIN documento_texto_completo t 
                ON d.ide_documento = t.ide_documento 
               AND t.modelo = %s
            WHERE t.ide_documento IS NULL
              AND d.status IN ('baixado', 'processado')
        """
        params: List[Any] = [MODEL_ID]

        if num_gpus > 1:
            query += " AND (d.ide_documento %% %s) = %s"
            params.extend([num_gpus, gpu_id])

        if years:
            placeholders = ",".join(["%s"] * len(years))
            query += f" AND d.ano IN ({placeholders})"
            params.extend(years)

        query += " ORDER BY d.ano DESC, d.ide_documento ASC"

        if limit:
            query += f" LIMIT {limit}"

        cur.execute(query, params)
        rows = cur.fetchall()

        docs = [
            {
                "ide_documento": r[0],
                "ano": r[1],
                "caminho_arquivo": r[2],
                "num_paginas": r[3] or 1,
            }
            for r in rows
        ]

        if sample_only:
            # Filtrar apenas os que estão no sample_5pct se solicitado
            sample_file = Path("/home/lucas.ocunha/masters/data/latent_vectors_sample_5pct_metadata.json")
            if sample_file.exists():
                import json

                with open(sample_file, "r") as f:
                    sample_stems = set(Path(d["doc_name"]).stem for d in json.load(f).get("documents", []))
                docs = [d for d in docs if str(d["ide_documento"]) in sample_stems]

        return docs


def extract_page_text(
    model: Qwen3_5ForConditionalGeneration,
    processor: AutoProcessor,
    image: Image.Image,
    device: str,
    max_new_tokens: int = 1536,
) -> str:
    """Extrai texto formatado em markdown de uma imagem de página."""
    conversation = [{"role": "user", "content": [{"type": "image", "image": image}]}]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    gen_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    text = processor.decode(gen_ids, skip_special_tokens=True).strip()
    return text


def main():
    parser = argparse.ArgumentParser(description="Extração Massiva com LightOnOCR-3-4B")
    parser.add_argument("--gpu-id", type=int, default=0, help="ID da GPU local (0 ou 1)")
    parser.add_argument("--num-gpus", type=int, default=1, help="Total de GPUs/workers particionando os dados")
    parser.add_argument("--batch-size", type=int, default=16, help="Documentos por commit no PostgreSQL")
    parser.add_argument("--dpi", type=int, default=150, help="DPI para renderização do PDF (default: 150)")
    parser.add_argument("--limit", type=int, default=None, help="Limite máximo de documentos para este job")
    parser.add_argument("--years", type=int, nargs="+", default=None, help="Filtrar por anos específicos")
    parser.add_argument("--sample-only", action="store_true", help="Processar apenas a amostra de 5% (70k docs)")
    parser.add_argument("--db-url", type=str, default=DEFAULT_DB_URL, help="URL de conexão PostgreSQL")
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR, help="Diretório dos PDFs")
    args = parser.parse_args()

    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    print(f"=== Inicializando LightOnOCR-3-4B Worker (GPU {args.gpu_id}/{args.num_gpus}) ===")
    print(f"Dispositivo PyTorch: {device} ({torch.cuda.get_device_name(args.gpu_id) if torch.cuda.is_available() else 'CPU'})")

    # 1. Carregar Modelo e Processador
    t0 = time.time()
    print(f"Carregando modelo {MODEL_ID} em bfloat16 no dispositivo {device}...")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    print(f"Modelo e processador carregados em {time.time()-t0:.2f}s!")

    # 2. Conectar ao Banco e Obter Documentos Pendentes
    print(f"Conectando ao PostgreSQL ({args.db_url})...")
    conn = psycopg.connect(args.db_url, autocommit=False)

    pending_docs = get_pending_documents(
        conn,
        gpu_id=args.gpu_id,
        num_gpus=args.num_gpus,
        years=args.years,
        sample_only=args.sample_only,
        limit=args.limit,
    )

    print(f"Documentos pendentes encontrados para este worker: {len(pending_docs):,}")
    if not pending_docs:
        print("Nenhum documento pendente para processar.")
        conn.close()
        return

    # 3. Loop de Processamento
    total_docs_processed = 0
    total_pages_processed = 0
    pbar = tqdm(pending_docs, desc=f"GPU-{args.gpu_id} OCR", unit="doc")

    page_insert_sql = """
        INSERT INTO documento_ocr_pagina 
            (ide_documento, numero_pagina, modelo, texto_extraido, num_caracteres, tempo_processamento_ms)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (ide_documento, numero_pagina, modelo) 
        DO UPDATE SET 
            texto_extraido = EXCLUDED.texto_extraido,
            num_caracteres = EXCLUDED.num_caracteres,
            tempo_processamento_ms = EXCLUDED.tempo_processamento_ms,
            criado_em = NOW();
    """

    doc_insert_sql = """
        INSERT INTO documento_texto_completo 
            (ide_documento, modelo, total_paginas, texto_completo, num_caracteres_total, tempo_total_ms)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (ide_documento, modelo) 
        DO UPDATE SET 
            total_paginas = EXCLUDED.total_paginas,
            texto_completo = EXCLUDED.texto_completo,
            num_caracteres_total = EXCLUDED.num_caracteres_total,
            tempo_total_ms = EXCLUDED.tempo_total_ms,
            atualizado_em = NOW();
    """

    batch_docs = 0

    for doc_info in pbar:
        if stop_requested:
            print("\nInterrompendo de forma segura...")
            break

        doc_id = doc_info["ide_documento"]
        file_path = Path(doc_info["caminho_arquivo"])
        if not file_path.exists():
            # Tentar no diretório padrão se caminho absoluto estiver desatualizado
            file_path = args.pdf_dir / f"{doc_id}.pdf"

        if not file_path.exists():
            continue

        doc_t0 = time.time()
        try:
            pdf_doc = pymupdf.open(file_path)
            num_pages = len(pdf_doc)
            if num_pages == 0:
                pdf_doc.close()
                continue
        except Exception as e:
            # PDF corrompido ou ilegível
            continue

        pages_text = []
        with conn.cursor() as cur:
            for page_idx in range(num_pages):
                page_t0 = time.time()
                try:
                    page = pdf_doc[page_idx]
                    pix = page.get_pixmap(dpi=args.dpi)
                    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
                    text = extract_page_text(model, processor, img, device)
                except Exception as e:
                    text = f"[ERRO AO EXTRAIR PÁGINA {page_idx+1}: {e}]"

                page_ms = int((time.time() - page_t0) * 1000)
                pages_text.append(text)
                total_pages_processed += 1

                # Inserir página individual
                cur.execute(
                    page_insert_sql,
                    (doc_id, page_idx + 1, MODEL_ID, text, len(text), page_ms),
                )

            pdf_doc.close()

            # Consolidar texto completo
            doc_total_ms = int((time.time() - doc_t0) * 1000)
            full_text = "\n\n--- PÁGINA QUEBRA ---\n\n".join(pages_text)
            cur.execute(
                doc_insert_sql,
                (doc_id, MODEL_ID, num_pages, full_text, len(full_text), doc_total_ms),
            )

        total_docs_processed += 1
        batch_docs += 1

        if batch_docs >= args.batch_size:
            conn.commit()
            batch_docs = 0

        pbar.set_postfix({"pags": total_pages_processed, "docs": total_docs_processed})

    if batch_docs > 0:
        conn.commit()

    conn.close()
    print(f"\n[FINALIZADO GPU-{args.gpu_id}] Total Docs: {total_docs_processed:,} | Total Páginas: {total_pages_processed:,}")


if __name__ == "__main__":
    main()
