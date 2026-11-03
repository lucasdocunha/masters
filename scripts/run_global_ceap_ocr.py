#!/home/lucas.ocunha/masters/.venv_ocr/bin/python3
"""
Pipeline Global de Extração de Texto e OCR para Todo o Acervo CEAP (1.4M documentos).
Estratégia Híbrida de Alta Performance em 2 Fases:
  • Fase 1 (Fast-Track Digital): 
      - Varre todos os PDFs pendentes em paralelo com PyMuPDF.
      - Extrai texto digital nativo instantaneamente (~3.000 docs/s).
      - Persiste diretamente no PostgreSQL (modelo='digital_pdf').
  • Fase 2 (RapidOCR Streaming em Memória):
      - Processa os documentos escaneados/rasterizados remanescentes com 24 workers.
      - Renderização in-memory (sem gerar centenas de gigabytes de PNGs temporários).
      - Commits periódicos em batches no PostgreSQL (modelo='rapidocr').
      - Checkpoint automático (resumível a qualquer momento).
"""

import os
import sys
import json
import time
import signal
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pymupdf
import psycopg

# Evitar concorrência excessiva de threads internas no ONNX / OpenCV
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

BASE_DIR = Path("/home/lucas.ocunha/masters")
DATA_DIR = BASE_DIR / "data"
PDF_DIR = Path("/media/ssd2/datasets/CEAP")
HIST_FILE = BASE_DIR / "ml_backend" / "models" / "annotations_history.json"
META_PATH = DATA_DIR / "latent_vectors_full_metadata.json"
EMBS_PATH = DATA_DIR / "latent_vectors_metric_learned_128d.npy"
YEAR_FILE = DATA_DIR / "pdf_to_year.json"


def sort_candidates_by_distance_to_top_classes(candidate_ids, min_examples=2000):
    """
    Ordena os documentos candidatos pela distância angular aos centróides das classes mais abundantes.
    Documentos mais distantes / desconhecidos ficam no início da fila de OCR.
    """
    if not candidate_ids:
        return candidate_ids

    if not (HIST_FILE.exists() and META_PATH.exists() and EMBS_PATH.exists()):
        print("   [AVISO] Arquivos métricos não encontrados. Mantendo ordem padrão.")
        return candidate_ids

    t0 = time.time()
    print("\n   Calculando prioridade de OCR (mais distantes dos centróides mais frequentes)...")

    with open(HIST_FILE, "r", encoding="utf-8") as f:
        hist = json.load(f)
    norm_hist = {k.replace(".pdf", ""): v for k, v in hist.items()}

    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    docs = meta["documents"]
    stems = [d["doc_name"].replace(".pdf", "") for d in docs]
    stem_to_idx = {s: i for i, s in enumerate(stems)}

    embs = np.load(EMBS_PATH)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs /= norms

    from collections import Counter
    counts = Counter(norm_hist.values())
    top_classes = [cls for cls, cnt in counts.most_common() if cnt >= min_examples]
    print(f"   • Classes dominantes como referência ({len(top_classes)} classes com >={min_examples} exemplos):")
    for cls in top_classes:
        print(f"     - {cls:15s}: {counts[cls]:6,d} exemplos")

    top_centroids = []
    for cls in top_classes:
        c_idxs = [stem_to_idx[s] for s, lbl in norm_hist.items() if lbl == cls and s in stem_to_idx]
        cent = embs[c_idxs].mean(axis=0)
        cent /= np.linalg.norm(cent)
        top_centroids.append(cent)
    top_centroids = np.array(top_centroids)  # shape: (K, 128)

    # Similaridade máxima de cada documento com qualquer uma das classes dominantes
    sims = np.dot(embs, top_centroids.T)  # shape: (N, K)
    max_sim = sims.max(axis=1)            # shape: (N,)

    # Mapear cada candidate_id para o seu índice e score
    cand_set = set(candidate_ids)
    cands_with_scores = []
    cands_without_meta = []

    for doc_id in candidate_ids:
        s = str(doc_id)
        if s in stem_to_idx:
            idx = stem_to_idx[s]
            cands_with_scores.append((doc_id, float(max_sim[idx])))
        else:
            cands_without_meta.append(doc_id)

    # Ordenar por menor similaridade máxima (ou seja, maior distância aos clusters dominantes)
    cands_with_scores.sort(key=lambda x: x[1])
    sorted_ids = [doc_id for doc_id, score in cands_with_scores] + cands_without_meta

    print(f"   ✓ Ordenação ativa concluída em {time.time() - t0:.2f}s para {len(candidate_ids):,} candidatos!")
    if cands_with_scores:
        top_1 = cands_with_scores[0]
        top_last = cands_with_scores[-1]
        print(f"   • 1º da fila (mais distante): Doc {top_1[0]} | Similaridade máx: {top_1[1]*100:.2f}% (Distância: {(1-top_1[1])*100:.2f}%)")
        print(f"   • Último da fila (mais próximo): Doc {top_last[0]} | Similaridade máx: {top_last[1]*100:.2f}% (Distância: {(1-top_last[1])*100:.2f}%)")

    return sorted_ids
DB_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/ceap")

stop_requested = False


def sig_handler(signum, frame):
    global stop_requested
    print(f"\n[AVISO] Sinal de encerramento ({signum}) recebido. Concluindo lote atual e encerrando com segurança...", flush=True)
    stop_requested = True


signal.signal(signal.SIGINT, sig_handler)
signal.signal(signal.SIGTERM, sig_handler)

# ------------------------------------------------------------------------------
# Funções de Extração Digital (Fase 1)
# ------------------------------------------------------------------------------

def extract_digital_worker(doc_id: int):
    pdf_p = PDF_DIR / f"{doc_id}.pdf"
    if not pdf_p.exists():
        return doc_id, None, 0, 0, 0

    t0 = time.time()
    try:
        doc = pymupdf.open(str(pdf_p))
        n_pages = len(doc)
        if n_pages == 0:
            doc.close()
            return doc_id, None, 0, 0, 0

        # Ler página 1
        txt = doc[0].get_text().strip().replace("\x00", "")
        doc.close()
        t_ms = max(1, int((time.time() - t0) * 1000))

        if len(txt) >= 35:
            alnum = sum(1 for c in txt if c.isalnum() or c.isspace())
            alpha = sum(1 for c in txt if c.isalpha())
            if (alnum / len(txt)) >= 0.65 and alpha >= 15:
                return doc_id, txt, len(txt), n_pages, t_ms

        return doc_id, None, 0, n_pages, t_ms
    except Exception:
        return doc_id, None, 0, 0, 0


# ------------------------------------------------------------------------------
# Funções de RapidOCR em Memória (Fase 2)
# ------------------------------------------------------------------------------

engine = None

def init_ocr_worker():
    global engine
    from rapidocr_onnxruntime import RapidOCR
    engine = RapidOCR(use_cls=False, det_limit_side_len=640, intra_op_num_threads=1, inter_op_num_threads=1)


def process_ocr_memory_task(doc_id: int):
    global engine
    pdf_p = PDF_DIR / f"{doc_id}.pdf"
    if not pdf_p.exists():
        return doc_id, "", 0, 0, 0

    t0 = time.time()
    try:
        doc = pymupdf.open(str(pdf_p))
        n_pages = len(doc)
        if n_pages == 0:
            doc.close()
            return doc_id, "", 0, 0, 0

        # Renderizar página 1 diretamente em buffer de memória RGB (zero disco)
        pix = doc[0].get_pixmap(dpi=110, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
        doc.close()

        res, _ = engine(img)
        t_ms = max(1, int((time.time() - t0) * 1000))

        if not res:
            return doc_id, "", 0, n_pages, t_ms

        full_txt = " ".join([line[1] for line in res]).replace("\x00", "")
        return doc_id, full_txt, len(full_txt), n_pages, t_ms
    except Exception:
        t_ms = max(1, int((time.time() - t0) * 1000))
        return doc_id, "", 0, 1, t_ms


# ------------------------------------------------------------------------------
# Banco de Dados
# ------------------------------------------------------------------------------

def get_already_processed_ids():
    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT ide_documento FROM documento_texto_completo")
            rows = cur.fetchall()
            return set(r[0] for r in rows)


def commit_db_batch(batch_rows, model_name, pdf_to_year):
    if not batch_rows:
        return

    # Sanitizar textos para remover NUL bytes (0x00) incompatíveis com PostgreSQL UTF-8
    clean_batch = []
    for doc_id, txt, char_count, n_pages, t_ms in batch_rows:
        clean_txt = (txt or "").replace("\x00", "")
        clean_batch.append((doc_id, clean_txt, len(clean_txt), n_pages, t_ms))
    batch_rows = clean_batch

    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            # 1. documento
            doc_data = []
            for doc_id, txt, char_count, n_pages, t_ms in batch_rows:
                ano = pdf_to_year.get(f"{doc_id}.pdf", 2024)
                caminho = f"/media/ssd2/datasets/CEAP/{doc_id}.pdf"
                doc_data.append((doc_id, ano, caminho, "baixado"))

            cur.executemany("""
                INSERT INTO documento (ide_documento, ano, caminho_arquivo, status)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (ide_documento) DO NOTHING;
            """, doc_data)

            # 2. documento_ocr_pagina
            page_data = [
                (doc_id, 1, model_name, txt, char_count, t_ms)
                for doc_id, txt, char_count, n_pages, t_ms in batch_rows
            ]
            cur.executemany("""
                INSERT INTO documento_ocr_pagina 
                    (ide_documento, numero_pagina, modelo, texto_extraido, num_caracteres, tempo_processamento_ms)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (ide_documento, numero_pagina, modelo) 
                DO UPDATE SET 
                    texto_extraido = EXCLUDED.texto_extraido,
                    num_caracteres = EXCLUDED.num_caracteres,
                    tempo_processamento_ms = EXCLUDED.tempo_processamento_ms,
                    criado_em = NOW();
            """, page_data)

            # 3. documento_texto_completo
            full_data = [
                (doc_id, model_name, n_pages, txt, char_count, t_ms)
                for doc_id, txt, char_count, n_pages, t_ms in batch_rows
            ]
            cur.executemany("""
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
            """, full_data)

            conn.commit()


# ------------------------------------------------------------------------------
# Execução Principal
# ------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Pipeline Global de OCR CEAP (1.4M documentos).")
    parser.add_argument("--phase", choices=["digital", "ocr", "all"], default="all", help="Fase a executar: 'digital', 'ocr' ou 'all' (default: all)")
    parser.add_argument("--workers", type=int, default=24, help="Número de workers paralelos (default: 24)")
    parser.add_argument("--batch-size", type=int, default=1000, help="Tamanho do lote para commit no banco (default: 1000)")
    parser.add_argument("--priority", choices=["farthest_from_top", "id_asc"], default="farthest_from_top", help="Ordem da fila: 'farthest_from_top' (mais distantes dos centróides com mais exemplos) ou 'id_asc' (default: farthest_from_top)")
    args = parser.parse_args()

    print("=" * 76)
    print(" PIPELINE GLOBAL DE OCR DO ACERVO CEAP (1.4M DOCUMENTOS)")
    print(f" Modo: {args.phase.upper()} | Prioridade: {args.priority.upper()} | Workers: {args.workers} | Batch DB: {args.batch_size}")
    print("=" * 76)

    # 1. Carregar lista de PDFs do disco
    t_start = time.time()
    print("\n1. Verificando documentos no SSD (/media/ssd2/datasets/CEAP)...")
    all_pdf_files = list(PDF_DIR.glob("*.pdf"))
    all_doc_ids = [int(p.stem) for p in all_pdf_files if p.stem.isdigit()]
    all_doc_ids.sort()
    print(f"   ✓ Total de PDFs encontrados no disco: {len(all_doc_ids):,}")

    with open(YEAR_FILE, "r", encoding="utf-8") as f:
        pdf_to_year = json.load(f)

    # 2. Verificar banco de dados
    print("\n2. Consultando documentos já extraídos no PostgreSQL...")
    processed_ids = get_already_processed_ids()
    print(f"   ✓ Documentos já presentes na tabela documento_texto_completo: {len(processed_ids):,}")

    pending_ids = [doc_id for doc_id in all_doc_ids if doc_id not in processed_ids]
    print(f"   ✓ Documentos pendentes de extração: {len(pending_ids):,}")

    if not pending_ids:
        print("\n[SUCESSO] Todos os 1.4M documentos já possuem texto extraído no banco!")
        return

    # ==========================================================================
    # FASE 1: Fast-Track Digital Sweep
    # ==========================================================================
    scanned_candidates = []
    if args.phase in ["digital", "all"]:
        print("\n" + "=" * 76)
        print(f" FASE 1: FAST-TRACK DIGITAL SWEEP ({len(pending_ids):,} PDFs)")
        print("=" * 76)
        t_phase1 = time.time()
        digital_extracted = 0
        current_batch = []

        with ProcessPoolExecutor(max_workers=min(args.workers, 20)) as ex:
            for i, res in enumerate(ex.map(extract_digital_worker, pending_ids, chunksize=250), 1):
                if stop_requested:
                    break

                doc_id, txt, char_count, n_pages, t_ms = res
                if txt is not None:
                    current_batch.append((doc_id, txt, char_count, n_pages, t_ms))
                    digital_extracted += 1
                else:
                    scanned_candidates.append(doc_id)

                if len(current_batch) >= args.batch_size:
                    commit_db_batch(current_batch, "digital_pdf", pdf_to_year)
                    current_batch = []

                if i % 25000 == 0 or i == len(pending_ids):
                    el = time.time() - t_phase1
                    rate = i / max(el, 0.1)
                    rem = (len(pending_ids) - i) / max(rate, 0.1)
                    print(f"   [Digital] {i:,}/{len(pending_ids):,} ({rate:.1f} docs/s | Restante: {rem/60:.1f} min) | Digitais encontrados: {digital_extracted:,}", flush=True)

        if current_batch:
            commit_db_batch(current_batch, "digital_pdf", pdf_to_year)

        print(f"\n   ✓ Fase 1 concluída em {time.time() - t_phase1:.1f}s!")
        print(f"   • Textos digitais gravados no PostgreSQL: {digital_extracted:,}")
        print(f"   • Documentos escaneados para OCR: {len(scanned_candidates):,}")

    # ==========================================================================
    # FASE 2: RapidOCR Streaming em Memória
    # ==========================================================================
    if args.phase in ["ocr", "all"] and not stop_requested:
        if args.phase == "ocr" or not scanned_candidates:
            # Reconsultar do banco
            processed_ids = get_already_processed_ids()
            scanned_candidates = [doc_id for doc_id in all_doc_ids if doc_id not in processed_ids]

        # Aplicar ordenação ativa: priorizar os mais distantes dos centróides mais frequentes
        if args.priority == "farthest_from_top":
            scanned_candidates = sort_candidates_by_distance_to_top_classes(scanned_candidates, min_examples=2000)

        print("\n" + "=" * 76)
        print(f" FASE 2: RAPIDOCR STREAMING EM MEMÓRIA ({len(scanned_candidates):,} ESCANEADOS)")
        print("=" * 76)
        print(f"   Iniciando {args.workers} workers com renderização direta em RAM (zero disco)...")

        t_phase2 = time.time()
        ocr_batch = []
        processed_ocr = 0

        with ProcessPoolExecutor(max_workers=args.workers, initializer=init_ocr_worker) as ex:
            for res in ex.map(process_ocr_memory_task, scanned_candidates, chunksize=25):
                if stop_requested:
                    break

                doc_id, txt, char_count, n_pages, t_ms = res
                ocr_batch.append((doc_id, txt, char_count, n_pages, t_ms))
                processed_ocr += 1

                if len(ocr_batch) >= min(args.batch_size, 500):
                    commit_db_batch(ocr_batch, "rapidocr", pdf_to_year)
                    ocr_batch = []

                if processed_ocr % 500 == 0 or processed_ocr == len(scanned_candidates):
                    el = time.time() - t_phase2
                    rate = processed_ocr / max(el, 0.1)
                    rem = (len(scanned_candidates) - processed_ocr) / max(rate, 0.1)
                    print(f"   [RapidOCR] {processed_ocr:,}/{len(scanned_candidates):,} ({rate:.1f} docs/s | Restante: {rem/60:.1f} min)", flush=True)

        if ocr_batch:
            commit_db_batch(ocr_batch, "rapidocr", pdf_to_year)

        print(f"\n   ✓ Fase 2 concluída em {time.time() - t_phase2:.1f}s!")

    print("\n" + "=" * 76)
    print(f" PIPELINE FINALIZADO EM {time.time() - t_start:.1f}s!")
    print("=" * 76)


if __name__ == "__main__":
    main()
