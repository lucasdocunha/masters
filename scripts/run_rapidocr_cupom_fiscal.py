#!/home/lucas.ocunha/masters/.venv_ocr/bin/python3
"""
Pipeline de Extração RapidOCR e Classificação para CUPOM FISCAL.
1. Identifica os candidatos a CUPOM FISCAL com similaridade >= 0.97 no espaço métrico ArcFace 128-D
   e os documentos já mapeados que ainda não possuem texto extraído no PostgreSQL.
2. Renderiza em paralelo os PNGs faltantes em data/rendered_pages/.
3. Extração com RapidOCR em pool de 24 workers.
4. Classificação automática de CUPOM FISCAL vs Falsos Positivos (NFC-e, Cartão TEF, DANFE, NFS-e).
5. Persistência completa no PostgreSQL (documento, documento_ocr_pagina, documento_texto_completo).
6. Organização automática em pastas na Área de Trabalho com thumbnails:
   - Desktop/validacao_cupom_fiscal/
"""

import os
import sys
import re
import json
import gzip
import time
import signal
import shutil
import hashlib
import argparse
import urllib.parse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import cv2
import pymupdf
import psycopg
import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
cv2.setNumThreads(1)

from rapidocr_onnxruntime import RapidOCR

BASE_DIR = Path("/home/lucas.ocunha/masters")
DATA_DIR = BASE_DIR / "data"
PDF_DIR = Path("/media/ssd2/datasets/CEAP")
RENDERED_DIR = DATA_DIR / "rendered_pages"
DESKTOP_DIR = Path("/home/lucas.ocunha/Desktop")
EMBS_PATH = DATA_DIR / "latent_vectors_metric_learned_128d.npy"
META_PATH = DATA_DIR / "latent_vectors_full_metadata.json"
HIST_PATH = BASE_DIR / "ml_backend" / "models" / "annotations_history.json"
YEAR_FILE = DATA_DIR / "pdf_to_year.json"
LOOKUP_FILE = DATA_DIR / "pdf_metadata_lookup.json"
DB_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/ceap")
THUMB_DIR = Path(os.path.expanduser("~/.cache/thumbnails/large"))
THUMB_DIR.mkdir(parents=True, exist_ok=True)

stop_requested = False


def sig_handler(signum, frame):
    global stop_requested
    print("\n[INFO] Sinal de interrupção recebido. Finalizando tarefas ativas com segurança...")
    stop_requested = True


signal.signal(signal.SIGINT, sig_handler)
signal.signal(signal.SIGTERM, sig_handler)

# Expressões Regulares de Detecção e Classificação para Cupons Térmicos
RE_CUPOM_FISCAL = re.compile(
    r'\bCUPOM\s*FISCAL\b|\bCOO\b|\bCCF\b|\bECF-IF\b|\bGNF\b|\bGRG\b|\bCER\b|'
    r'IMPRESSORA\s*FISCAL|MEM[ÓO]RIA\s*FISCAL|\bCF-E-SAT\b|\bCFE-SAT\b|EXTRATO\s*SAT|'
    r'CUPOM\s*FISCAL\s*ELETR[ÔO]NICO|BEMATECH|DARUMA|ELGIN|SWEDA|EPSON|MECANISMO\s*FISCAL',
    re.IGNORECASE
)

RE_NFCE = re.compile(r'DANFE\s*NFC-?E|NOTA\s*FISCAL\s*DE\s*CONSUMIDOR\s*ELETR|NFC-?E\b', re.IGNORECASE)
RE_CARTAO = re.compile(r'COMPROVANTE\s*(DE\s*)?(D[EO]\s*)?(PAGAMENTO|CR[ÉE]DITO|D[ÉE]BITO|VENDA)|CART[ÃA]O\s*(DE\s*)?(CR[ÉE]DITO|D[ÉE]BITO)|REDE|CIELO|GETNET|STONE|PAGSEGURO', re.IGNORECASE)
RE_DANFE = re.compile(r'DANFE|DOCUMENTO\s*AUXILIAR\s*DA\s*NOTA\s*FISCAL|CHAVE\s*DE\s*ACESSO', re.IGNORECASE)
RE_NFSE = re.compile(r'NFS-E|NOTA\s*FISCAL\s*DE\s*SERVI[CÇ]OS|DANFSE', re.IGNORECASE)
RE_BOLETO = re.compile(r'FICHA\s*DE\s*COMPENSA|RECIBO\s*DO\s*SACADO|RECIBO\s*DO\s*PAGADOR', re.IGNORECASE)

engine = None


def init_ocr_worker():
    global engine
    engine = RapidOCR(use_cls=False, det_limit_side_len=640, intra_op_num_threads=1, inter_op_num_threads=1)


def render_worker_task(task):
    doc_name, pdf_path_str, png_path_str = task
    try:
        png_p = Path(png_path_str)
        if png_p.exists():
            return True
        pdf_p = Path(pdf_path_str)
        if not pdf_p.exists():
            return False
        doc = pymupdf.open(str(pdf_p))
        if len(doc) == 0:
            doc.close()
            return False
        pix = doc[0].get_pixmap(dpi=110, alpha=False)
        pix.save(str(png_p))
        doc.close()
        return True
    except Exception:
        return False


def is_valid_digital_text(txt: str) -> bool:
    txt_clean = txt.strip()
    if len(txt_clean) < 30:
        return False
    alnum_count = sum(1 for c in txt_clean if c.isalnum() or c.isspace())
    if (alnum_count / len(txt_clean)) < 0.65:
        return False
    alpha_count = sum(1 for c in txt_clean if c.isalpha())
    if alpha_count < 15:
        return False
    return True


def classify_text(txt: str) -> str:
    txt_upper = (txt or "").upper()

    if RE_CUPOM_FISCAL.search(txt_upper):
        return "cupom_fiscal_confirmado"
    if RE_NFCE.search(txt_upper):
        return "nfce_bobina"
    if RE_CARTAO.search(txt_upper):
        return "comprovante_cartao"
    if RE_DANFE.search(txt_upper):
        return "danfe_nfe"
    if RE_NFSE.search(txt_upper):
        return "nfse"
    if RE_BOLETO.search(txt_upper):
        return "boleto"
    return "outros_sem_marcador"


def process_ocr_task(task):
    global engine
    doc_id, png_path_str = task
    t0 = time.time()
    try:
        img = cv2.imread(png_path_str)
        if img is None:
            return doc_id, "", 0, "erro_imagem", 0

        res, _ = engine(img)
        t_elapsed_ms = int((time.time() - t0) * 1000)

        if not res:
            return doc_id, "", 0, "outros_sem_marcador", t_elapsed_ms

        full_txt = " ".join([line[1] for line in res])
        char_count = len(full_txt)
        cls = classify_text(full_txt)

        return doc_id, full_txt, char_count, cls, t_elapsed_ms
    except Exception as e:
        t_elapsed_ms = int((time.time() - t0) * 1000)
        return doc_id, "", 0, f"erro_ocr: {e}", t_elapsed_ms


def setup_thumbnail(image_path: Path):
    try:
        uri = "file://" + urllib.parse.quote(str(image_path.resolve()))
        h = hashlib.md5(uri.encode("utf-8")).hexdigest()
        thumb_p = THUMB_DIR / f"{h}.png"
        if thumb_p.exists():
            return True
        doc = pymupdf.open(str(image_path))
        if len(doc) == 0:
            doc.close()
            return False
        pix = doc[0].get_pixmap(dpi=72)
        pix.save(str(thumb_p))
        doc.close()
        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="RapidOCR nos documentos CUPOM FISCAL do CEAP.")
    parser.add_argument("--min-sim", type=float, default=0.97, help="Similaridade mínima no espaço ArcFace (default: 0.97)")
    parser.add_argument("--max-candidates", type=int, default=None, help="Limite máximo de candidatos para processar")
    parser.add_argument("--workers", type=int, default=24, help="Número de workers paralelos para OCR (default: 24)")
    args = parser.parse_args()

    print("=" * 76)
    print(" EXTRAÇÃO RAPIDOCR & VALIDAÇÃO AUTOMÁTICA DE CUPOM FISCAL")
    print(f" Limiar de Similaridade Mínima: {args.min_sim * 100:.1f}% | Workers: {args.workers}")
    print("=" * 76)
    t_start = time.time()

    # 1. Carregar Histórico e Centróide ArcFace
    print("\n1. Carregando anotações e calculando centróide CUPOM FISCAL 128-D...")
    with open(HIST_PATH, "r", encoding="utf-8") as f:
        history = json.load(f)
    norm_hist = {k.replace(".pdf", ""): v for k, v in history.items()}

    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    docs = meta["documents"]
    stems = [d["doc_name"].replace(".pdf", "") for d in docs]
    stem_to_idx = {s: i for i, s in enumerate(stems)}

    embs = np.load(EMBS_PATH)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs /= norms

    cf_idxs = [stem_to_idx[s] for s, lbl in norm_hist.items() if lbl == "cupom_fiscal" and s in stem_to_idx]
    print(f"   ✓ Documentos CUPOM FISCAL anotados no espaço métrico: {len(cf_idxs):,}")

    cf_centroid = embs[cf_idxs].mean(axis=0)
    cf_centroid /= np.linalg.norm(cf_centroid)

    sims = np.dot(embs, cf_centroid)

    with open(LOOKUP_FILE, "r", encoding="utf-8") as f:
        lookup = json.load(f)

    # 2. Filtrar Candidatos
    print(f"\n2. Filtrando candidatos com similaridade >= {args.min_sim:.2f}...")
    candidates = []
    for i, s in enumerate(stems):
        sim = float(sims[i])
        if sim >= args.min_sim:
            is_already = (norm_hist.get(s) == "cupom_fiscal")
            fname = f"{s}.pdf"
            forn = lookup.get(fname, {}).get("fornecedor", "")
            
            candidates.append({
                "doc_id": s,
                "doc_name": fname,
                "similarity": round(sim, 4),
                "fornecedor": forn,
                "already_annotated": is_already
            })

    candidates.sort(key=lambda x: -x["similarity"])
    if args.max_candidates:
        candidates = candidates[:args.max_candidates]

    unann_count = sum(1 for c in candidates if not c["already_annotated"])
    already_count = sum(1 for c in candidates if c["already_annotated"])

    print(f"   ✓ Total selecionado para o pipeline: {len(candidates):,}")
    print(f"     • Não anotados (novos candidatos) : {unann_count:,}")
    print(f"     • Já mapeados como CUPOM FISCAL   : {already_count:,}")

    # 3. Renderização Paralela de PNGs
    print(f"\n3. Verificando renderização em data/rendered_pages/...")
    RENDERED_DIR.mkdir(parents=True, exist_ok=True)
    missing_render_tasks = []
    for c in candidates:
        png_p = RENDERED_DIR / f"{c['doc_id']}.png"
        if not png_p.exists():
            pdf_p = PDF_DIR / c["doc_name"]
            missing_render_tasks.append((c["doc_name"], str(pdf_p), str(png_p)))

    if missing_render_tasks:
        print(f"   Renderizando {len(missing_render_tasks):,} imagens faltantes...")
        t_rend = time.time()
        with ProcessPoolExecutor(max_workers=min(24, os.cpu_count() or 16)) as ex:
            list(ex.map(render_worker_task, missing_render_tasks))
        print(f"   ✓ Renderização concluída em {time.time() - t_rend:.1f}s ({len(missing_render_tasks)/(time.time()-t_rend):.1f} docs/s)")
    else:
        print("   ✓ Todos os PNGs já se encontram renderizados no disco.")

    # 4. Extração Híbrida: Digital Fast-Track vs RapidOCR
    print("\n4. Executando Extração Híbrida (Fast-Track Digital + RapidOCR 24 workers)...")
    results = []
    to_ocr_tasks = []
    digital_hits = 0

    t_fase1 = time.time()
    for c in candidates:
        doc_id = c["doc_id"]
        pdf_p = PDF_DIR / c["doc_name"]

        # Tentar extração nativa digital
        has_digital = False
        if pdf_p.exists():
            try:
                doc = pymupdf.open(str(pdf_p))
                txt = "".join(p.get_text() for p in doc).strip()
                doc.close()
                if is_valid_digital_text(txt):
                    cls = classify_text(txt)
                    results.append({
                        "doc_id": doc_id,
                        "doc_name": c["doc_name"],
                        "similarity": c["similarity"],
                        "fornecedor": c["fornecedor"],
                        "metodo": "digital_pdf",
                        "texto": txt,
                        "num_caracteres": len(txt),
                        "classe": cls,
                        "tempo_ms": 1
                    })
                    has_digital = True
                    digital_hits += 1
            except Exception:
                pass

        if not has_digital:
            png_p = RENDERED_DIR / f"{doc_id}.png"
            to_ocr_tasks.append((doc_id, str(png_p)))

    print(f"   Fase 1 (Fast-Track Digital) concluída em {time.time() - t_fase1:.1f}s:")
    print(f"   • Textos digitais processados diretamente: {digital_hits:,} ({digital_hits/len(candidates)*100:.1f}%)")
    print(f"   • Documentos para RapidOCR (escaneados): {len(to_ocr_tasks):,} ({len(to_ocr_tasks)/len(candidates)*100:.1f}%)")

    # Fase 2: RapidOCR
    if to_ocr_tasks:
        cands_by_id = {c["doc_id"]: c for c in candidates}
        num_workers = min(args.workers, os.cpu_count() or 16)
        print(f"\n   Executando RapidOCR em paralelo com {num_workers} workers...")
        ocr_start = time.time()
        processed_ocr = 0

        with ProcessPoolExecutor(max_workers=num_workers, initializer=init_ocr_worker) as ex:
            for doc_id, txt, char_count, cls, t_ms in ex.map(process_ocr_task, to_ocr_tasks, chunksize=25):
                c = cands_by_id[doc_id]
                results.append({
                    "doc_id": doc_id,
                    "doc_name": c["doc_name"],
                    "similarity": c["similarity"],
                    "fornecedor": c["fornecedor"],
                    "metodo": "rapidocr",
                    "texto": txt,
                    "num_caracteres": char_count,
                    "classe": cls,
                    "tempo_ms": t_ms
                })
                processed_ocr += 1

                if processed_ocr % 1000 == 0 or processed_ocr == len(to_ocr_tasks):
                    el = time.time() - ocr_start
                    rate = processed_ocr / max(el, 0.1)
                    rem = (len(to_ocr_tasks) - processed_ocr) / max(rate, 0.1)
                    print(f"   [RapidOCR] {processed_ocr:,}/{len(to_ocr_tasks):,} ({rate:.1f} docs/s | Restante: {rem/60:.1f} min)", flush=True)

        print(f"   ✓ RapidOCR finalizado em {time.time() - ocr_start:.1f}s!")

    # 5. Estatísticas de Classificação
    print("\n5. Estatísticas de Classificação:")
    cls_counter = Counter(r["classe"] for r in results)
    for cls, count in cls_counter.most_common():
        print(f"   • {cls:28s}: {count:6,d} ({count/len(results)*100:.1f}%)")

    # 6. Sincronização no PostgreSQL
    print(f"\n6. Sincronizando {len(results):,} registros no PostgreSQL...")
    with open(YEAR_FILE, "r", encoding="utf-8") as f:
        pdf_to_year = json.load(f)

    try:
        with psycopg.connect(DB_URL) as conn:
            with conn.cursor() as cur:
                doc_rows = []
                pg_rows = []
                txt_rows = []

                for r in results:
                    did = int(r["doc_id"])
                    ano = pdf_to_year.get(r["doc_name"], 2024)
                    caminho = f"/media/ssd2/datasets/CEAP/{r['doc_name']}"
                    sanitized_txt = (r["texto"] or "").replace("\x00", "")

                    doc_rows.append((did, ano, caminho, "baixado", r["classe"]))
                    pg_rows.append((did, 1, r["metodo"], sanitized_txt, r["num_caracteres"], r["tempo_ms"]))
                    txt_rows.append((did, r["metodo"], 1, sanitized_txt, r["num_caracteres"], r["tempo_ms"]))

                batch_size = 1000
                for i in range(0, len(doc_rows), batch_size):
                    cur.executemany("""
                        INSERT INTO documento (ide_documento, ano, caminho_arquivo, status, classe_predita)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (ide_documento) DO UPDATE
                        SET classe_predita = EXCLUDED.classe_predita,
                            status = EXCLUDED.status;
                    """, doc_rows[i:i + batch_size])

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
                    """, pg_rows[i:i + batch_size])

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
                    """, txt_rows[i:i + batch_size])

                conn.commit()
        print("   ✓ Banco PostgreSQL sincronizado com sucesso!")
    except Exception as e:
        print(f"   [AVISO] Falha ao sincronizar PostgreSQL: {e}")

    # 7. Organizar Pastas de Validação no Desktop
    print("\n7. Organizando pastas de validação no Desktop...")
    batch_dir = DATA_DIR / "validacao_batches" / "validacao_cupom_fiscal"
    desktop_link = DESKTOP_DIR / "validacao_cupom_fiscal"

    p_confirmados = batch_dir / "1_confirmados_ocr"
    p_nfce_cartao = batch_dir / "2_nfce_e_cartao"
    p_falsos = batch_dir / "3_falsos_positivos_detectados"

    for p in [p_confirmados, p_nfce_cartao, p_falsos]:
        p.mkdir(parents=True, exist_ok=True)

    for r in results:
        doc_id = r["doc_id"]
        src_png = RENDERED_DIR / f"{doc_id}.png"
        if not src_png.exists():
            continue

        cls = r["classe"]
        if cls == "cupom_fiscal_confirmado":
            dst = p_confirmados / f"{doc_id}.png"
        elif cls in ("nfce_bobina", "comprovante_cartao"):
            dst = p_nfce_cartao / f"{doc_id}.png"
        elif cls in ("danfe_nfe", "nfse", "boleto"):
            dst = p_falsos / f"{doc_id}.png"
        else:
            dst = p_nfce_cartao / f"{doc_id}.png"

        if not dst.exists():
            try:
                os.link(src_png, dst)
            except OSError:
                shutil.copy2(src_png, dst)
        setup_thumbnail(src_png)

    # Salvar resultados completos em json.gz
    manifest_file = batch_dir / "rapidocr_results.json.gz"
    with gzip.open(manifest_file, "wt", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)

    if desktop_link.is_symlink() or desktop_link.exists():
        desktop_link.unlink()
    desktop_link.symlink_to(batch_dir)

    print(f"   ✓ Pasta criada: {desktop_link}")
    print(f"     • 1_confirmados_ocr           : {len(os.listdir(p_confirmados)):,} cupons confirmados")
    print(f"     • 2_nfce_e_cartao             : {len(os.listdir(p_nfce_cartao)):,} cupons NFC-e / TEF a revisar")
    print(f"     • 3_falsos_positivos_detectados: {len(os.listdir(p_falsos)):,} detectados")

    print("\n" + "=" * 76)
    print(f" PIPELINE FINALIZADO EM {time.time() - t_start:.1f}s!")
    print("=" * 76)


if __name__ == "__main__":
    main()
