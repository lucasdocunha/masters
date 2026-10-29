#!/usr/bin/env python3
"""
Extrai Lote de 1.000 Comprovantes de Serviços Postais (Correios / ECT) para Validação Visual
- Filtra candidatos de alta similaridade na subcota 'SERVIÇOS POSTAIS' e fornecedores dos Correios
- Renderiza em PNG (120 DPI)
- Executa OCR rápido para confirmação inequívoca de palavras-chave dos Correios
- Organiza no Desktop do usuário:
    /home/lucas.ocunha/Desktop/validacao_correios_1000/
        ├── 1_confirmadas_por_ocr/
        ├── 2_alta_similaridade_candidatos/
        └── manifest.json
"""

import os
import sys
import re
import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pymupdf
import cv2

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
cv2.setNumThreads(1)

from rapidocr_onnxruntime import RapidOCR

RE_CORREIOS_KEYWORDS = re.compile(
    r'CORREIOS|EMPRESA\s*BRASILEIRA\s*DE\s*CORREIOS|ECT|SEDEX|PAC|POSTAL|'
    r'FRANQUIA|AVISO\s*DE\s*RECEBIMENTO|CARTA|TELEGRAFOS|CHANCELA|'
    r'VALOR\s*POSTAL|SERVI[CÇ]O\s*POSTAL|AG[EÊ]NCIA\s*DE\s*CORREIOS',
    re.IGNORECASE
)

engine = None


def init_worker():
    global engine
    engine = RapidOCR(use_cls=False, det_limit_side_len=380, intra_op_num_threads=1, inter_op_num_threads=1)


def process_correios_doc(task):
    global engine
    doc_id, pdf_path_str, png_path_str, forn_name = task
    try:
        png_p = Path(png_path_str)
        pdf_p = Path(pdf_path_str)

        if not png_p.exists():
            if not pdf_p.exists():
                return doc_id, False, "PDF_NAO_ENCONTRADO", forn_name
            doc = pymupdf.open(str(pdf_p))
            if len(doc) == 0:
                doc.close()
                return doc_id, False, "PDF_VAZIO", forn_name
            page = doc[0]
            pix = page.get_pixmap(dpi=110, alpha=False)
            png_p.parent.mkdir(parents=True, exist_ok=True)
            pix.save(str(png_p))
            doc.close()

        img = cv2.imread(str(png_p))
        if img is None:
            return doc_id, False, "ERRO_IMAGEM", forn_name

        h = img.shape[0]
        crop = img[:int(h * 0.40), :]
        dt_boxes, _ = engine.text_det(crop)
        if dt_boxes is None or len(dt_boxes) == 0:
            return doc_id, False, "", forn_name

        dt_boxes = sorted(dt_boxes, key=lambda b: np.min(b[:, 1]))[:6]
        img_crops = engine.get_crop_img_list(crop, dt_boxes)

        for i in range(0, len(img_crops), 6):
            batch = img_crops[i:i + 6]
            rec_res, _ = engine.text_rec(batch)
            if not rec_res:
                continue
            txt = " ".join([r[0] for r in rec_res])
            m = RE_CORREIOS_KEYWORDS.search(txt)
            if m:
                return doc_id, True, m.group(0), forn_name

        return doc_id, False, "", forn_name
    except Exception as e:
        return doc_id, False, f"ERRO: {e}", forn_name


def main():
    target_count = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    root_dir = Path("/home/lucas.ocunha/masters")
    pdf_dir = Path("/media/ssd2/datasets/CEAP")
    rendered_dir = root_dir / "data" / "rendered_pages"
    rendered_dir.mkdir(parents=True, exist_ok=True)
    desktop_output = root_dir / "data" / "validacao_batches" / "validacao_correios_1000"

    print(f"=== EXTRAÇÃO DE {target_count:,} CANDIDATOS DE SERVIÇOS POSTAIS (CORREIOS / ECT) ===")
    t0 = time.time()

    embs_path = root_dir / "data" / "latent_vectors_metric_learned_128d.npy"
    meta_path = root_dir / "data" / "latent_vectors_full_metadata.json"
    history_path = root_dir / "ml_backend" / "models" / "annotations_history.json"
    lookup_path = root_dir / "data" / "pdf_metadata_lookup.json"

    print("1. Carregando embeddings 128-D e histórico de anotações...")
    embs = np.load(embs_path)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs /= norms

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    docs = meta["documents"]
    n_docs = len(docs)
    stems = [Path(d["doc_name"]).stem for d in docs]

    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)
    annotated_stems = set(history.keys())

    with open(lookup_path, "r", encoding="utf-8") as f:
        lookup = json.load(f)

    # Identificar âncoras de correios na subcota SERVIÇOS POSTAIS
    correios_indices = []
    for i in range(n_docs):
        info = lookup.get(docs[i]["doc_name"], {})
        sub = info.get("subcota", "").upper()
        forn = info.get("fornecedor", "").upper()
        if "POSTAIS" in sub and ("CORREIOS" in forn or "ECT" in forn or "TELEGRAFOS" in forn):
            correios_indices.append(i)
            if len(correios_indices) >= 200:
                break

    print(f"   Âncoras de correios identificadas: {len(correios_indices)}")
    centroid = embs[correios_indices].mean(axis=0)
    centroid /= np.linalg.norm(centroid)

    print("2. Rankeando candidatos por similaridade e filtros de serviços postais...")
    sims = np.dot(embs, centroid)

    scored_candidates = []
    for i in range(n_docs):
        stem = stems[i]
        if stem in annotated_stems:
            continue
        doc_name = docs[i]["doc_name"]
        info = lookup.get(doc_name, {})
        forn = info.get("fornecedor", "").upper()
        sub = info.get("subcota", "").upper()

        is_correios_supplier = (
            "CORREIOS" in forn or "ECT" in forn or "TELEGRAFOS" in forn or
            "TELÉGRAFOS" in forn or "FRANQUIA POSTAL" in forn or "AGENCIA DE CORREIOS" in forn or
            "AGÊNCIA DE CORREIOS" in forn
        )
        is_correios_subcota = "POSTAIS" in sub

        if is_correios_supplier or is_correios_subcota:
            score = float(sims[i]) + (0.40 if is_correios_supplier else 0.20)
            scored_candidates.append((i, stem, doc_name, score, float(sims[i]), forn))

    scored_candidates.sort(key=lambda x: -x[3])
    selected = scored_candidates[:target_count]
    print(f"   Selecionados {len(selected)} candidatos de correios.")

    print(f"3. Processando OCR e gerando PNGs para {len(selected)} documentos...")
    tasks = []
    for item in selected:
        idx, stem, doc_name, score, sim_val, forn = item
        pdf_path = pdf_dir / doc_name
        png_path = rendered_dir / f"{stem}.png"
        tasks.append((stem, str(pdf_path), str(png_path), forn))

    t_ocr_start = time.time()
    n_workers = min(12, os.cpu_count() or 4)
    results = []
    with ProcessPoolExecutor(max_workers=n_workers, initializer=init_worker) as executor:
        for res in executor.map(process_correios_doc, tasks, chunksize=16):
            results.append(res)
            if len(results) % 100 == 0:
                print(f"   [{len(results)}/{len(selected)}] processados em {time.time()-t_ocr_start:.1f}s...", flush=True)

    folder_confirmed = desktop_output / "1_confirmadas_por_ocr"
    folder_candidates = desktop_output / "2_alta_similaridade_candidatos"
    folder_confirmed.mkdir(parents=True, exist_ok=True)
    folder_candidates.mkdir(parents=True, exist_ok=True)

    confirmed_count = 0
    candidate_count = 0
    manifest = []

    for item, (stem, is_ocr, ocr_tag, forn) in zip(selected, results):
        idx, stem, doc_name, score, sim_val, forn = item
        src_png = rendered_dir / f"{stem}.png"
        if not src_png.exists():
            continue

        if is_ocr:
            dst = folder_confirmed / f"{stem}.png"
            confirmed_count += 1
            cat = "confirmado_ocr"
        else:
            dst = folder_candidates / f"{stem}.png"
            candidate_count += 1
            cat = "alta_similaridade"

        if not dst.exists():
            try:
                os.link(src_png, dst)
            except Exception:
                import shutil
                shutil.copy2(src_png, dst)

        manifest.append({
            "doc_id": stem,
            "filename": doc_name,
            "similarity": round(sim_val, 4),
            "score": round(score, 4),
            "ocr_match": is_ocr,
            "ocr_tag": ocr_tag,
            "fornecedor": forn,
            "category": cat
        })

    with open(desktop_output / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\n==================================================================")
    print("      LOTE DE 1.000 CORREIOS GERADO COM SUCESSO!                 ")
    print("==================================================================")
    print(f"Destino: {desktop_output}")
    print(f"  ├── 1_confirmadas_por_ocr:         {confirmed_count:4d} documentos")
    print(f"  └── 2_alta_similaridade_candidatos: {candidate_count:4d} documentos")
    print(f"Tempo total: {time.time() - t0:.1f} segundos")
    print("==================================================================")


if __name__ == "__main__":
    main()
