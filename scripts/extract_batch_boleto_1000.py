#!/usr/bin/env python3
"""
Extrai Lote de 1.000 Boletos Bancários via Cluster DINOv2 (Visual)
- Utiliza os dois boletos confirmados (7206758.pdf e 7224565.pdf) como sementes
- Busca estritamente no espaço visual 384-D do DINOv2
- Renderiza página 1 em PNG (110 DPI)
- Executa OCR rápido para confirmação inequívoca de termos de boleto (Cedente, Sacado, Vencimento, Banco, etc.)
- Organiza no Desktop do usuário:
    /home/lucas.ocunha/Desktop/validacao_boleto_1000/
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

RE_BOLETO_KEYWORDS = re.compile(
    r'BOLETO|RECIBO\s*D[EO]\s*SACADO|CEDENTE|SACADO|BENEFICI[AÁ]RIO|'
    r'PAGADOR|VENCIMENTO|NOSSO\s*N[UÚ]MERO|AG[EÊ]NCIA\s*/\s*C[OÓ]DIGO|'
    r'VALOR\s*COBRADO|VALOR\s*DO\s*DOCUMENTO|AUTENTICA[CÇ][AÃ]O\s*MEC[AÂ]NICA|'
    r'FICHA\s*DE\s*COMPENSA[CÇ][AÃ]O|BANCO\s*DO\s*BRASIL|BRADESCO|ITA[UÚ]|'
    r'SANTANDER|CAIXA\s*ECON[OÔ]MICA|LINHA\s*DIGIT[AÁ]VEL|GERA[CÇ][AÃ]O\s*DE\s*BOLETO',
    re.IGNORECASE
)

engine = None


def init_worker():
    global engine
    engine = RapidOCR(use_cls=False, det_limit_side_len=380, intra_op_num_threads=1, inter_op_num_threads=1)


def process_boleto_doc(task):
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

        # Boletos geralmente têm cabeçalho/recibo do sacado na metade superior ou rodapé com código de barras
        # Analisamos a metade superior (0% a 55%)
        h = img.shape[0]
        crop = img[:int(h * 0.55), :]
        dt_boxes, _ = engine.text_det(crop)
        if dt_boxes is None or len(dt_boxes) == 0:
            return doc_id, False, "", forn_name

        dt_boxes = sorted(dt_boxes, key=lambda b: np.min(b[:, 1]))[:12]
        img_crops = engine.get_crop_img_list(crop, dt_boxes)

        for i in range(0, len(img_crops), 6):
            batch = img_crops[i:i + 6]
            rec_res, _ = engine.text_rec(batch)
            if not rec_res:
                continue
            txt = " ".join([r[0] for r in rec_res])
            m = RE_BOLETO_KEYWORDS.search(txt)
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
    desktop_output = Path("/home/lucas.ocunha/Desktop/validacao_boleto_1000")

    print(f"=== EXTRAÇÃO DE {target_count:,} CANDIDATOS DE BOLETO BANCÁRIO VIA DINOv2 ===")
    t0 = time.time()

    npz_path = root_dir / "data" / "latent_vectors_full.npz"
    meta_path = root_dir / "data" / "latent_vectors_full_metadata.json"
    history_path = root_dir / "ml_backend" / "models" / "annotations_history.json"
    lookup_path = root_dir / "data" / "pdf_metadata_lookup.json"

    print("1. Carregando embeddings visuais DINOv2 (384-D) e histórico...")
    npz = np.load(npz_path, mmap_mode="r")
    dino = npz["dino_doc_embeddings"]

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    docs = meta["documents"]
    n_docs = len(docs)
    doc_to_idx = {d["doc_name"]: i for i, d in enumerate(docs)}

    seed1 = "7206758.pdf"
    seed2 = "7224565.pdf"

    idx1 = doc_to_idx[seed1]
    idx2 = doc_to_idx[seed2]

    v1 = dino[idx1].astype(np.float32)
    v2 = dino[idx2].astype(np.float32)
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)

    centroid = (v1 + v2) / 2.0
    centroid /= np.linalg.norm(centroid)

    print(f"   Sementes: {seed1} (idx {idx1}) e {seed2} (idx {idx2})")
    print(f"   Similaridade visual direta entre as sementes: {float(np.dot(v1, v2)):.4f}")

    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)
    annotated_stems = set(history.keys())

    with open(lookup_path, "r", encoding="utf-8") as f:
        lookup = json.load(f)

    print("2. Calculando similaridade DINOv2 para todos os 1.34M documentos...")
    chunk_size = 100000
    sims = np.zeros(n_docs, dtype=np.float32)
    for start in range(0, n_docs, chunk_size):
        end = min(start + chunk_size, n_docs)
        sub = dino[start:end].astype(np.float32)
        norms = np.linalg.norm(sub, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        sub /= norms
        sims[start:end] = np.dot(sub, centroid)

    # Rankear excluindo documentos já anotados
    sorted_indices = np.argsort(-sims)
    selected = []
    for idx in sorted_indices:
        doc_name = docs[idx]["doc_name"]
        stem = Path(doc_name).stem
        if stem in annotated_stems:
            continue
        sim_val = float(sims[idx])
        info = lookup.get(doc_name, {})
        forn = info.get("fornecedor", "")
        selected.append((idx, stem, doc_name, sim_val, sim_val, forn))
        if len(selected) >= target_count:
            break

    print(f"   Selecionados {len(selected)} candidatos mais próximos no DINOv2.")
    print(f"   Similaridade DINO min: {selected[-1][3]:.4f} | max: {selected[0][3]:.4f}")

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
        for res in executor.map(process_boleto_doc, tasks, chunksize=16):
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

        if is_ocr:
            dst_png = folder_confirmed / f"{stem}.png"
            confirmed_count += 1
            cat_name = "confirmada_por_ocr"
        else:
            dst_png = folder_candidates / f"{stem}.png"
            candidate_count += 1
            cat_name = "alta_similaridade_candidato"

        if src_png.exists() and not dst_png.exists():
            try:
                os.link(src_png, dst_png)
            except OSError:
                import shutil
                shutil.copy2(src_png, dst_png)

        manifest.append({
            "doc_id": stem,
            "doc_name": doc_name,
            "dino_similarity": round(sim_val, 4),
            "ocr_confirmed": is_ocr,
            "ocr_keyword": ocr_tag,
            "fornecedor": forn,
            "category": cat_name,
        })

    manifest_path = desktop_output / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 66)
    print("      LOTE DE 1.000 BOLETOS (DINOv2) GERADO COM SUCESSO!        ")
    print("=" * 66)
    print(f"Destino: {desktop_output}")
    print(f"  ├── 1_confirmadas_por_ocr:          {confirmed_count} documentos")
    print(f"  └── 2_alta_similaridade_candidatos:  {candidate_count} documentos")
    print(f"Tempo total: {time.time()-t0:.1f} segundos")
    print("=" * 66)


if __name__ == "__main__":
    main()
