#!/usr/bin/env python3
"""
Extrai Lote de 1.000 Boletos Bancários Legítimos (FEBRABAN / Bancos / Condomínios / Imobiliárias)
- Exclui estritamente Sem Parar, CGMP e concessionárias de pedágio
- Utiliza sementes de boletos bancários reais (Itaú, BB, Bradesco, Caixa, Condomínios)
- Busca no espaço visual 384-D do DINOv2 combinado com filtros temáticos
- Renderiza em PNG (110 DPI)
- Executa RapidOCR com 12 workers paralelos para confirmação inequívoca de Ficha de Compensação / Boleto
- Salva no Desktop do usuário:
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
import shutil
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

RE_BOLETO_REAL = re.compile(
    r'(FICHA\s*DE\s*(COMPENSA[CÇ][AÃ]O|CAIXA)|RECIBO\s*D[OE]\s*(SACADO|PAGADOR)|'
    r'LOCAL\s*DE\s*PAGAMENTO|PAG[AÁ]VEL\s*EM\s*QUALQUER\s*BANCO|'
    r'AG[EÊ]NCIA\s*/\s*C[OÓ]DIGO|NOSSO\s*N[UÚ]MERO|AUTENTICA[CÇ][AÃ]O\s*MEC[AÂ]NICA|'
    r'COMPROVANTE\s*DE\s*PAGAMENTO\s*DE\s*T[IÍ]TULO|LINHA\s*DIGIT[AÁ]VEL|'
    r'BANCO\s*ITAU|BANCO\s*DO\s*BRASIL|BRADESCO|SANTANDER|CAIXA\s*ECON[OÔ]MICA|'
    r'SICOOB|SICREDI|CEDENTE|SACADO|VALOR\s*DO\s*DOCUMENTO)',
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

        # Boletos têm dados chave em 60% da página (topo ou rodapé da ficha de compensação)
        h = img.shape[0]
        crop = img[:int(h * 0.65), :]
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
            m = RE_BOLETO_REAL.search(txt)
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
    batch_output = root_dir / "data" / "validacao_batches" / "validacao_boleto_1000"

    print(f"=== EXTRAÇÃO DE {target_count:,} BOLETOS BANCÁRIOS LEGÍTIMOS (BANCOS / CONDOMÍNIOS) ===")
    t0 = time.time()

    npz_path = root_dir / "data" / "latent_vectors_full.npz"
    meta_path = root_dir / "data" / "latent_vectors_full_metadata.json"
    history_path = root_dir / "ml_backend" / "models" / "annotations_history.json"
    lookup_path = root_dir / "data" / "pdf_metadata_lookup.json"

    print("1. Carregando metadados e DINOv2 visual...")
    npz = np.load(npz_path, mmap_mode="r")
    dino = npz["dino_doc_embeddings"]

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    docs = meta["documents"]
    n_docs = len(docs)
    doc_to_idx = {d["doc_name"]: i for i, d in enumerate(docs)}

    # Sementes de boletos reais: carregar os validados pelo usuário em CONFIRMADOS
    confirmados_dir = batch_output / "CONFIRMADOS"
    confirmed_seeds = []
    if confirmados_dir.exists():
        confirmed_seeds = [f"{f.stem}.pdf" for f in sorted(confirmados_dir.glob("*.png"))]

    fallback_seeds = ["5456002.pdf", "5493044.pdf", "5514782.pdf", "5582976.pdf", "5723091.pdf"]
    real_seeds = list(dict.fromkeys(confirmed_seeds + fallback_seeds))
    seed_indices = [doc_to_idx[s] for s in real_seeds if s in doc_to_idx]
    print(f"   Sementes reais selecionadas: {len(seed_indices)} boletos bancários confirmados ({len(confirmed_seeds)} em CONFIRMADOS)")

    seed_vecs = dino[seed_indices].astype(np.float32)
    norms_s = np.linalg.norm(seed_vecs, axis=1, keepdims=True)
    norms_s[norms_s == 0] = 1.0
    seed_vecs /= norms_s
    centroid = seed_vecs.mean(axis=0)
    centroid /= np.linalg.norm(centroid)

    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)
    annotated_stems = set(history.keys())

    with open(lookup_path, "r", encoding="utf-8") as f:
        lookup = json.load(f)

    print("2. Filtrando candidatos e calculando similaridade DINOv2...")
    # Pré-selecionar candidatos excluindo estritamente Sem Parar e pedágios
    scored_candidates = []
    chunk_size = 100000
    sims = np.zeros(n_docs, dtype=np.float32)
    for start in range(0, n_docs, chunk_size):
        end = min(start + chunk_size, n_docs)
        sub = dino[start:end].astype(np.float32)
        norms = np.linalg.norm(sub, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        sub /= norms
        sims[start:end] = np.dot(sub, centroid)

    for i in range(n_docs):
        doc_name = docs[i]["doc_name"]
        stem = Path(doc_name).stem
        if stem in annotated_stems:
            continue

        info = lookup.get(doc_name, {})
        forn = info.get("fornecedor", "").upper()
        sub = info.get("subcota", "").upper()

        # Bloquear 100% de Sem Parar, pedágio e tags rodoviárias
        if any(k in forn for k in ["SEM PARAR", "CGMP", "CONECTCAR", "VELOE", "MOVE MAIS", "AUTOPISTA", "RODOVIA", "CCR"]):
            continue
        if "PEDÁGIO" in sub or "PEDAGIO" in sub:
            continue

        is_condo = any(k in forn for k in ["CONDOMINIO", "CONDOMÍNIO", "EDIFICIO", "EDIFÍCIO", "CENTRO EMPRESARIAL", "TOWER", "WALL STREET"])
        is_imob = any(k in forn for k in ["IMOBILIARIA", "IMOBILIÁRIA", "ADMINISTRADORA", "ADMINISTRACAO DE IMOVEIS", "PREDIAL", "LOCAÇÕES"])
        is_banco = any(k in forn for k in ["BANCO BRADESCO", "BANCO DO BRASIL", "BANCO ITAU", "BANCO SANTANDER", "CAIXA ECONOMICA", "BANCO SAFRA", "SICOOB", "SICREDI"])
        
        sim_val = float(sims[i])
        # Score ponderado: similaridade visual DINOv2 + peso temático
        if is_condo or is_imob or is_banco:
            score = sim_val + 0.35
            scored_candidates.append((i, stem, doc_name, score, sim_val, forn))
        elif sim_val >= 0.88 and any(k in sub for k in ["ESCRITÓRIO", "ESCRITORIO", "CONSULTORIA", "DIVULGAÇÃO", "DIVULGACAO", "LOCAÇÃO"]):
            score = sim_val + 0.15
            scored_candidates.append((i, stem, doc_name, score, sim_val, forn))

    scored_candidates.sort(key=lambda x: -x[3])
    selected = scored_candidates[:target_count]
    print(f"   Selecionados {len(selected)} candidatos a boleto legítimo.")
    print(f"   Similaridade visual média: {np.mean([s[4] for s in selected]):.4f}")

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

    # Limpar apenas as subpastas geradas, preservando CONFIRMADOS
    folder_confirmed = batch_output / "1_confirmadas_por_ocr"
    folder_candidates = batch_output / "2_alta_similaridade_candidatos"
    if folder_confirmed.exists():
        shutil.rmtree(folder_confirmed)
    if folder_candidates.exists():
        shutil.rmtree(folder_candidates)
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

    manifest_path = batch_output / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 66)
    print("   LOTE DE 1.000 BOLETOS BANCÁRIOS REAIS GERADO COM SUCESSO!     ")
    print("=" * 66)
    print(f"Destino: {batch_output}")
    print(f"  ├── 1_confirmadas_por_ocr:          {confirmed_count} documentos")
    print(f"  └── 2_alta_similaridade_candidatos:  {candidate_count} documentos")
    print(f"Tempo total: {time.time()-t0:.1f} segundos")
    print("=" * 66)


if __name__ == "__main__":
    main()
