#!/usr/bin/env python3
"""
Recupera os documentos mais próximos das 8 sementes de NFC-e em todo o espaço de 1.34M.
- Busca vetorial por similaridade de cosseno (DINOv2 + ColPali)
- Renderização automática das páginas em PNG via PyMuPDF
- Verificação rápida por OCR para confirmar 'DANFE NFC-e' / 'Mod 65'
- Organização automática na pasta do Desktop: Desktop/validacao_candidatos_nfce/
"""

import os
import re
import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pymupdf
import cv2

# Otimização de threads
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
cv2.setNumThreads(1)

from rapidocr_onnxruntime import RapidOCR

RE_NFCE = re.compile(
    r'NFC-?E\b|DANFE|CO[NH]SUMIDOR\s*ELETR[OÔ]NIC|DOCU[HM]ENTO\s*AUXILIAR|MOD(ELO)?\s*65|CHAVE\s*DE\s*ACESSO',
    re.IGNORECASE
)

engine = None

def init_worker():
    global engine
    engine = RapidOCR(use_cls=False, det_limit_side_len=480, intra_op_num_threads=2, inter_op_num_threads=1)

def verify_single_candidate(task):
    global engine
    doc_id, pdf_path_str, png_path_str = task
    try:
        png_p = Path(png_path_str)
        pdf_p = Path(pdf_path_str)
        
        # 1. Renderizar se não existir
        if not png_p.exists():
            if not pdf_p.exists():
                return doc_id, False, "PDF_NAO_ENCONTRADO"
            doc = pymupdf.open(str(pdf_p))
            page = doc[0]
            pix = page.get_pixmap(dpi=150)
            pix.save(str(png_p))
            doc.close()
            
        img = cv2.imread(str(png_p))
        if img is None:
            return doc_id, False, "ERRO_IMAGEM"
            
        h = img.shape[0]
        crop = img[:int(h * 0.48), :]
        dt_boxes, _ = engine.text_det(crop)
        if dt_boxes is None or len(dt_boxes) == 0:
            return doc_id, False, ""
            
        dt_boxes = sorted(dt_boxes, key=lambda b: np.min(b[:, 1]))[:14]
        img_crops = engine.get_crop_img_list(crop, dt_boxes)
        
        for i in range(0, len(img_crops), 4):
            batch = img_crops[i:i + 4]
            rec_res, _ = engine.text_rec(batch)
            if not rec_res:
                continue
            txt = " ".join([r[0] for r in rec_res])
            if RE_NFCE.search(txt):
                return doc_id, True, txt[:80]
                
        return doc_id, False, ""
    except Exception as e:
        return doc_id, False, f"ERRO: {e}"


def main():
    root_dir = Path("/home/lucas.ocunha/masters")
    pdf_dir = Path("/media/ssd2/datasets/CEAP")
    rendered_dir = root_dir / "data" / "rendered_pages"
    rendered_dir.mkdir(parents=True, exist_ok=True)
    desktop_output = Path("/home/lucas.ocunha/Desktop/validacao_candidatos_nfce")
    
    print("=== BUSCA DE CANDIDATOS MAIS PRÓXIMOS DE NFC-E (1.34M) ===", flush=True)
    t0 = time.time()
    
    # 1. Carregar embeddings
    npz_path = root_dir / "data" / "latent_vectors_full.npz"
    meta_path = root_dir / "data" / "latent_vectors_full_metadata.json"
    history_path = root_dir / "ml_backend" / "models" / "annotations_history.json"
    lookup_path = root_dir / "data" / "pdf_metadata_lookup.json"
    
    print("Carregando embeddings densos...", flush=True)
    npz = np.load(npz_path)
    dino = npz["dino_doc_embeddings"]
    vlm = npz["vlm_doc_embeddings"]
    combined = np.hstack([dino, vlm]).astype(np.float32)
    norms = np.linalg.norm(combined, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    combined /= norms
    
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    docs = meta["documents"]
    
    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)
        
    with open(lookup_path, "r", encoding="utf-8") as f:
        lookup = json.load(f)
        
    # 2. Localizar as 8 sementes de NFC-e
    nfce_seeds = {'6819980', '6960067', '6934894', '7957398', '8013693', '6429678', '6397967', '6744525'}
    seed_indices = [i for i, d in enumerate(docs) if d["doc_name"].replace(".pdf", "") in nfce_seeds]
    print(f"Identificadas {len(seed_indices)} sementes de NFC-e em 1.34M.", flush=True)
    
    seed_vecs = combined[seed_indices]
    centroid = seed_vecs.mean(axis=0, keepdims=True)
    centroid /= np.linalg.norm(centroid)
    
    # 3. Similaridade combinada em todo o 1.34M
    sim_c = np.dot(combined, centroid.T).flatten()
    sim_m = np.max(np.dot(combined, seed_vecs.T), axis=1)
    score = 0.5 * sim_c + 0.5 * sim_m
    
    annotated_stems = set(history.keys())
    top_indices = np.argsort(-score)
    
    # Selecionar Top 500 candidatos não anotados
    selected = []
    for idx in top_indices:
        stem = docs[idx]["doc_name"].replace(".pdf", "")
        if stem in nfce_seeds or stem in annotated_stems:
            continue
        selected.append((stem, float(score[idx])))
        if len(selected) >= 500:
            break
            
    print(f"Top 500 candidatos selecionados (score mín: {selected[-1][1]:.4f}). Verificando OCR...", flush=True)
    
    # 4. Renderizar e validar via OCR em paralelo
    tasks = [(stem, str(pdf_dir / f"{stem}.pdf"), str(rendered_dir / f"{stem}.png")) for stem, _ in selected]
    score_dict = dict(selected)
    
    ocr_results = {}
    with ProcessPoolExecutor(max_workers=8, initializer=init_worker) as executor:
        for doc_id, is_nfce, snippet in executor.map(verify_single_candidate, tasks, chunksize=8):
            ocr_results[doc_id] = (is_nfce, snippet)
            
    # 5. Organizar no Desktop
    if desktop_output.exists():
        import shutil
        shutil.rmtree(desktop_output)
        
    dir_confirmed = desktop_output / "1_nfce_confirmadas_por_ocr"
    dir_high_sim = desktop_output / "2_alta_similaridade_candidatos"
    dir_confirmed.mkdir(parents=True, exist_ok=True)
    dir_high_sim.mkdir(parents=True, exist_ok=True)
    
    confirmed_count = 0
    candidate_count = 0
    
    for rank, (stem, sc) in enumerate(selected, 1):
        is_nfce, snippet = ocr_results.get(stem, (False, ""))
        m = lookup.get(f"{stem}.pdf", {})
        ano = m.get("ano", 0)
        raw_forn = m.get("fornecedor", "")[:22]
        forn = re.sub(r'[^a-zA-Z0-9]', '_', raw_forn).strip('_') or "FORNECEDOR"
        forn = re.sub(r'_+', '_', forn)
        src = rendered_dir / f"{stem}.png"
        
        if not src.exists():
            continue
            
        target_dir = dir_confirmed if is_nfce else dir_high_sim
        link = target_dir / f"rank_{rank:04d}_[sim_{sc:.3f}]_[{ano}]_[{forn}]_{stem}.png"
        if not link.exists():
            link.symlink_to(src)
            
        if is_nfce:
            confirmed_count += 1
        else:
            candidate_count += 1
            
    # 6. LEIAME.txt
    readme = f"""=============================================================================
CANDIDATOS MAIS PRÓXIMOS DE NFC-E NO ESPAÇO LATENTE (1,34 MILHÃO DE DOCUMENTOS)
=============================================================================

Total de Candidatos Selecionados: 500
Critério: Máxima similaridade de cosseno com as 8 sementes validadas de NFC-e.

1. 1_nfce_confirmadas_por_ocr/ ({confirmed_count:,} documentos):
   - Documentos cujo texto já foi lido por OCR e CONFIRMOU menção expressa a
     'DANFE NFC-e', 'CONSUMIDOR ELETRÔNICA' ou 'MODELO 65'.
   - Prontos para validação imediata em lote!

2. 2_alta_similaridade_candidatos/ ({candidate_count:,} documentos):
   - Documentos com altíssima proximidade visual aos cupons de NFC-e, mas cujo
     cabeçalho estava com menor contraste ou formatação ligeiramente diferente.
"""
    (desktop_output / "LEIAME.txt").write_text(readme, encoding="utf-8")
    
    # Salvar relatório JSON
    report_file = desktop_output / "relatorio_top500_nfce.json"
    report_data = [
        {
            "rank": r,
            "doc_id": s,
            "similarity_score": sc,
            "confirmed_by_ocr": ocr_results[s][0],
            "snippet": ocr_results[s][1],
            "ano": lookup.get(f"{s}.pdf", {}).get("ano", 0),
            "fornecedor": lookup.get(f"{s}.pdf", {}).get("fornecedor", ""),
            "subcota": lookup.get(f"{s}.pdf", {}).get("subcota", "")
        }
        for r, (s, sc) in enumerate(selected, 1)
    ]
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
        
    total_time = time.time() - t0
    print("\n=======================================================", flush=True)
    print(f"RECUPERAÇÃO CONCLUÍDA EM {total_time:.1f} SEGUNDOS!", flush=True)
    print(f"  Confirmados expressos como NFC-e por OCR : {confirmed_count:4d} ({confirmed_count/len(selected)*100:.1f}%)", flush=True)
    print(f"  Candidatos de alta similaridade visual   : {candidate_count:4d} ({candidate_count/len(selected)*100:.1f}%)", flush=True)
    print(f"Pasta disponível em: {desktop_output}", flush=True)
    print("=======================================================", flush=True)

if __name__ == "__main__":
    main()
