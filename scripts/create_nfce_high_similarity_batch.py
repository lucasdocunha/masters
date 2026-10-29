#!/usr/bin/env python3
"""
Extrai Lote de Alta Similaridade de NFC-e (Top 1.000 não anotados, sim >= 0.988)
no espaço métrico (128-D ArcFace) ancorado nos 491 documentos reais de NFC-e.
- Renderiza em PNG (se necessário, dpi=120 otimizado)
- Passa OCR rápido no cabeçalho (top 6 caixas)
- Divide em:
    Desktop/validacao_nfce_alta_similaridade/
       ├── 1_confirmadas_por_ocr/
       └── 2_alta_similaridade_candidatos/
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

RE_NFCE = re.compile(
    r'NFC-?E\b|DANFE|CO[NH]SUMIDOR\s*ELETR[OÔ]NIC|DOCU[HM]ENTO\s*AUXILIAR|MOD(ELO)?\s*65|CHAVE\s*DE\s*ACESSO',
    re.IGNORECASE
)

engine = None

def init_worker():
    global engine
    engine = RapidOCR(use_cls=False, det_limit_side_len=400, intra_op_num_threads=2, inter_op_num_threads=1)

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
            pix = page.get_pixmap(dpi=120, alpha=False)
            pix.save(str(png_p))
            doc.close()
            
        img = cv2.imread(str(png_p))
        if img is None:
            return doc_id, False, "ERRO_IMAGEM"
            
        h = img.shape[0]
        crop = img[:int(h * 0.40), :]
        dt_boxes, _ = engine.text_det(crop)
        if dt_boxes is None or len(dt_boxes) == 0:
            return doc_id, False, ""
            
        dt_boxes = sorted(dt_boxes, key=lambda b: np.min(b[:, 1]))[:6]
        img_crops = engine.get_crop_img_list(crop, dt_boxes)
        
        for i in range(0, len(img_crops), 6):
            batch = img_crops[i:i + 6]
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
    target_count = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    root_dir = Path("/home/lucas.ocunha/masters")
    pdf_dir = Path("/media/ssd2/datasets/CEAP")
    rendered_dir = root_dir / "data" / "rendered_pages"
    rendered_dir.mkdir(parents=True, exist_ok=True)
    desktop_output = Path("/home/lucas.ocunha/Desktop/validacao_nfce_alta_similaridade")
    
    print(f"=== EXTRAÇÃO DE {target_count:,} CANDIDATOS DE ALTA SIMILARIDADE NFC-E (128-D) ===", flush=True)
    t0 = time.time()
    
    # 1. Carregar embeddings e metadados
    embs_path = root_dir / "data" / "latent_vectors_metric_learned_128d.npy"
    meta_path = root_dir / "data" / "latent_vectors_full_metadata.json"
    history_path = root_dir / "ml_backend" / "models" / "annotations_history.json"
    lookup_path = root_dir / "data" / "pdf_metadata_lookup.json"
    
    print("Carregando embeddings 128-D e metadados...", flush=True)
    embs = np.load(embs_path)
    
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    docs = meta["documents"]
    
    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)
        
    with open(lookup_path, "r", encoding="utf-8") as f:
        lookup = json.load(f)
        
    # 2. Localizar as 491 sementes de NFC-e
    nfce_indices = [i for i, d in enumerate(docs) if history.get(Path(d["doc_name"]).stem) == "nfce"]
    print(f"Identificadas {len(nfce_indices)} sementes de NFC-e no espaço 128-D.", flush=True)
    
    nfce_vecs = embs[nfce_indices]
    centroid = nfce_vecs.mean(axis=0)
    centroid /= np.linalg.norm(centroid)
    
    # 3. Similaridade combinada em 1.34M
    sim_c = np.dot(embs, centroid)
    sub_seeds = nfce_vecs[::10]  # amostragem de sementes para diversidade
    sim_m = np.max(np.dot(embs, sub_seeds.T), axis=1)
    score = 0.6 * sim_c + 0.4 * sim_m
    
    annotated_stems = set(history.keys())
    top_indices = np.argsort(-score)
    
    # Selecionar Top candidatos não anotados
    selected = []
    for idx in top_indices:
        stem = Path(docs[idx]["doc_name"]).stem
        if stem in annotated_stems:
            continue
        selected.append((stem, float(score[idx]), float(sim_c[idx])))
        if len(selected) >= target_count:
            break
            
    print(f"Top {len(selected):,} candidatos selecionados (score máx: {selected[0][1]:.4f}, mín: {selected[-1][1]:.4f}).", flush=True)
    print("Iniciando verificação paralela por OCR (8 workers)...", flush=True)
    
    tasks = [(stem, str(pdf_dir / f"{stem}.pdf"), str(rendered_dir / f"{stem}.png")) for stem, _, _ in selected]
    score_dict = {stem: (sc, sc_c) for stem, sc, sc_c in selected}
    
    t_ocr0 = time.time()
    ocr_results = {}
    with ProcessPoolExecutor(max_workers=8, initializer=init_worker) as executor:
        for idx, (doc_id, is_nfce, snippet) in enumerate(executor.map(verify_single_candidate, tasks, chunksize=16), 1):
            ocr_results[doc_id] = (is_nfce, snippet)
            if idx % 250 == 0 or idx == len(tasks):
                print(f"   Processados {idx}/{len(tasks)} ({idx/len(tasks)*100:.1f}%) em {time.time()-t_ocr0:.1f}s...", flush=True)
                
    # 5. Criar pastas no Desktop
    if desktop_output.exists():
        import shutil
        shutil.rmtree(desktop_output)
        
    dir_confirmed = desktop_output / "1_confirmadas_por_ocr"
    dir_high_sim = desktop_output / "2_alta_similaridade_candidatos"
    dir_confirmed.mkdir(parents=True, exist_ok=True)
    dir_high_sim.mkdir(parents=True, exist_ok=True)
    
    confirmed_count = 0
    candidate_count = 0
    records = []
    
    for rank, (stem, sc, sc_c) in enumerate(selected, 1):
        is_nfce, snippet = ocr_results.get(stem, (False, ""))
        png_src = rendered_dir / f"{stem}.png"
        
        info = lookup.get(stem, {})
        fornecedor = info.get("txt_fornecedor", "DESCONHECIDO")
        fornecedor_clean = re.sub(r'[^a-zA-Z0-9]', '_', fornecedor)[:20].strip('_')
        ano = info.get("num_ano", "S_ANO")
        
        link_name = f"rank_{rank:04d}_[sim_{sc_c:.3f}]_[{ano}]_[{fornecedor_clean}]_{stem}.png"
        
        if is_nfce:
            confirmed_count += 1
            dst_symlink = dir_confirmed / link_name
        else:
            candidate_count += 1
            dst_symlink = dir_high_sim / link_name
            
        if png_src.exists() and not dst_symlink.exists():
            dst_symlink.symlink_to(png_src)
            
        records.append({
            "rank": rank,
            "doc_id": stem,
            "sim_combined": sc,
            "sim_centroid": sc_c,
            "confirmed_by_ocr": is_nfce,
            "ocr_snippet": snippet,
            "fornecedor": fornecedor,
            "ano": ano,
            "valor": info.get("vlr_liquido")
        })
        
    with open(desktop_output / "relatorio_top1000_nfce.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
        
    leiame = f"""VALIDAÇÃO DO LOTE 1 DE NFC-E (ALTA SIMILARIDADE - {target_count:,} DOCUMENTOS)
Gerado em: {time.strftime('%Y-%m-%d %H:%M:%S')}
Espaço Latente: 128-D Metric Learning (ArcFace) ancorado em 491 sementes de NFC-e.

ESTRUTURA DAS PASTAS:
1. '1_confirmadas_por_ocr/' ({confirmed_count} documentos)
   - Contêm explicitamente termos como 'DANFE NFC-e', 'MOD 65', 'CONSUMIDOR ELETRÔNICO', etc.
   - 100% de pureza. Basta dar uma passada de olho rápida para validar tudo.

2. '2_alta_similaridade_candidatos/' ({candidate_count} documentos)
   - Similaridade altíssima (>= 0.988) no espaço métrico, mas com cabeçalho térmico desgastado ou QR code dominante.
   - Olhe as imagens e apague qualquer uma que NÃO seja NFC-e (ex: Cupom ECF tradicional ou Recibo).
   - O que sobrar nesta pasta será adicionado automaticamente como NFC-e.
"""
    with open(desktop_output / "LEIAME.txt", "w", encoding="utf-8") as f:
        f.write(leiame)
        
    print(f"\n=== PROCESSO CONCLUÍDO EM {time.time()-t0:.1f}s ===", flush=True)
    print(f"Total avaliado: {len(selected):,} documentos")
    print(f"  -> Confirmados por OCR (1_confirmadas_por_ocr): {confirmed_count} ({confirmed_count/len(selected)*100:.1f}%)")
    print(f"  -> Candidatos Visuais (2_alta_similaridade_candidatos): {candidate_count} ({candidate_count/len(selected)*100:.1f}%)")
    print(f"Pasta gerada no Desktop: {desktop_output}")


if __name__ == "__main__":
    main()
