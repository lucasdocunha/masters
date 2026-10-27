#!/usr/bin/env python3
"""
Classificador Otimizado dos 27.097 Cupons da Classe NFC via RapidOCR.
- Execução paralela sem sobrecarga de threads (8 workers x 2 threads ONNX)
- Detecção rápida (det_limit_side_len=480, use_cls=False)
- Early-exit por mini-lotes (avalia cabeçalho fiscal e para na primeira correspondência)
- Checkpoint incremental a cada 250 documentos
- Organização automática em lotes de 2.000 no Desktop
"""

import os
import sys
import re
import json
import time
import signal
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from collections import Counter

# Otimização estrita de variáveis de ambiente para evitar thread contention
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import cv2
cv2.setNumThreads(1)

import numpy as np
from rapidocr_onnxruntime import RapidOCR

engine = None
stop_requested = False


def sig_handler(signum, frame):
    global stop_requested
    print(f"\n[AVISO] Sinal de interrupção recebido ({signum}). Salvando checkpoint...", flush=True)
    stop_requested = True


signal.signal(signal.SIGINT, sig_handler)
signal.signal(signal.SIGTERM, sig_handler)

# Regex refinadas para classificação fiscal
RE_CUPOM = re.compile(
    r'CUP[O0D]M[\s_-]*[FE1][I1L]SCAL|C00\b|COO\s*[:\.]|CCF\s*[:\.]|ECF\b|GNF\s*[:\.]|'
    r'M[AÁ]QUINA\s*REGISTRADORA|REDU[CÇ][AÃ]O\s*Z|LEITURA\s*X|'
    r'BEMATECH|DARUMA|EPSON|SWEDA|URANO',
    re.IGNORECASE
)
RE_NFCE = re.compile(
    r'NFC-?E\b|DANFE\s*NFC|CONSUMIDOR\s*ELETR[OÔ]NIC|MOD(ELO)?\s*65|CHAVE\s*DE\s*ACESSO',
    re.IGNORECASE
)
RE_SAT = re.compile(
    r'SAT\b|CF-?E\b|EXTRATO\s*N[ºO\.]|MOD(ELO)?\s*59|DE\s*OLHO\s*NA\s*NOTA',
    re.IGNORECASE
)


def init_worker():
    global engine
    engine = RapidOCR(
        use_cls=False,
        det_limit_side_len=480,
        intra_op_num_threads=2,
        inter_op_num_threads=1
    )


def process_single_doc(task):
    """
    Classifica um documento com early-exit em mini-lotes.
    """
    global engine
    doc_id, png_path = task
    try:
        img = cv2.imread(png_path)
        if img is None:
            return doc_id, "4_outros_sem_marcador", "IMAGEM_NAO_ENCONTRADA"
        
        h = img.shape[0]
        # Corta os primeiros 48% onde se concentram cabeçalho e marcadores fiscais
        crop = img[:int(h * 0.48), :]
        
        dt_boxes, _ = engine.text_det(crop)
        if dt_boxes is None or len(dt_boxes) == 0:
            return doc_id, "4_outros_sem_marcador", ""
            
        # Ordenar caixas de cima para baixo
        dt_boxes = sorted(dt_boxes, key=lambda b: np.min(b[:, 1]))[:16]
        img_crops = engine.get_crop_img_list(crop, dt_boxes)
        
        snippets = []
        # Avaliar em mini-lotes de 4 caixas com parada antecipada (early-exit)
        for i in range(0, len(img_crops), 4):
            batch = img_crops[i:i + 4]
            rec_res, _ = engine.text_rec(batch)
            if not rec_res:
                continue
            txt = " ".join([r[0] for r in rec_res])
            snippets.append(txt)
            
            # Prioridade fiscal
            if RE_SAT.search(txt):
                return doc_id, "3_com_cfe_sat", txt[:90]
            elif RE_NFCE.search(txt):
                return doc_id, "2_com_danfe_nfce", txt[:90]
            elif RE_CUPOM.search(txt):
                return doc_id, "1_com_cupom_fiscal_ecf", txt[:90]
                
        # Verificação combinada em todas as caixas avaliadas
        full_text = " ".join(snippets)
        if RE_SAT.search(full_text):
            return doc_id, "3_com_cfe_sat", full_text[:90]
        elif RE_NFCE.search(full_text):
            return doc_id, "2_com_danfe_nfce", full_text[:90]
        elif RE_CUPOM.search(full_text):
            return doc_id, "1_com_cupom_fiscal_ecf", full_text[:90]
            
        return doc_id, "4_outros_sem_marcador", full_text[:90]
    except Exception as e:
        return doc_id, "4_outros_sem_marcador", f"ERRO: {e}"


def organize_output_folders(results_dict, output_base, metadata, rendered_dir):
    """
    Cria as pastas no Desktop e organiza os arquivos em lotes de 2.000 para não travar o Nautilus.
    """
    print("\nOrganizando pastas e atalhos no Desktop...", flush=True)
    output_base.mkdir(parents=True, exist_ok=True)
    
    categories = [
        "1_com_cupom_fiscal_ecf",
        "2_com_danfe_nfce",
        "3_com_cfe_sat",
        "4_outros_sem_marcador"
    ]
    
    for cat in categories:
        (output_base / cat).mkdir(parents=True, exist_ok=True)
        
    cat_items = {cat: [] for cat in categories}
    for doc_id, val in results_dict.items():
        cat = val[0] if isinstance(val, (list, tuple)) else val
        if cat in cat_items:
            cat_items[cat].append(doc_id)
        else:
            cat_items["4_outros_sem_marcador"].append(doc_id)
            
    for cat, doc_ids in cat_items.items():
        cat_dir = output_base / cat
        total = len(doc_ids)
        print(f"Criando atalhos para {cat}: {total:,} documentos...", flush=True)
        
        if total <= 2500:
            for i, d in enumerate(doc_ids, 1):
                m = metadata.get(f"{d}.pdf", {})
                ano = m.get("ano", 0)
                raw_forn = m.get("fornecedor", "")[:22]
                forn = re.sub(r'[^a-zA-Z0-9]', '_', raw_forn).strip('_') or "FORNECEDOR"
                forn = re.sub(r'_+', '_', forn)
                src = rendered_dir / f"{d}.png"
                if src.exists():
                    link = cat_dir / f"rank_{i:04d}_[{ano}]_[{forn}]_{d}.png"
                    if not link.exists():
                        link.symlink_to(src)
        else:
            batch_size = 2000
            for b_idx in range(0, total, batch_size):
                batch_docs = doc_ids[b_idx:b_idx + batch_size]
                sub_name = f"lote_{b_idx//batch_size + 1:02d}_{len(batch_docs)}"
                sub_dir = cat_dir / sub_name
                sub_dir.mkdir(parents=True, exist_ok=True)
                for i, d in enumerate(batch_docs, 1):
                    m = metadata.get(f"{d}.pdf", {})
                    ano = m.get("ano", 0)
                    raw_forn = m.get("fornecedor", "")[:22]
                    forn = re.sub(r'[^a-zA-Z0-9]', '_', raw_forn).strip('_') or "FORNECEDOR"
                    forn = re.sub(r'_+', '_', forn)
                    src = rendered_dir / f"{d}.png"
                    if src.exists():
                        link = sub_dir / f"rank_{i:04d}_[{ano}]_[{forn}]_{d}.png"
                        if not link.exists():
                            link.symlink_to(src)

    readme = f"""=============================================================================
CLASSIFICAÇÃO MASSIVA DOS {len(results_dict):,} CUPONS NFC POR OCR
=============================================================================

1. 1_com_cupom_fiscal_ecf/ ({len(cat_items['1_com_cupom_fiscal_ecf']):,} docs):
   - Documentos com identificadores fiscais de Cupom ECF ('CUPOM FISCAL', 'COO', 'CCF', 'ECF', 'GNF', marcas de impressoras térmicas Bematech/Daruma/Sweda/etc).
   - Particionados em subpastas de 2.000 para navegação leve e imediata no Nautilus/Thunar.

2. 2_com_danfe_nfce/ ({len(cat_items['2_com_danfe_nfce']):,} docs):
   - Documento Auxiliar da Nota Fiscal de Consumidor Eletrônica (DANFE NFC-e - Modelo 65).
   - Menção expressa a NFC-e, Modelo 65 ou Chave de Acesso de 44 dígitos.

3. 3_com_cfe_sat/ ({len(cat_items['3_com_cfe_sat']):,} docs):
   - Extrato do Cupom Fiscal Eletrônico do SAT (CF-e-SAT - Modelo 59).
   - Menção expressa a CF-e, SAT Nº ou Extrato Modelo 59.

4. 4_outros_sem_marcador/ ({len(cat_items['4_outros_sem_marcador']):,} docs):
   - Recibos desbotados sem marcadores visíveis, pedágios, comprovantes de táxi ou páginas em branco.
"""
    (output_base / "LEIAME.txt").write_text(readme, encoding="utf-8")
    print("Organização de pastas concluída com sucesso!", flush=True)


def main():
    root_dir = Path("/home/lucas.ocunha/masters")
    rendered_dir = root_dir / "data" / "rendered_pages"
    checkpoint_file = root_dir / "data" / "nfc_27k_ocr_checkpoint.json"
    output_base = Path("/home/lucas.ocunha/Desktop/validacao_nfc_todos_27k")
    
    # 1. Carregar lista de IDs NFC
    history_file = root_dir / "ml_backend" / "models" / "annotations_history.json"
    with open(history_file, "r", encoding="utf-8") as f:
        history = json.load(f)
    nfc_ids = sorted([k for k, v in history.items() if v == "nfc"])
    total_docs = len(nfc_ids)
    print(f"=== CLASSIFICADOR OCR MASSIVO: {total_docs:,} DOCUMENTOS NFC ===", flush=True)
    
    # 2. Carregar Metadados
    lookup_file = root_dir / "data" / "pdf_metadata_lookup.json"
    with open(lookup_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
        
    # 3. Carregar Checkpoint ou Inicializar com Piloto Anterior
    results_dict = {}
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                results_dict = json.load(f)
            print(f"Checkpoint carregado: {len(results_dict):,} documentos já processados.", flush=True)
        except Exception as e:
            print(f"Aviso ao ler checkpoint: {e}", flush=True)
    else:
        # Tentar aproveitar os 995 do piloto anterior
        pilot_file = Path("/home/lucas.ocunha/Desktop/validacao_nfc_agrupado_por_ocr/relatorio_ocr_detalhado.json")
        if pilot_file.exists():
            try:
                with open(pilot_file, "r", encoding="utf-8") as f:
                    pilot_data = json.load(f)
                for item in pilot_data:
                    m = re.search(r'(\d+)\.png', item.get("file", ""))
                    if m:
                        d_id = m.group(1)
                        if d_id in history and history[d_id] == "nfc":
                            results_dict[d_id] = (item["category"], item.get("snippet", ""))
                print(f"Aproveitados {len(results_dict):,} documentos pré-classificados do piloto!", flush=True)
            except Exception as e:
                print(f"Aviso ao ler piloto: {e}", flush=True)
                
    pending_ids = [d for d in nfc_ids if d not in results_dict]
    print(f"Documentos pendentes: {len(pending_ids):,}", flush=True)
    
    if pending_ids:
        tasks = [(d, str(rendered_dir / f"{d}.png")) for d in pending_ids]
        num_workers = min(8, os.cpu_count() or 4)
        print(f"Disparando {num_workers} workers otimizados (intra_op=2)...", flush=True)
        
        t0 = time.time()
        processed_count = 0
        total_pending = len(pending_ids)
        
        with ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker) as executor:
            for doc_id, cat, snippet in executor.map(process_single_doc, tasks, chunksize=8):
                results_dict[doc_id] = (cat, snippet)
                processed_count += 1
                
                # Checkpoint e status a cada 250 documentos
                if processed_count % 250 == 0 or processed_count == total_pending or stop_requested:
                    elapsed = time.time() - t0
                    speed = processed_count / elapsed if elapsed > 0 else 0
                    remaining = (total_pending - processed_count) / speed if speed > 0 else 0
                    
                    # Salvar checkpoint
                    with open(checkpoint_file, "w", encoding="utf-8") as f:
                        json.dump(results_dict, f, ensure_ascii=False)
                        
                    # Contagem em tempo real
                    counts = Counter(val[0] if isinstance(val, (list, tuple)) else val for val in results_dict.values())
                    ecf_c = counts.get("1_com_cupom_fiscal_ecf", 0)
                    nfce_c = counts.get("2_com_danfe_nfce", 0)
                    sat_c = counts.get("3_com_cfe_sat", 0)
                    out_c = counts.get("4_outros_sem_marcador", 0)
                    
                    progress_pct = (processed_count / total_pending) * 100
                    overall_pct = (len(results_dict) / total_docs) * 100
                    print(
                        f"[{processed_count:,}/{total_pending:,} ({progress_pct:4.1f}%)] | "
                        f"Total: {len(results_dict):,}/{total_docs:,} ({overall_pct:4.1f}%) | "
                        f"ECF: {ecf_c:,} | NFC-e: {nfce_c:,} | SAT: {sat_c:,} | Outros: {out_c:,} | "
                        f"Vel: {speed:4.1f} doc/s | ETA: ~{remaining/60:4.1f} min",
                        flush=True
                    )
                    
                    if stop_requested:
                        print("Parada solicitada. Checkpoint salvo com segurança.", flush=True)
                        break
                        
        total_time = time.time() - t0
        print(f"\nFase OCR concluída em {total_time/60:.1f} minutos ({processed_count/total_time:.1f} docs/s)!", flush=True)
    else:
        print("Todos os 27.097 documentos já constam no checkpoint!", flush=True)
        
    # Salvar Checkpoint Final
    with open(checkpoint_file, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, ensure_ascii=False)
        
    # Organizar atalhos no Desktop
    organize_output_folders(results_dict, output_base, metadata, rendered_dir)
    
    # Resumo Final
    final_counts = Counter(val[0] if isinstance(val, (list, tuple)) else val for val in results_dict.values())
    print("\n=======================================================", flush=True)
    print(f"CLASSIFICAÇÃO FINAL DOS {len(results_dict):,} CUPONS NFC:", flush=True)
    for cat in sorted(final_counts.keys()):
        c = final_counts[cat]
        pct = (c / len(results_dict)) * 100
        print(f"  {cat:30s}: {c:6,d} ({pct:5.1f}%)", flush=True)
    print("=======================================================", flush=True)
    print(f"Arquivos organizados e disponíveis em: {output_base}", flush=True)


if __name__ == "__main__":
    main()
