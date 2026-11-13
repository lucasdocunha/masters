#!/usr/bin/env python3
"""
Auditoria rigorosa dos 711 documentos da classe 'hotelaria':
- Classifica como 'nfse' os que são Notas Fiscais de Serviços (municipais, eletrônicas ou convencionais)
- Marca para descarte (fora das classes ativas) os que são recibos avulsos, extratos/folios ou não fiscais.
Usa pre-warming do RapidOCR por worker e timeout de 6s com signal.alarm.
"""

import os
import re
import sys
import json
import time
import signal
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import pymupdf

# Force single thread for sub-libraries to avoid deadlocks
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

RE_NFSE_POSITIVE = re.compile(
    r'(NOTA\s*FISCAL\s*(ELETR[OÔ]NICA)?\s*DE\s*SERVI[CÇ]O|NFS-?E|PREFEITURA|'
    r'SECRETARIA\s*MUNICIPAL|ISSQN|IMPOSTO\s*SOBRE\s*SERVI[CÇ]O|'
    r'RECIBO\s*PROVIS[OÓ]RIO\s*DE\s*SERVI[CÇ]O|RPS|'
    r'TRIBUTA[CÇ][AÃ]O\s*MUNICIPAL|TOMADOR\s*D[EO]\s*SERVI[CÇ]O|'
    r'PRESTADOR\s*D[EO]\s*SERVI[CÇ]O|C[OÓ]DIGO\s*DO\s*SERVI[CÇ]O|'
    r'09\.01|NOTA\s*FISCAL\s*FATURA\s*DE\s*SERVI[CÇ]O|'
    r'NOTA\s*FISCAL\s*MUNICIPAL|SERVI[CÇ]OS\s*TRIBUT[AÁ]VEIS)',
    re.IGNORECASE
)

RE_NON_NFSE = re.compile(
    r'(RECIBO\s*DO\s*PAGADOR|EXTRATO\s*DE\s*CONTA|GUEST\s*FOLIO|'
    r'COMPROVANTE\s*DE\s*HOSPEDAGEM|FICHA\s*DE\s*H[OÓ]SPEDE|'
    r'COMPROVANTE\s*DE\s*CR[EÉ]DITO|REDE\s*VISA|CIELO|PAGSEGURO|'
    r'RECIBO\s*DE\s*LOCA[CÇ][AÃ]O|CONDOM[IÍ]NIO)',
    re.IGNORECASE
)

_worker_ocr = None

def init_worker():
    global _worker_ocr
    from rapidocr_onnxruntime import RapidOCR
    _worker_ocr = RapidOCR()

def timeout_handler(signum, frame):
    raise TimeoutError("OCR timeout")

def process_one(doc_id):
    pdf_path = f"/media/ssd2/datasets/CEAP/{doc_id}.pdf"
    if not os.path.exists(pdf_path):
        return {
            "doc_id": doc_id,
            "status": "error",
            "decision": "drop",
            "reason": "pdf_not_found",
            "sample_text": ""
        }

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(6)
    
    try:
        doc = pymupdf.open(pdf_path)
        text = ""
        for p in doc:
            text += p.get_text() + " "
        doc.close()
        
        ocr_used = False
        if len(text.strip()) < 50:
            ocr_used = True
            doc = pymupdf.open(pdf_path)
            pix = doc[0].get_pixmap(dpi=100)
            img_bytes = pix.tobytes("png")
            doc.close()
            
            global _worker_ocr
            if _worker_ocr is None:
                from rapidocr_onnxruntime import RapidOCR
                _worker_ocr = RapidOCR()
            result, _ = _worker_ocr(img_bytes)
            if result:
                text = " ".join([line[1] for line in result])
            else:
                text = ""

        signal.alarm(0) # Cancel alarm

        t_clean = re.sub(r'\s+', ' ', text).strip()
        t_up = t_clean.upper()
        
        matches_pos = RE_NFSE_POSITIVE.findall(t_up)
        matches_neg = RE_NON_NFSE.findall(t_up)
        
        has_nf = "NOTA FISCAL" in t_up
        has_iss = "ISS" in t_up or "SERVI" in t_up
        has_pref = "PREFEITURA" in t_up or "MUNICIP" in t_up
        
        if matches_pos or (has_nf and (has_iss or has_pref)):
            decision = "nfse"
            reason = f"matches_positive: {[m[0] if isinstance(m, tuple) else m for m in matches_pos][:2]}"
        elif matches_neg and not has_nf:
            decision = "drop"
            reason = f"matches_negative: {[m[0] if isinstance(m, tuple) else m for m in matches_neg][:2]}"
        elif "RECIBO" in t_up and not has_nf:
            decision = "drop"
            reason = "recibo_avulso_sem_nf"
        else:
            decision = "drop"
            reason = "sem_evidencia_nfse"

        return {
            "doc_id": doc_id,
            "status": "ok",
            "ocr_used": ocr_used,
            "decision": decision,
            "reason": reason,
            "sample_text": t_clean[:120]
        }
    except TimeoutError:
        signal.alarm(0)
        return {
            "doc_id": doc_id,
            "status": "timeout",
            "decision": "drop",
            "reason": "ocr_timeout_exceeded",
            "sample_text": ""
        }
    except Exception as e:
        signal.alarm(0)
        tb = traceback.format_exc().replace("\n", " -- ")
        return {
            "doc_id": doc_id,
            "status": "error",
            "decision": "drop",
            "reason": f"exception: {tb}",
            "sample_text": ""
        }

def main():
    # Warm up OCR once in main process before spawning pool
    print("Pre-aquecendo RapidOCR...")
    init_worker()
    
    with open("data/models/annotations_history.json") as f:
        h = json.load(f)
    
    hotel_ids = [k for k, v in h.items() if v == "hotelaria"]
    print(f"Auditando {len(hotel_ids)} documentos de hotelaria...")
    
    jsonl_path = Path("data/evaluation/audit_hotelaria_progress.jsonl")
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    if jsonl_path.exists():
        jsonl_path.unlink() # Fresh start
        
    t0 = time.time()
    count = 0
    all_results = []
    
    with open(jsonl_path, "w", buffering=1, encoding="utf-8") as out_f:
        with ProcessPoolExecutor(max_workers=4, initializer=init_worker) as executor:
            futures = {executor.submit(process_one, did): did for did in hotel_ids}
            for fut in as_completed(futures):
                res = fut.result()
                out_f.write(json.dumps(res, ensure_ascii=False) + "\n")
                all_results.append(res)
                count += 1
                if count % 50 == 0 or count == len(hotel_ids):
                    elapsed = time.time() - t0
                    rate = count / max(elapsed, 0.01)
                    print(f"Progresso: {count}/{len(hotel_ids)} ({count/len(hotel_ids)*100:.1f}%) - {rate:.1f} docs/s")
                    
    nfse = [r for r in all_results if r["decision"] == "nfse"]
    drop = [r for r in all_results if r["decision"] == "drop"]
    errors = [r for r in all_results if r["status"] == "error"]
    
    print("\n" + "="*55)
    print("RESUMO FINAL DA AUDITORIA HOTELARIA:")
    print(f"  Total avaliado: {len(all_results)}")
    print(f"  Confirmados como 'nfse': {len(nfse)} ({len(nfse)/len(all_results)*100:.1f}%)")
    print(f"  Descartados (fora das classes): {len(drop)} ({len(drop)/len(all_results)*100:.1f}%)")
    print(f"  Erros técnicos: {len(errors)}")
    print("="*55)
    
    with open("data/evaluation/audit_hotelaria_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "total": len(all_results),
            "nfse_count": len(nfse),
            "drop_count": len(drop),
            "error_count": len(errors),
            "drop_samples": drop[:20],
            "nfse_samples": nfse[:10]
        }, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
