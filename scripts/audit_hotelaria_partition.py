#!/usr/bin/env python3
"""
Auditor de partição para hotelaria -> nfse / drop.
Pode ser executado com argumento --part N --num-parts M para paralelismo seguro entre processos independentes.
"""

import sys
import os
import re
import json
import time
import argparse
from pathlib import Path
import pymupdf
from rapidocr_onnxruntime import RapidOCR

RE_NFSE_POSITIVE = re.compile(
    r'(NOTA\s*FISCAL\s*(ELETR[OÔ]NICA)?\s*DE\s*SERVI[CÇ]O|NFS-?E|PREFEITURA|'
    r'SECRETARIA\s*MUNICIPAL|ISSQN|IMPOSTO\s*SOBRE\s*SERVI[CÇ]O|'
    r'RECIBO\s*PROVIS[OÓ]RIO\s*DE\s*SERVI[CÇ]O|RPS|'
    r'TRIBUTA[CÇ][AÃ]O\s*MUNICIPAL|TOMADOR\s*D[EO]\s*SERVI[CÇ]O|'
    r'PRESTADOR\s*D[EO]\s*SERVI[CÇ]O|C[OÓ]DIGO\s*DO\s*SERVI[CÇ]O|'
    r'09\.01|NOTA\s*FISCAL\s*FATURA\s*DE\s*SERVI[CÇ]O|'
    r'NOTA\s*FISCAL\s*MUNICIPAL|SERVI[CÇ]OS\s*TRIBUT[AÁ]VEIS|'
    r'SAO\s*PAULO\s*NF-?E|PREFEITURA\s*DA\s*CIDADE|'
    r'MUNICIPIO\s*D[EO]|MUNIC[IÍ]PIO\s*D[EO])',
    re.IGNORECASE
)

RE_NON_NFSE = re.compile(
    r'(RECIBO\s*DO\s*PAGADOR|BALANCETE|DEMONSTRATIVO|CONDOM[IÍ]NIO|'
    r'EXTRATO\s*DE\s*CONTA|GUEST\s*FOLIO|COMPROVANTE\s*DE\s*HOSPEDAGEM|'
    r'FICHA\s*DE\s*H[OÓ]SPEDE|COMPROVANTE\s*DE\s*CR[EÉ]DITO|'
    r'REDE\s*VISA|CIELO|PAGSEGURO|RECIBO\s*DE\s*LOCA[CÇ][AÃ]O)',
    re.IGNORECASE
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--num-parts", type=int, default=1)
    args = parser.parse_args()

    with open("data/models/annotations_history.json") as f:
        h = json.load(f)
        
    hotel_ids = sorted([k for k, v in h.items() if v == "hotelaria"])
    
    # Partition
    my_ids = [doc_id for i, doc_id in enumerate(hotel_ids) if i % args.num_parts == args.part]
    print(f"[Part {args.part}/{args.num_parts}] Total de documentos alocados: {len(my_ids)}")
    
    out_dir = Path("data/evaluation/audit_parts")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"part_{args.part}.jsonl"
    
    # Check already done
    done_ids = set()
    if out_file.exists():
        with open(out_file) as f:
            for line in f:
                if line.strip():
                    try:
                        done_ids.add(json.loads(line)["doc_id"])
                    except:
                        pass
                        
    todo_ids = [d for d in my_ids if d not in done_ids]
    print(f"[Part {args.part}] Já processados: {len(done_ids)}, Restantes: {len(todo_ids)}")
    
    ocr = RapidOCR()
    t0 = time.time()
    
    with open(out_file, "a", buffering=1, encoding="utf-8") as f_out:
        for idx, doc_id in enumerate(todo_ids, start=len(done_ids) + 1):
            pdf_path = f"/media/ssd2/datasets/CEAP/{doc_id}.pdf"
            if not os.path.exists(pdf_path):
                rec = {"doc_id": doc_id, "status": "error", "decision": "drop", "reason": "not_found", "text": ""}
                f_out.write(json.dumps(rec) + "\n")
                continue
                
            try:
                doc = pymupdf.open(pdf_path)
                page = doc[0]
                text = page.get_text()
                
                # If native text is short, crop top 40% for fast OCR
                ocr_used = False
                if len(text.strip()) < 40:
                    ocr_used = True
                    rect = pymupdf.Rect(0, 0, page.rect.width, page.rect.height * 0.40)
                    pix = page.get_pixmap(clip=rect, dpi=90)
                    res, _ = ocr(pix.tobytes("png"))
                    if res:
                        text = " ".join([line[1] for line in res])
                    else:
                        text = ""
                doc.close()
                
                t_clean = re.sub(r'\s+', ' ', text).strip()
                t_up = t_clean.upper()
                
                pos = RE_NFSE_POSITIVE.findall(t_up)
                neg = RE_NON_NFSE.findall(t_up)
                has_nf = "NOTA FISCAL" in t_up or "NFS-E" in t_up or "NFSE" in t_up
                has_iss = "ISS" in t_up or "SERVI" in t_up or "TRIBUT" in t_up
                has_pref = "PREFEITURA" in t_up or "MUNICIP" in t_up
                
                if pos or (has_nf and (has_iss or has_pref)):
                    decision = "nfse"
                    reason = f"pos:{[p[0] if isinstance(p, tuple) else p for p in pos][:2]}"
                elif neg and not has_nf:
                    decision = "drop"
                    reason = f"neg:{[n[0] if isinstance(n, tuple) else n for n in neg][:2]}"
                elif "RECIBO" in t_up and not has_nf:
                    decision = "drop"
                    reason = "recibo_avulso"
                elif len(t_clean) < 20:
                    decision = "drop"
                    reason = "sem_texto_legivel"
                else:
                    # Se menciona hotel/hospedagem e diária, vamos checar se tem dados fiscais
                    decision = "drop"
                    reason = "nao_identificado_como_nfse"
                    
                rec = {
                    "doc_id": doc_id,
                    "status": "ok",
                    "ocr_used": ocr_used,
                    "decision": decision,
                    "reason": reason,
                    "sample": t_clean[:100]
                }
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                
            except Exception as e:
                rec = {"doc_id": doc_id, "status": "error", "decision": "drop", "reason": str(e), "text": ""}
                f_out.write(json.dumps(rec) + "\n")
                
            if idx % 25 == 0 or idx == len(my_ids):
                rate = (idx - len(done_ids)) / max(time.time() - t0, 0.01)
                print(f"[Part {args.part}] {idx}/{len(my_ids)} ({idx/len(my_ids)*100:.1f}%) - {rate:.2f} doc/s")

    print(f"[Part {args.part}] Concluído!")

if __name__ == "__main__":
    main()
