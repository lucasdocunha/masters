#!/usr/bin/env python3
"""
Classificador Rápido de Cupons (ECF, NFC-e, SAT) usando RapidOCR.
Varre as amostras da pasta de validação, extrai o texto do cabeçalho
e agrupa automaticamente nas pastas correspondentes no Desktop.
"""

import os
import re
import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import cv2

# Limitar threads internas do ONNX para não sobrecarregar a CPU
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from rapidocr_onnxruntime import RapidOCR

engine = None

# Regex refinadas para identificar cada tipo
RE_CUPOM = re.compile(r'CUP[O0D]M[\s_-]*[FE1][I1L]SCAL|C00|COO\s*[:\.]|CCF\s*[:\.]|ECF\b|GNF\s*[:\.]', re.IGNORECASE)
RE_NFCE = re.compile(r'NFC-?E|DANFE\s*NFC|CONSUMIDOR\s*ELETR[OÔ]NIC|MOD(ELO)?\s*65|CHAVE\s*DE\s*ACESSO', re.IGNORECASE)
RE_SAT = re.compile(r'SAT\b|CF-?E|EXTRATO\s*N[ºO\.]|MOD(ELO)?\s*59|DE\s*OLHO\s*NA\s*NOTA', re.IGNORECASE)


def init_worker():
    global engine
    engine = RapidOCR()


def classify_single_image(task):
    img_path, target_base = task
    global engine
    try:
        real_path = img_path.resolve()
        img = cv2.imread(str(real_path))
        if img is None:
            return {"file": img_path.name, "category": "4_outros_sem_marcador", "src": real_path, "snippet": "ERRO_LEITURA"}
        
        h, w = img.shape[:2]
        # Corta os primeiros 50% da altura onde ficam cabeçalho e marcadores fiscais
        crop = img[:int(h * 0.50), :]
        
        res, _ = engine(crop)
        if not res:
            return {"file": img_path.name, "category": "4_outros_sem_marcador", "src": real_path, "snippet": ""}
        
        full_text = " ".join([r[1] for r in res])
        
        has_cupom = bool(RE_CUPOM.search(full_text))
        has_nfce = bool(RE_NFCE.search(full_text))
        has_sat = bool(RE_SAT.search(full_text))
        
        # Classificação com prioridades claras
        if has_sat:
            category = "3_com_cfe_sat"
        elif has_nfce:
            category = "2_com_danfe_nfce"
        elif has_cupom:
            category = "1_com_cupom_fiscal_ecf"
        else:
            category = "4_outros_sem_marcador"
            
        snippet = full_text[:120].strip()
        return {
            "file": img_path.name,
            "category": category,
            "src": real_path,
            "snippet": snippet,
            "has_cupom": has_cupom,
            "has_nfce": has_nfce,
            "has_sat": has_sat
        }
    except Exception as e:
        return {"file": img_path.name, "category": "4_outros_sem_marcador", "src": img_path, "snippet": f"ERRO: {e}"}


def main():
    source_dir = Path("/home/lucas.ocunha/Desktop/validacao_subtipos_nfc")
    output_dir = Path("/home/lucas.ocunha/Desktop/validacao_nfc_agrupado_por_ocr")
    
    png_files = sorted(list(source_dir.rglob("*.png")))
    print(f"=== INICIANDO OCR RÁPIDO EM {len(png_files):,} DOCUMENTOS ===")
    
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
        
    categories = [
        "1_com_cupom_fiscal_ecf",
        "2_com_danfe_nfce",
        "3_com_cfe_sat",
        "4_outros_sem_marcador"
    ]
    for cat in categories:
        (output_dir / cat).mkdir(parents=True, exist_ok=True)
        
    tasks = [(p, output_dir) for p in png_files]
    
    t0 = time.time()
    num_workers = min(12, os.cpu_count() or 4)
    print(f"Processando com {num_workers} workers paralelos...")
    
    with ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker) as executor:
        results = list(executor.map(classify_single_image, tasks))
        
    dt = time.time() - t0
    print(f"\nOCR finalizado em {dt:.2f} segundos ({len(png_files)/dt:.1f} documentos/s)!")
    
    # Criar os symlinks organizados
    counts = {cat: 0 for cat in categories}
    for r in results:
        cat = r["category"]
        counts[cat] += 1
        dest_link = output_dir / cat / r["file"]
        if not dest_link.exists():
            dest_link.symlink_to(r["src"])
            
    print("\n=== RESULTADO DO AGRUPAMENTO POR OCR ===")
    for cat in categories:
        pct = (counts[cat] / len(results)) * 100
        print(f"{cat:30s}: {counts[cat]:4d} documentos ({pct:.1f}%)")
        
    # Salvar relatório json
    with open(output_dir / "relatorio_ocr_detalhado.json", "w", encoding="utf-8") as f:
        clean_results = [{k: (str(v) if isinstance(v, Path) else v) for k, v in r.items()} for r in results]
        json.dump(clean_results, f, indent=2, ensure_ascii=False)
        
    print(f"\nPastas criadas e disponíveis em: {output_dir}")


if __name__ == "__main__":
    main()
