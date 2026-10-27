#!/usr/bin/env python3
"""
Classificador Massivo dos 27k Documentos da Classe NFC usando RapidOCR.
- Execução paralela em 14 núcleos de CPU
- Checkpoint incremental a cada 500 documentos
- Organização em subpastas e lotes de 2.000 no Desktop
"""

import os
import re
import sys
import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Otimização de threads para multiprocessamento com ONNX Runtime
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import cv2
from rapidocr_onnxruntime import RapidOCR

engine = None

# Expressões regulares aprimoradas para os marcadores fiscais
RE_CUPOM = re.compile(
    r'CUP[O0D]M[\s_-]*[FE1][I1L]SCAL|C00\b|COO\s*[:\.]|CCF\s*[:\.]|ECF-IF|ECF\b|GNF\s*[:\.]|FAB\s*[:\.]|MFD\s*[:\.]',
    re.IGNORECASE
)
RE_NFCE = re.compile(
    r'NFC-?E|DANFE\s*NFC|CONSUMIDOR\s*ELETR[OÔ]NIC|MOD(ELO)?\s*65|CHAVE\s*DE\s*ACESSO',
    re.IGNORECASE
)
RE_SAT = re.compile(
    r'SAT\b|CF-?E|EXTRATO\s*N[ºO\.]|MOD(ELO)?\s*59|DE\s*OLHO\s*NA\s*NOTA',
    re.IGNORECASE
)


def init_worker():
    global engine
    engine = RapidOCR()


def process_single_doc(item):
    """
    Processa um documento: lê imagem, recorta cabeçalho superior e aplica OCR.
    """
    global engine
    doc_id, png_path_str = item
    try:
        img = cv2.imread(png_path_str)
        if img is None:
            return doc_id, "4_outros_sem_marcador", "IMAGEM_NAO_ENCONTRADA"
        
        h, w = img.shape[:2]
        # Corta a metade superior (onde ficam cabeçalho, dados fiscais e tipo de cupom)
        crop = img[:int(h * 0.48), :]
        
        # Redimensiona para largura máxima de 1000px mantendo proporção para alta velocidade
        if w > 1000:
            scale = 1000.0 / w
            crop = cv2.resize(crop, (1000, int(crop.shape[0] * scale)))
            
        res, _ = engine(crop)
        if not res:
            return doc_id, "4_outros_sem_marcador", ""
        
        full_text = " ".join([r[1] for r in res])
        
        has_cupom = bool(RE_CUPOM.search(full_text))
        has_nfce = bool(RE_NFCE.search(full_text))
        has_sat = bool(RE_SAT.search(full_text))
        
        # Categorização com prioridades fiscais
        if has_sat:
            cat = "3_com_cfe_sat"
        elif has_nfce:
            cat = "2_com_danfe_nfce"
        elif has_cupom:
            cat = "1_com_cupom_fiscal_ecf"
        else:
            cat = "4_outros_sem_marcador"
            
        return doc_id, cat, full_text[:100]
    except Exception as e:
        return doc_id, "4_outros_sem_marcador", f"ERRO: {e}"


def organize_output_folders(results_dict, output_base, metadata, rendered_dir):
    """
    Cria as pastas no Desktop e organiza os arquivos em lotes de 2.000.
    """
    print("\nOrganizando pastas e atalhos no Desktop...")
    output_base.mkdir(parents=True, exist_ok=True)
    
    categories = [
        "1_com_cupom_fiscal_ecf",
        "2_com_danfe_nfce",
        "3_com_cfe_sat",
        "4_outros_sem_marcador"
    ]
    
    for cat in categories:
        cat_dir = output_base / cat
        cat_dir.mkdir(parents=True, exist_ok=True)
        
    # Agrupar IDs por categoria
    cat_items = {cat: [] for cat in categories}
    for doc_id, (cat, snippet) in results_dict.items():
        cat_items[cat].append(doc_id)
        
    for cat, doc_ids in cat_items.items():
        cat_dir = output_base / cat
        total = len(doc_ids)
        print(f"Criando atalhos para {cat}: {total:,} documentos...")
        
        if total <= 2500:
            # Pasta única
            for i, d in enumerate(doc_ids, 1):
                m = metadata.get(f"{d}.pdf", {})
                ano = m.get("ano", 0)
                forn = re.sub(r'[^a-zA-Z0-9]', '_', m.get("fornecedor", "")[:20]).strip('_') or "FORNECEDOR"
                forn = re.sub(r'_+', '_', forn)
                src = rendered_dir / f"{d}.png"
                if src.exists():
                    link = cat_dir / f"rank_{i:04d}_[{ano}]_[{forn}]_{d}.png"
                    if not link.exists():
                        link.symlink_to(src)
        else:
            # Sub-lotes de 2.000 para não travar o Nautilus
            batch_size = 2000
            for b_idx in range(0, total, batch_size):
                batch_docs = doc_ids[b_idx:b_idx + batch_size]
                sub_name = f"lote_{b_idx//batch_size + 1:02d}_{len(batch_docs)}"
                sub_dir = cat_dir / sub_name
                sub_dir.mkdir(parents=True, exist_ok=True)
                for i, d in enumerate(batch_docs, 1):
                    m = metadata.get(f"{d}.pdf", {})
                    ano = m.get("ano", 0)
                    forn = re.sub(r'[^a-zA-Z0-9]', '_', m.get("fornecedor", "")[:20]).strip('_') or "FORNECEDOR"
                    forn = re.sub(r'_+', '_', forn)
                    src = rendered_dir / f"{d}.png"
                    if src.exists():
                        link = sub_dir / f"rank_{i:04d}_[{ano}]_[{forn}]_{d}.png"
                        if not link.exists():
                            link.symlink_to(src)

    # Criar LEIAME.txt
    readme = f"""=============================================================================
CLASSIFICAÇÃO MASSIVA DOS {len(results_dict):,} CUPONS NFC POR OCR
=============================================================================

1. 1_com_cupom_fiscal_ecf/ ({len(cat_items['1_com_cupom_fiscal_ecf']):,} docs):
   - Documentos confirmados com a palavra 'CUPOM FISCAL', 'COO', 'CCF', 'ECF' ou 'GNF'.
   - Divididos em lotes de 2.000 para navegação leve e fluida no Nautilus.

2. 2_com_danfe_nfce/ ({len(cat_items['2_com_danfe_nfce']):,} docs):
   - Documentos Auxiliares da Nota Fiscal de Consumidor Eletrônica (DANFE NFC-e - Mod 65).
   - Identificados por menção expressa a 'NFC-e' ou 'Consumidor Eletrônica'.

3. 3_com_cfe_sat/ ({len(cat_items['3_com_cfe_sat']):,} docs):
   - Cupons fiscais eletrônicos emitidos pelo SAT de São Paulo (CF-e-SAT - Mod 59).
   - Identificados por menção a 'CF-e', 'SAT' ou 'Extrato'.

4. 4_outros_sem_marcador/ ({len(cat_items['4_outros_sem_marcador']):,} docs):
   - Cupons muito desbotados, recibos manuais de táxi, pedágios ou comprovantes postais.
"""
    (output_base / "LEIAME.txt").write_text(readme, encoding="utf-8")
    print("Organização concluída com sucesso!")


def main():
    root_dir = Path("/home/lucas.ocunha/masters")
    rendered_dir = root_dir / "data" / "rendered_pages"
    checkpoint_file = root_dir / "data" / "nfc_27k_ocr_checkpoint.json"
    output_base = Path("/home/lucas.ocunha/Desktop/validacao_nfc_todos_27k")
    
    # 1. Carregar lista de 27.097 IDs de NFC
    history_file = root_dir / "ml_backend" / "models" / "annotations_history.json"
    with open(history_file, "r", encoding="utf-8") as f:
        history = json.load(f)
    nfc_ids = sorted([k for k, v in history.items() if v == "nfc"])
    total_docs = len(nfc_ids)
    print(f"=== INICIANDO OCR MASSIVO DE {total_docs:,} CUPONS NFC ===")
    
    # 2. Carregar Metadados
    lookup_file = root_dir / "data" / "pdf_metadata_lookup.json"
    with open(lookup_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
        
    # 3. Carregar Checkpoint se existir
    results_dict = {}
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                results_dict = json.load(f)
            print(f"Checkpoint recuperado: {len(results_dict):,} documentos já processados anteriormente.")
        except Exception as e:
            print(f"Aviso ao ler checkpoint: {e}")
            
    pending_ids = [d for d in nfc_ids if d not in results_dict]
    print(f"Documentos pendentes para processar: {len(pending_ids):,}")
    
    if pending_ids:
        tasks = [(d, str(rendered_dir / f"{d}.png")) for d in pending_ids]
        num_workers = min(14, os.cpu_count() or 4)
        print(f"Disparando {num_workers} processos paralelos...")
        
        t0 = time.time()
        batch_counter = 0
        total_pending = len(pending_ids)
        
        with ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker) as executor:
            # Processar em lotes usando submit/as_completed ou imap_unordered
            # Para manter memória baixa e salvar checkpoint periodicamente:
            futures = [executor.submit(process_single_doc, t) for t in tasks]
            
            for future in as_completed(futures):
                doc_id, cat, snippet = future.result()
                results_dict[doc_id] = (cat, snippet)
                batch_counter += 1
                
                # Checkpoint e status a cada 500 documentos
                if batch_counter % 500 == 0 or batch_counter == total_pending:
                    elapsed = time.time() - t0
                    speed = batch_counter / elapsed if elapsed > 0 else 0
                    remaining = (total_pending - batch_counter) / speed if speed > 0 else 0
                    
                    # Salvar checkpoint
                    with open(checkpoint_file, "w", encoding="utf-8") as f:
                        json.dump(results_dict, f, ensure_ascii=False)
                        
                    progress_pct = (batch_counter / total_pending) * 100
                    overall_pct = (len(results_dict) / total_docs) * 100
                    print(
                        f"[{batch_counter:,}/{total_pending:,} ({progress_pct:.1f}%)] | "
                        f"Total Geral: {len(results_dict):,}/{total_docs:,} ({overall_pct:.1f}%) | "
                        f"Velocidade: {speed:.1f} docs/s | Restante: ~{remaining/60:.1f} min"
                    )
                    
        total_time = time.time() - t0
        print(f"\nProcessamento concluído em {total_time/60:.1f} minutos!")
    else:
        print("Todos os documentos já estavam no checkpoint!")
        
    # 4. Salvar Checkpoint Final
    with open(checkpoint_file, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, ensure_ascii=False)
        
    # 5. Criar e Organizar as Pastas no Desktop
    organize_output_folders(results_dict, output_base, metadata, rendered_dir)
    
    # 6. Sumário Final
    from collections import Counter
    summary = Counter(cat for cat, _ in results_dict.values())
    print("\n=======================================================")
    print(f"CLASSIFICAÇÃO FINAL DOS {len(results_dict):,} DOCUMENTOS:")
    for cat, count in sorted(summary.items()):
        pct = (count / len(results_dict)) * 100
        print(f"  {cat:30s}: {count:6,d} ({pct:.1f}%)")
    print("=======================================================")
    print(f"Pasta final disponível em: {output_base}")


if __name__ == "__main__":
    main()
