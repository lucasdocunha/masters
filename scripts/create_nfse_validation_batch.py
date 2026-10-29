#!/home/lucas.ocunha/masters/.venv_ocr/bin/python3
"""
Gera lote de validação para NFSe (Nota Fiscal de Serviços Eletrônica)
- Identificada como a classe com MAIOR volume de documentos não mapeados em até 96% de similaridade (38.826 documentos no total).
- Extrai lote de 1.000 documentos no topo da similaridade (>= 98%, >= 97% e >= 96%).
- Salva em: data/validacao_batches/validacao_nfse/
    ├── 1_quase_identicos_98pct/
    ├── 2_alta_similaridade_97pct/
    ├── 3_alta_similaridade_96pct/
    ├── CONFIRMADOS/
    ├── manifest.json
    └── confirmar_validacao.py
"""

import os
import sys
import re
import json
import time
import shutil
import hashlib
import urllib.parse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pymupdf
import cv2

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
cv2.setNumThreads(1)

THUMB_LARGE_DIR = Path.home() / ".cache" / "thumbnails" / "large"
THUMB_NORMAL_DIR = Path.home() / ".cache" / "thumbnails" / "normal"
THUMB_LARGE_DIR.mkdir(parents=True, exist_ok=True)
THUMB_NORMAL_DIR.mkdir(parents=True, exist_ok=True)


def render_pdf_page(task):
    stem, pdf_p_str, png_p_str = task
    try:
        png_p = Path(png_p_str)
        pdf_p = Path(pdf_p_str)
        if not png_p.exists():
            if not pdf_p.exists():
                return False
            doc = pymupdf.open(str(pdf_p))
            if len(doc) == 0:
                doc.close()
                return False
            page = doc[0]
            pix = page.get_pixmap(dpi=110, alpha=False)
            png_p.parent.mkdir(parents=True, exist_ok=True)
            pix.save(str(png_p))
            doc.close()
        return True
    except Exception:
        return False


def render_freedesktop_thumb(png_file: Path) -> bool:
    try:
        uri = "file://" + urllib.parse.quote(str(png_file.resolve()))
        h = hashlib.md5(uri.encode("utf-8")).hexdigest()
        thumb_p = THUMB_LARGE_DIR / f"{h}.png"
        if thumb_p.exists():
            return True
        img = cv2.imread(str(png_file))
        if img is None:
            return False
        h_orig, w_orig = img.shape[:2]
        scale = 256.0 / max(h_orig, w_orig)
        resized = cv2.resize(img, (int(w_orig * scale), int(h_orig * scale)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(thumb_p), resized)
        return True
    except Exception:
        return False


def main():
    t0 = time.time()
    target_count = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    base_dir = Path("/home/lucas.ocunha/masters")
    data_dir = base_dir / "data"
    pdf_dir = Path("/media/ssd2/datasets/CEAP")
    rendered_dir = data_dir / "rendered_pages"
    rendered_dir.mkdir(parents=True, exist_ok=True)

    dest_dir = data_dir / "validacao_batches" / "validacao_nfse"

    print("==================================================================")
    print(" LOTE DE VALIDAÇÃO: NFSe (CLASSE LÍDER EM SIMILARES NÃO MAPEADOS)")
    print("==================================================================")

    history_p = base_dir / "ml_backend" / "models" / "annotations_history.json"
    with open(history_p, "r", encoding="utf-8") as f:
        history = json.load(f)

    meta_p = data_dir / "latent_vectors_full_metadata.json"
    with open(meta_p, "r", encoding="utf-8") as f:
        meta = json.load(f)
    docs = meta["documents"]
    n_docs = len(docs)
    stem_to_idx = {Path(d["doc_name"]).stem: i for i, d in enumerate(docs)}

    # Sementes da classe NFSe
    nfse_stems = [s for s, lbl in history.items() if lbl == "nfse" and s in stem_to_idx]
    print(f"1. Sementes conhecidas de NFSe: {len(nfse_stems):,} documentos")

    npz = np.load(data_dir / "latent_vectors_full.npz", mmap_mode="r")
    dino = npz["dino_doc_embeddings"]

    # Calcular Centróide de NFSe
    indices_seed = [stem_to_idx[s] for s in nfse_stems]
    seed_vecs = dino[indices_seed].astype(np.float32)
    seed_vecs /= np.linalg.norm(seed_vecs, axis=1, keepdims=True)
    centroid = seed_vecs.mean(axis=0)
    centroid /= np.linalg.norm(centroid)

    lookup_p = data_dir / "pdf_metadata_lookup.json"
    with open(lookup_p, "r", encoding="utf-8") as f:
        lookup = json.load(f)

    # Similaridades em chunks
    print("2. Calculando similaridade com centróide NFSe para todos os documentos...")
    chunk_size = 100000
    sims = np.zeros(n_docs, dtype=np.float32)
    for start in range(0, n_docs, chunk_size):
        end = min(start + chunk_size, n_docs)
        sub_d = dino[start:end].astype(np.float32)
        norms = np.linalg.norm(sub_d, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        sub_d /= norms
        sims[start:end] = np.dot(sub_d, centroid)

    # Filtrar apenas documentos NÃO mapeados
    unmapped_candidates = []
    for i in range(n_docs):
        dname = docs[i]["doc_name"]
        stem = Path(dname).stem
        if stem in history:
            continue
        sim_val = float(sims[i])
        if sim_val >= 0.96:
            unmapped_candidates.append((i, stem, dname, sim_val))

    unmapped_candidates.sort(key=lambda x: -x[3])
    print(f"   Total de não mapeados com >= 96% de similaridade: {len(unmapped_candidates):,}")

    # Separar em faixas até completar o target_count (1.000)
    # Tier 1: >= 0.98
    # Tier 2: 0.97 a 0.98
    # Tier 3: 0.96 a 0.97
    tier1_all = [c for c in unmapped_candidates if c[3] >= 0.98]
    tier2_all = [c for c in unmapped_candidates if 0.97 <= c[3] < 0.98]
    tier3_all = [c for c in unmapped_candidates if 0.96 <= c[3] < 0.97]

    # Distribuição equilibrada para o lote de 1.000
    t1_selected = tier1_all[:350]
    t2_selected = tier2_all[:350]
    t3_selected = tier3_all[:300]
    total_selected = len(t1_selected) + len(t2_selected) + len(t3_selected)

    print(f"\n3. Seleção do Lote ({total_selected} documentos para validação):")
    print(f"  • Tier 1 (>= 98% similaridade):  {len(t1_selected)} documentos (de {len(tier1_all):,})")
    print(f"  • Tier 2 (97% a 98% similaridade): {len(t2_selected)} documentos (de {len(tier2_all):,})")
    print(f"  • Tier 3 (96% a 97% similaridade): {len(t3_selected)} documentos (de {len(tier3_all):,})")

    # Preparar pastas de destino
    dest_tier1 = dest_dir / "1_quase_identicos_98pct"
    dest_tier2 = dest_dir / "2_alta_similaridade_97pct"
    dest_tier3 = dest_dir / "3_alta_similaridade_96pct"
    dest_confirmados = dest_dir / "CONFIRMADOS"

    for d in [dest_tier1, dest_tier2, dest_tier3]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    dest_confirmados.mkdir(parents=True, exist_ok=True)

    all_batches = []
    for item in t1_selected:
        all_batches.append((dest_tier1, "tier1_98pct", item))
    for item in t2_selected:
        all_batches.append((dest_tier2, "tier2_97pct", item))
    for item in t3_selected:
        all_batches.append((dest_tier3, "tier3_96pct", item))

    # Renderização rápida paralela
    print(f"\n4. Renderizando PNGs (110 DPI) para {total_selected} documentos...")
    render_tasks = []
    for target_dir, tier_tag, item in all_batches:
        idx, stem, dname, sim_val = item
        pdf_path = pdf_dir / dname
        png_path = rendered_dir / f"{stem}.png"
        render_tasks.append((stem, str(pdf_path), str(png_path)))

    t_render = time.time()
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(render_pdf_page, render_tasks))
    print(f"   Renderização concluída em {time.time()-t_render:.1f}s!")

    # Criar Hardlinks e compor manifesto
    manifest = []
    png_links_created = []

    for target_dir, tier_tag, item in all_batches:
        idx, stem, dname, sim_val = item
        src_png = rendered_dir / f"{stem}.png"
        dst_png = target_dir / f"{stem}.png"

        if src_png.exists() and not dst_png.exists():
            os.link(src_png, dst_png)
            png_links_created.append(dst_png)

        info = lookup.get(dname, {})
        manifest.append({
            "doc_id": stem,
            "doc_name": dname,
            "tier": tier_tag,
            "folder": target_dir.name,
            "similarity": round(sim_val, 4),
            "fornecedor": info.get("fornecedor", ""),
            "valor_documento": info.get("valor_documento", ""),
            "data_emissao": info.get("data_emissao", ""),
            "subcota": info.get("subcota", ""),
        })

    with open(dest_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # Miniaturas FreeDesktop
    print(f"5. Gerando miniaturas FreeDesktop para o visualizador de arquivos...")
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(render_freedesktop_thumb, png_links_created))

    # Script confirmar_validacao.py
    confirm_script_code = """#!/usr/bin/env python3
\"\"\"
Confirma os documentos validados como NFSe e os integra ao annotations_history.json.
\"\"\"
import json
import sys
from pathlib import Path

BASE_DIR = Path("/home/lucas.ocunha/masters")
HISTORY_FILE = BASE_DIR / "ml_backend" / "models" / "annotations_history.json"
CURRENT_DIR = Path(__file__).resolve().parent
CONFIRMADOS_DIR = CURRENT_DIR / "CONFIRMADOS"

def main():
    print("=== CONFIRMAÇÃO DO LOTE DE NFSe ===")
    
    include_all = "-a" in sys.argv
    if include_all:
        png_files = list(CURRENT_DIR.rglob("*.png"))
    else:
        png_files = list(CONFIRMADOS_DIR.glob("*.png"))
        if not png_files:
            print(f"Nenhum arquivo PNG encontrado na pasta: {CONFIRMADOS_DIR.name}")
            print("Dica: mova as NFS-e aprovadas para a pasta CONFIRMADOS/ ou use -a para confirmar todos.")
            return

    stems = sorted(list(set(p.stem for p in png_files)))
    print(f"Total de documentos a confirmar como NFSe: {len(stems)}")

    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        history = json.load(f)

    added = 0
    already = 0
    for s in stems:
        if s not in history:
            history[s] = "nfse"
            added += 1
        elif history[s] == "nfse":
            already += 1
        else:
            print(f"  Atualizando doc {s} de '{history[s]}' para 'nfse'...")
            history[s] = "nfse"
            added += 1

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    print(f"\\n[SUCESSO] {added} novos documentos confirmados como 'nfse' em annotations_history.json!")
    print(f"Total atualizado no histórico: {len(history):,} anotações.")

if __name__ == "__main__":
    main()
"""
    confirm_path = dest_dir / "confirmar_validacao.py"
    with open(confirm_path, "w", encoding="utf-8") as f:
        f.write(confirm_script_code)
    confirm_path.chmod(0o755)

    print("\n" + "=" * 66)
    print("   LOTE DE VALIDAÇÃO DE NFSe GERADO COM SUCESSO!             ")
    print("=" * 66)
    print(f"Destino: {dest_dir}")
    print(f"  ├── 1_quase_identicos_98pct:     {len(t1_selected)} imagens")
    print(f"  ├── 2_alta_similaridade_97pct:   {len(t2_selected)} imagens")
    print(f"  ├── 3_alta_similaridade_96pct:   {len(t3_selected)} imagens")
    print(f"  ├── CONFIRMADOS/                 (pasta pronta para arrastar os aprovados)")
    print(f"  ├── manifest.json                ({len(manifest)} itens detalhados)")
    print(f"  └── confirmar_validacao.py       (script de confirmação automática)")
    print(f"Tempo total: {time.time()-t0:.1f} segundos\n")


if __name__ == "__main__":
    main()
