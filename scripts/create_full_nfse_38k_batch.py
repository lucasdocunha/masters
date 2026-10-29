#!/home/lucas.ocunha/masters/.venv_ocr/bin/python3
"""
Gera o Lote COMPLETO de 38.826 documentos candidatos a NFSe (Nota Fiscal de Serviços Eletrônica)
com similaridade >= 96% em relação ao centróide de NFSe.

Organiza em lotes de no máximo 2.000 imagens para não travar o Nautilus / Thunar:
  data/validacao_batches/validacao_nfse_38k/
    ├── 1_similares_98_a_100pct/       (3.805 docs)
    │     ├── lote_01/                 (2.000 docs)
    │     └── lote_02/                 (1.805 docs)
    ├── 2_similares_97_a_98pct/        (15.342 docs)
    │     ├── lote_01/ a lote_08/      (2.000 docs cada)
    ├── 3_similares_96_a_97pct/        (19.679 docs)
    │     ├── lote_01/ a lote_10/      (2.000 docs cada)
    ├── CONFIRMADOS/                   (pasta pronta para aprovações manuais)
    ├── manifest_summary.json
    └── confirmar_validacao.py         (script com suporte a -a, --tier1, --tier2, --tier3)
"""

import os
import sys
import gzip
import json
import time
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pymupdf

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


def render_worker_task(task):
    stem, pdf_p_str, png_p_str = task
    try:
        png_p = Path(png_p_str)
        if png_p.exists():
            return True
        pdf_p = Path(pdf_p_str)
        if not pdf_p.exists():
            return False
        doc = pymupdf.open(str(pdf_p))
        if len(doc) == 0:
            doc.close()
            return False
        page = doc[0]
        pix = page.get_pixmap(dpi=110, alpha=False)
        pix.save(str(png_p))
        doc.close()
        return True
    except Exception:
        return False


def main():
    t0 = time.time()
    base_dir = Path("/home/lucas.ocunha/masters")
    data_dir = base_dir / "data"
    pdf_dir = Path("/media/ssd2/datasets/CEAP")
    rendered_dir = data_dir / "rendered_pages"
    rendered_dir.mkdir(parents=True, exist_ok=True)

    dest_dir = data_dir / "validacao_batches" / "validacao_nfse_38k"

    print("==================================================================")
    print(" GERAÇÃO DO LOTE COMPLETO DE NFSe (38.826 DOCUMENTOS >= 96% SIM)")
    print("==================================================================")

    # 1. Carregar histórico e metadados
    history_p = base_dir / "ml_backend" / "models" / "annotations_history.json"
    with open(history_p, "r", encoding="utf-8") as f:
        history = json.load(f)

    meta_p = data_dir / "latent_vectors_full_metadata.json"
    with open(meta_p, "r", encoding="utf-8") as f:
        meta = json.load(f)
    docs = meta["documents"]
    n_docs = len(docs)
    stem_to_idx = {Path(d["doc_name"]).stem: i for i, d in enumerate(docs)}

    # 2. Centróide NFSe
    nfse_stems = [s for s, lbl in history.items() if lbl == "nfse" and s in stem_to_idx]
    print(f"1. Sementes conhecidas de NFSe: {len(nfse_stems):,} documentos")

    npz = np.load(data_dir / "latent_vectors_full.npz", mmap_mode="r")
    dino = npz["dino_doc_embeddings"]

    indices_seed = [stem_to_idx[s] for s in nfse_stems]
    seed_vecs = dino[indices_seed].astype(np.float32)
    seed_vecs /= np.linalg.norm(seed_vecs, axis=1, keepdims=True)
    centroid = seed_vecs.mean(axis=0)
    centroid /= np.linalg.norm(centroid)

    lookup_p = data_dir / "pdf_metadata_lookup.json"
    with open(lookup_p, "r", encoding="utf-8") as f:
        lookup = json.load(f)

    # 3. Similaridade para todos os documentos
    print("2. Calculando similaridades com centróide NFSe para todos os documentos...")
    chunk_size = 100000
    sims = np.zeros(n_docs, dtype=np.float32)
    for start in range(0, n_docs, chunk_size):
        end = min(start + chunk_size, n_docs)
        sub_d = dino[start:end].astype(np.float32)
        norms = np.linalg.norm(sub_d, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        sub_d /= norms
        sims[start:end] = np.dot(sub_d, centroid)

    # 4. Selecionar todos os não mapeados >= 0.96
    candidates = []
    for i in range(n_docs):
        dname = docs[i]["doc_name"]
        stem = Path(dname).stem
        if stem in history:
            continue
        sim_val = float(sims[i])
        if sim_val >= 0.96:
            candidates.append((stem, dname, sim_val))

    candidates.sort(key=lambda x: -x[2])
    total_candidates = len(candidates)
    print(f"   Total de candidatos >= 96% de similaridade: {total_candidates:,}")

    # 5. Agrupar em faixas de similaridade
    tier1 = [c for c in candidates if c[2] >= 0.98]
    tier2 = [c for c in candidates if 0.97 <= c[2] < 0.98]
    tier3 = [c for c in candidates if 0.96 <= c[2] < 0.97]

    print(f"\n3. Divisão dos Candidatos:")
    print(f"  • Tier 1 (>= 98%):    {len(tier1):,} documentos")
    print(f"  • Tier 2 (97% a 98%): {len(tier2):,} documentos")
    print(f"  • Tier 3 (96% a 97%): {len(tier3):,} documentos")

    # 6. Preparar estrutura de pastas em lotes de 2.000 arquivos
    print(f"\n4. Criando estrutura de subpastas em {dest_dir.name}...")
    if dest_dir.exists():
        # Limpar subpastas antigas preservando CONFIRMADOS
        for sub in dest_dir.iterdir():
            if sub.is_dir() and sub.name != "CONFIRMADOS":
                shutil.rmtree(sub)
    dest_dir.mkdir(parents=True, exist_ok=True)
    (dest_dir / "CONFIRMADOS").mkdir(parents=True, exist_ok=True)

    tier_folders = [
        ("1_similares_98_a_100pct", tier1),
        ("2_similares_97_a_98pct", tier2),
        ("3_similares_96_a_97pct", tier3),
    ]

    all_links_to_create = []  # (src_png, dst_png, doc_info)
    batch_size = 2000

    for tier_name, tier_items in tier_folders:
        tier_path = dest_dir / tier_name
        tier_path.mkdir(parents=True, exist_ok=True)

        n_batches = (len(tier_items) + batch_size - 1) // batch_size
        for b_idx in range(n_batches):
            b_start = b_idx * batch_size
            b_end = min(b_start + batch_size, len(tier_items))
            batch_slice = tier_items[b_start:b_end]

            lote_folder_name = f"lote_{b_idx+1:02d}_{len(batch_slice)}"
            lote_path = tier_path / lote_folder_name
            lote_path.mkdir(parents=True, exist_ok=True)

            for item in batch_slice:
                stem, dname, sim_val = item
                src_png = rendered_dir / f"{stem}.png"
                dst_png = lote_path / f"{stem}.png"
                all_links_to_create.append((stem, dname, sim_val, src_png, dst_png, tier_name, lote_folder_name))

    print(f"   Pastas organizadas. Total de imagens a linkar: {len(all_links_to_create):,}")

    # 7. Identificar quais precisam de renderização
    to_render = []
    for stem, dname, sim_val, src_png, dst_png, tier_name, lote_folder_name in all_links_to_create:
        if not src_png.exists():
            pdf_path = pdf_dir / dname
            to_render.append((stem, str(pdf_path), str(src_png)))

    print(f"\n5. Renderizando {len(to_render):,} imagens pendentes (de {total_candidates:,} totais)...")
    if to_render:
        t_render = time.time()
        n_workers = min(24, os.cpu_count() or 8)
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            # Dividir em blocos de 50 para máxima vazão
            results = list(pool.map(render_worker_task, to_render, chunksize=50))
        elapsed_render = time.time() - t_render
        print(f"   Renderização concluída em {elapsed_render:.1f}s ({len(to_render)/elapsed_render:.1f} docs/s)!")
    else:
        print("   Todas as imagens já estavam em cache!")

    # 8. Criar Hardlinks em alta velocidade
    print(f"\n6. Criando hardlinks para as pastas de validação...")
    t_links = time.time()
    linked_count = 0
    manifest = []

    for stem, dname, sim_val, src_png, dst_png, tier_name, lote_folder_name in all_links_to_create:
        if src_png.exists() and not dst_png.exists():
            os.link(src_png, dst_png)
            linked_count += 1

        info = lookup.get(dname, {})
        manifest.append({
            "doc_id": stem,
            "doc_name": dname,
            "tier": tier_name,
            "lote": lote_folder_name,
            "similarity": round(sim_val, 4),
            "fornecedor": info.get("fornecedor", ""),
            "valor_documento": info.get("valor_documento", ""),
            "data_emissao": info.get("data_emissao", ""),
            "subcota": info.get("subcota", ""),
        })

    print(f"   {linked_count:,} hardlinks criados em {time.time()-t_links:.2f}s!")

    # 9. Salvar Manifesto compactado e Resumo
    print("7. Salvando manifestos...")
    manifest_gz_path = dest_dir / "manifest.json.gz"
    with gzip.open(manifest_gz_path, "wt", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False)

    summary = {
        "classe": "nfse",
        "total_documentos": total_candidates,
        "gerado_em": time.ctime(),
        "tempo_geracao_s": round(time.time() - t0, 1),
        "tiers": {
            "1_similares_98_a_100pct": len(tier1),
            "2_similares_97_a_98pct": len(tier2),
            "3_similares_96_a_97pct": len(tier3),
        }
    }
    with open(dest_dir / "manifest_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 10. Script de Confirmação Automatizada
    confirm_script_code = """#!/usr/bin/env python3
\"\"\"
Script de confirmação rápida para o lote de 38.826 NFS-e.

Uso:
  ./confirmar_validacao.py              # Confirma apenas o que estiver na pasta CONFIRMADOS/
  ./confirmar_validacao.py -a           # Confirma TODOS os 38.826 documentos do lote
  ./confirmar_validacao.py --tier1      # Confirma todos os 3.805 documentos do Tier 1 (>= 98%)
  ./confirmar_validacao.py --tier2      # Confirma todos os 15.342 documentos do Tier 2 (97-98%)
  ./confirmar_validacao.py --tier3      # Confirma todos os 19.679 documentos do Tier 3 (96-97%)
  ./confirmar_validacao.py --folder <caminho>  # Confirma uma subpasta específica (ex: lote_01)
\"\"\"
import os
import sys
import json
from pathlib import Path

BASE_DIR = Path("/home/lucas.ocunha/masters")
HISTORY_FILE = BASE_DIR / "ml_backend" / "models" / "annotations_history.json"
CURRENT_DIR = Path(__file__).resolve().parent

def main():
    print("=== CONFIRMAÇÃO DO LOTE DE NFSe (38k) ===")
    
    png_files = []
    mode = "CONFIRMADOS"

    if "-a" in sys.argv or "--all" in sys.argv:
        mode = "TODOS OS ARQUIVOS DO LOTE"
        png_files = list(CURRENT_DIR.rglob("*.png"))
    elif "--tier1" in sys.argv:
        mode = "TIER 1 (>= 98%)"
        t1 = CURRENT_DIR / "1_similares_98_a_100pct"
        png_files = list(t1.rglob("*.png"))
    elif "--tier2" in sys.argv:
        mode = "TIER 2 (97% a 98%)"
        t2 = CURRENT_DIR / "2_similares_97_a_98pct"
        png_files = list(t2.rglob("*.png"))
    elif "--tier3" in sys.argv:
        mode = "TIER 3 (96% a 97%)"
        t3 = CURRENT_DIR / "3_similares_96_a_97pct"
        png_files = list(t3.rglob("*.png"))
    elif "--folder" in sys.argv:
        idx = sys.argv.index("--folder")
        if idx + 1 < len(sys.argv):
            f_path = Path(sys.argv[idx + 1])
            if not f_path.is_absolute():
                f_path = CURRENT_DIR / f_path
            mode = f"PASTA: {f_path.name}"
            png_files = list(f_path.rglob("*.png"))
    else:
        conf_dir = CURRENT_DIR / "CONFIRMADOS"
        png_files = list(conf_dir.glob("*.png"))
        if not png_files:
            print(f"Nenhum arquivo PNG na pasta CONFIRMADOS/.")
            print("\\nOpções disponíveis:")
            print("  ./confirmar_validacao.py -a        (Confirma todos os 38.826)")
            print("  ./confirmar_validacao.py --tier1   (Confirma os 3.805 do Tier 1 >= 98%)")
            print("  ./confirmar_validacao.py --tier2   (Confirma os 15.342 do Tier 2 97-98%)")
            print("  ./confirmar_validacao.py --folder <subpasta>")
            return

    stems = sorted(list(set(p.stem for p in png_files if p.name.endswith(".png"))))
    print(f"Modo: {mode}")
    print(f"Total de documentos a integrar: {len(stems):,}")

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
            history[s] = "nfse"
            added += 1

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    print(f"\\n[SUCESSO] {added:,} novos documentos confirmados como 'nfse'!")
    print(f"Total acumulado em annotations_history.json: {len(history):,} anotações.")

if __name__ == "__main__":
    main()
"""
    confirm_path = dest_dir / "confirmar_validacao.py"
    with open(confirm_path, "w", encoding="utf-8") as f:
        f.write(confirm_script_code)
    confirm_path.chmod(0o755)

    # 11. Criar também link conveniente na pasta validacao_batches
    alt_link = data_dir / "validacao_batches" / "validacao_nfse"
    if alt_link.exists() and not alt_link.is_symlink():
        shutil.rmtree(alt_link)
    if not alt_link.exists():
        os.symlink(dest_dir.name, alt_link)

    print("\n" + "=" * 66)
    print("   LOTE COMPLETO DE 38.826 NFS-e GERADO COM SUCESSO!          ")
    print("=" * 66)
    print(f"Destino: {dest_dir}")
    print(f"  ├── 1_similares_98_a_100pct: {len(tier1):,} imagens em {len(tier_folders[0][1]) // 2000 + 1} lotes")
    print(f"  ├── 2_similares_97_a_98pct:  {len(tier2):,} imagens em {len(tier_folders[1][1]) // 2000 + 1} lotes")
    print(f"  ├── 3_similares_96_a_97pct:  {len(tier3):,} imagens em {len(tier_folders[2][1]) // 2000 + 1} lotes")
    print(f"  ├── CONFIRMADOS/             (pasta pronta para arrastar)")
    print(f"  ├── manifest_summary.json    (resumo executivo)")
    print(f"  ├── manifest.json.gz         (manifesto completo compactado)")
    print(f"  └── confirmar_validacao.py   (script com opções flexíveis)")
    print(f"Tempo total: {time.time()-t0:.1f} segundos\n")


if __name__ == "__main__":
    main()
