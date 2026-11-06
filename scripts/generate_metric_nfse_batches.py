#!/home/lucas.ocunha/masters/.venv_ocr/bin/python3
"""
Gera do ZERO os lotes de validação de NFS-e no NOVO espaço métrico aprendido (ArcFace 128-D).
- Exclui 100% dos documentos já validados (as 19.474 NFS-e e demais classes do histórico).
- Apaga e recria a pasta validacao_nfse_38k/ do zero.
- Isola fornecedores rejeitados conhecidos em validacao_nfse_removidos_falsos_positivos/.
- Organiza os candidatos limpos em 4 Tiers por faixa de similaridade:
    • 1_similares_98_a_100pct/  (~11.7k docs em lotes de 2.000)
    • 2_similares_97_a_98pct/   (~9.9k docs em lotes de 2.000)
    • 3_similares_96_a_97pct/   (~9.1k docs em lotes de 2.000)
    • 4_similares_95_a_96pct/   (~8.2k docs em lotes de 2.000)
- Renderiza páginas pendentes em paralelo e cria hardlinks instantâneos.
- Cria atalho direto na Área de Trabalho (Desktop/validacao_nfse_38k).
"""

import os
import sys
import gzip
import json
import time
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import numpy as np
import pymupdf

BASE_DIR = Path("/home/lucas.ocunha/masters")
sys.path.append(str(BASE_DIR))
DATA_DIR = BASE_DIR / "data"
PDF_DIR = Path("/media/ssd2/datasets/CEAP")
RENDERED_DIR = DATA_DIR / "rendered_pages"
DEST_DIR = DATA_DIR / "validacao_batches" / "validacao_nfse_38k"
REMOVED_DIR = DATA_DIR / "validacao_batches" / "validacao_nfse_removidos_falsos_positivos"
DESKTOP_DIR = Path("/home/lucas.ocunha/Desktop")
HISTORY_FILE = BASE_DIR / "data" / "models" / "annotations_history.json"
META_FILE = DATA_DIR / "latent_vectors_full_metadata.json"
LOOKUP_FILE = DATA_DIR / "pdf_metadata_lookup.json"
METRIC_EMBS_FILE = DATA_DIR / "latent_vectors_metric_learned_128d.npy"

# Fornecedores rejeitados acumulados
from scripts.isolate_nfse_round3_and_confirm import REJECTED_SUPPLIERS

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
    print("==================================================================")
    print(" GERAÇÃO DO ZERO DOS LOTES DE VALIDAÇÃO NFSe (NOVO ESPAÇO MÉTRICO)")
    print("==================================================================")

    # 1. Carregar histórico e metadados
    print("\n1. Carregando anotações e metadados...")
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        history = json.load(f)

    with open(META_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)
    docs = meta["documents"]
    n_docs = len(docs)
    stem_to_idx = {Path(d["doc_name"]).stem: i for i, d in enumerate(docs)}

    with open(LOOKUP_FILE, "r", encoding="utf-8") as f:
        lookup = json.load(f)

    nfse_confirmed = [k[:-4] if k.endswith(".pdf") else k for k, v in history.items() if v == "nfse"]
    print(f"   Total no histórico: {len(history):,} docs (NFS-e confirmadas: {len(nfse_confirmed):,})")

    # 2. Centróide NFS-e no Espaço Métrico 128-D
    print("\n2. Calculando centróide de NFS-e no espaço métrico (128-D)...")
    metric_embs = np.load(METRIC_EMBS_FILE, mmap_mode="r")
    nfse_idxs = [stem_to_idx[s] for s in nfse_confirmed if s in stem_to_idx]
    centroid = np.mean(metric_embs[nfse_idxs], axis=0)
    centroid /= np.linalg.norm(centroid)

    # 3. Similaridade para todos os 1.34M documentos
    print("   Projetando similaridades contra o centróide para todos os documentos...")
    chunk_size = 100000
    sims = np.zeros(n_docs, dtype=np.float32)
    for start in range(0, n_docs, chunk_size):
        end = min(start + chunk_size, n_docs)
        chunk = metric_embs[start:end]
        sims[start:end] = np.dot(chunk, centroid)

    # 4. Selecionar candidatos não mapeados com similaridade >= 0.95
    print("\n3. Filtrando documentos não mapeados com similaridade >= 0.95...")
    candidates = []
    for i in range(n_docs):
        stem = Path(docs[i]["doc_name"]).stem
        if stem in history:
            continue
        sim_val = float(sims[i])
        if sim_val >= 0.95:
            candidates.append((stem, docs[i]["doc_name"], sim_val))

    candidates.sort(key=lambda x: -x[2])
    print(f"   Total de candidatos >= 95% não mapeados: {len(candidates):,}")

    # 5. Filtrar e isolar falsos positivos conhecidos
    def is_rejected(dname):
        forn = lookup.get(dname, {}).get("fornecedor", "").upper()
        return any(r in forn for r in REJECTED_SUPPLIERS)

    clean_candidates = []
    rejected_candidates = []
    for c in candidates:
        if is_rejected(c[1]):
            rejected_candidates.append(c)
        else:
            clean_candidates.append(c)

    print(f"   Falsos positivos conhecidos isolados: {len(rejected_candidates):,}")
    print(f"   Candidatos limpos selecionados: {len(clean_candidates):,}")

    # 6. Agrupar em Tiers por Similaridade
    tier1 = [c for c in clean_candidates if c[2] >= 0.98]
    tier2 = [c for c in clean_candidates if 0.97 <= c[2] < 0.98]
    tier3 = [c for c in clean_candidates if 0.96 <= c[2] < 0.97]
    tier4 = [c for c in clean_candidates if 0.95 <= c[2] < 0.96]

    print("\n4. Divisão dos Candidatos Limpos por Faixa de Similaridade:")
    print(f"   • Tier 1 (>= 98%):    {len(tier1):,} docs")
    print(f"   • Tier 2 (97% a 98%): {len(tier2):,} docs")
    print(f"   • Tier 3 (96% a 97%): {len(tier3):,} docs")
    print(f"   • Tier 4 (95% a 96%): {len(tier4):,} docs")
    print(f"   • Total de Candidatos: {len(clean_candidates):,} docs")

    # 7. Apagar a pasta anterior e recriar estrutura do zero
    print(f"\n5. Apagando pasta antiga e recriando {DEST_DIR.name} do zero...")
    if DEST_DIR.exists():
        shutil.rmtree(DEST_DIR)
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    RENDERED_DIR.mkdir(parents=True, exist_ok=True)

    tier_folders = [
        ("1_similares_98_a_100pct", tier1),
        ("2_similares_97_a_98pct", tier2),
        ("3_similares_96_a_97pct", tier3),
        ("4_similares_95_a_96pct", tier4),
    ]

    all_links_to_create = []
    batch_size = 2000

    for tier_name, tier_items in tier_folders:
        tier_path = DEST_DIR / tier_name
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
                src_png = RENDERED_DIR / f"{stem}.png"
                dst_png = lote_path / f"{stem}.png"
                all_links_to_create.append((stem, dname, sim_val, src_png, dst_png, tier_name, lote_folder_name))

    print(f"   Estrutura de lotes criada. Total de imagens a linkar: {len(all_links_to_create):,}")

    # 8. Renderizar imagens pendentes em paralelo
    to_render = []
    for stem, dname, sim_val, src_png, dst_png, tier_name, lote_folder_name in all_links_to_create:
        if not src_png.exists():
            pdf_path = PDF_DIR / dname
            to_render.append((stem, str(pdf_path), str(src_png)))

    print(f"\n6. Renderizando imagens pendentes ({len(to_render):,} de {len(all_links_to_create):,})...")
    if to_render:
        t_render = time.time()
        n_workers = min(24, os.cpu_count() or 8)
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            list(pool.map(render_worker_task, to_render, chunksize=50))
        elapsed_render = time.time() - t_render
        print(f"   Renderização concluída em {elapsed_render:.1f}s ({len(to_render)/elapsed_render:.1f} docs/s)!")
    else:
        print("   Todas as imagens já se encontravam renderizadas em cache!")

    # 9. Criar Hardlinks
    print(f"\n7. Criando {len(all_links_to_create):,} hardlinks nas pastas dos lotes...")
    t_links = time.time()
    linked_count = 0
    manifest = []

    for stem, dname, sim_val, src_png, dst_png, tier_name, lote_folder_name in all_links_to_create:
        if src_png.exists() and not dst_png.exists():
            try:
                os.link(src_png, dst_png)
                linked_count += 1
            except Exception:
                shutil.copy2(src_png, dst_png)
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

    print(f"   {linked_count:,} arquivos organizados em {time.time()-t_links:.2f}s!")

    # 10. Salvar Manifestos
    print("\n8. Gravando manifest_summary.json e manifest.json.gz...")
    manifest_gz_path = DEST_DIR / "manifest.json.gz"
    with gzip.open(manifest_gz_path, "wt", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False)

    summary = {
        "status": "criado_do_zero",
        "espaco_metrico": "ArcFace_128D",
        "data_geracao": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_candidatos_limpos": len(clean_candidates),
        "total_falsos_positivos_isolados": len(rejected_candidates),
        "tiers": {
            "1_similares_98_a_100pct": {
                "total": len(tier1),
                "lotes": (len(tier1) + batch_size - 1) // batch_size
            },
            "2_similares_97_a_98pct": {
                "total": len(tier2),
                "lotes": (len(tier2) + batch_size - 1) // batch_size
            },
            "3_similares_96_a_97pct": {
                "total": len(tier3),
                "lotes": (len(tier3) + batch_size - 1) // batch_size
            },
            "4_similares_95_a_96pct": {
                "total": len(tier4),
                "lotes": (len(tier4) + batch_size - 1) // batch_size
            },
        }
    }

    with open(DEST_DIR / "manifest_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 11. Atualizar atalho na Área de Trabalho (Desktop)
    desktop_shortcut = DESKTOP_DIR / "validacao_nfse_38k"
    if desktop_shortcut.is_symlink() or desktop_shortcut.exists():
        desktop_shortcut.unlink()
    desktop_shortcut.symlink_to(DEST_DIR)
    print(f"   Atalho atualizado na Área de Trabalho: {desktop_shortcut}")

    elapsed = time.time() - t0
    print(f"\n==================================================================")
    print(f" PROCESSO CONCLUÍDO COM SUCESSO EM {elapsed:.1f}s!")
    print(f" Total organizado: {len(clean_candidates):,} documentos limpos prontos!")
    print(f"==================================================================")


if __name__ == "__main__":
    main()
