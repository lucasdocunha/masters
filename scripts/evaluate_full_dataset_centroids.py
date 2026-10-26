#!/usr/bin/env python3
"""
Avalia o dataset COMPLETO (1.339.678 documentos) contra os centróides das classes mapeadas.
Calcula a similaridade cosseno exata (DINOv2 + ColPali = 512-D) e quantifica:
- Documentos com similaridade >= 0.90 (distância bem pequena)
- Atribuição exclusiva (argmax com sim >= 0.90)
- Faixas de similaridade (>= 0.85, >= 0.80)
"""

import json
import time
from pathlib import Path
import numpy as np

BASE_DIR = Path("/home/lucas.ocunha/masters")
DATA_DIR = BASE_DIR / "data"
NPZ_FILE = DATA_DIR / "latent_vectors_full.npz"
META_FILE = DATA_DIR / "latent_vectors_full_metadata.json"
HISTORY_FILE = BASE_DIR / "ml_backend" / "models" / "annotations_history.json"
OUTPUT_JSON = DATA_DIR / "full_dataset_centroids_sim90_analysis.json"

CLASSES = [
    "uber",
    "nfse",
    "nfe",
    "energia",
    "nfc",
    "recibo",
    "agua",
    "passagem_aerea",
    "telefonia",
    "pedagio",
    "iptu"
]

def main():
    t0 = time.time()
    print("=== ANÁLISE DO DATASET COMPLETO POR CENTRÓIDES (SIM >= 90%) ===")

    print("1. Carregando metadados de documentos...")
    with open(META_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)
    docs = meta["documents"]
    total_docs = len(docs)
    print(f"   Total de documentos no dataset completo: {total_docs:,}")

    stem_to_idx = {Path(d["doc_name"]).stem: i for i, d in enumerate(docs)}

    print("2. Carregando histórico de anotações validadas...")
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        history = json.load(f)

    print("3. Abrindo vetores densos (mmap)...")
    npz = np.load(NPZ_FILE, mmap_mode="r")
    dino = npz["dino_doc_embeddings"]
    vlm = npz["vlm_doc_embeddings"]

    print("4. Calculando centróides normalizados (512-D)...")
    centroids = []
    class_seeds_count = {}
    for c in CLASSES:
        seeds = [s for s, l in history.items() if l == c and s in stem_to_idx]
        class_seeds_count[c] = len(seeds)
        seed_idx = [stem_to_idx[s] for s in seeds]
        
        d_chunk = dino[seed_idx]
        v_chunk = vlm[seed_idx]
        combined = np.hstack([d_chunk, v_chunk]).astype(np.float32)
        norms = np.linalg.norm(combined, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        combined /= norms
        
        c_vec = np.mean(combined, axis=0)
        c_vec /= np.linalg.norm(c_vec)
        centroids.append(c_vec)
        print(f"   - {c:<16}: {len(seeds):>4} sementes")

    C = np.array(centroids).T  # (512, len(CLASSES))

    # Métricas globais
    chunk_size = 100000
    n_chunks = (total_docs + chunk_size - 1) // chunk_size

    # Contagens
    sim_90_counts = np.zeros(len(CLASSES), dtype=np.int64)
    sim_85_counts = np.zeros(len(CLASSES), dtype=np.int64)
    sim_80_counts = np.zeros(len(CLASSES), dtype=np.int64)
    exclusive_90_counts = np.zeros(len(CLASSES), dtype=np.int64)
    exclusive_85_counts = np.zeros(len(CLASSES), dtype=np.int64)
    exclusive_80_counts = np.zeros(len(CLASSES), dtype=np.int64)

    total_with_any_90 = 0
    total_with_any_85 = 0
    total_with_any_80 = 0

    print(f"\n5. Processando {total_docs:,} documentos em {n_chunks} lotes de {chunk_size:,}...")
    t_chunks = time.time()

    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, total_docs)

        d_chunk = dino[start:end]
        v_chunk = vlm[start:end]
        comb_chunk = np.hstack([d_chunk, v_chunk]).astype(np.float32)
        norms = np.linalg.norm(comb_chunk, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        comb_chunk /= norms

        # Similaridade cosseno: (chunk_len, len(CLASSES))
        sims = np.dot(comb_chunk, C)

        # Contagens por classe (não exclusivas)
        sim_90_counts += np.sum(sims >= 0.90, axis=0)
        sim_85_counts += np.sum(sims >= 0.85, axis=0)
        sim_80_counts += np.sum(sims >= 0.80, axis=0)

        # Exclusivas (argmax)
        best_class = np.argmax(sims, axis=1)
        best_sim = np.max(sims, axis=1)

        mask_90 = best_sim >= 0.90
        mask_85 = best_sim >= 0.85
        mask_80 = best_sim >= 0.80

        total_with_any_90 += int(np.sum(mask_90))
        total_with_any_85 += int(np.sum(mask_85))
        total_with_any_80 += int(np.sum(mask_80))

        # Atribuição exclusiva
        for i in range(len(CLASSES)):
            exclusive_90_counts[i] += int(np.sum((best_class == i) & mask_90))
            exclusive_85_counts[i] += int(np.sum((best_class == i) & mask_85))
            exclusive_80_counts[i] += int(np.sum((best_class == i) & mask_80))

        if (chunk_idx + 1) % 2 == 0 or (chunk_idx + 1) == n_chunks:
            elapsed = time.time() - t_chunks
            docs_done = end
            pct = (docs_done / total_docs) * 100
            print(f"   Progresso: {docs_done:>9,} / {total_docs:,} ({pct:5.1f}%) | Tempo: {elapsed:.1f}s")

    total_time = time.time() - t0
    print(f"\n[SUCESSO] Varredura do dataset completo concluída em {total_time:.2f}s!")

    # Montar resultados
    results = {
        "total_documents": total_docs,
        "total_with_sim_ge_90": total_with_any_90,
        "total_with_sim_ge_85": total_with_any_85,
        "total_with_sim_ge_80": total_with_any_80,
        "execution_time_seconds": round(total_time, 2),
        "classes": {}
    }

    print("\n" + "=" * 95)
    print(f"{'CLASSE':<16} | {'SEMENTES':<8} | {'SIM >= 90%':<12} | {'EXCLUSIVO (>=90%)':<18} | {'% DATASET':<10} | {'SIM >= 85%':<12}")
    print("=" * 95)

    # Ordenar por contagem exclusiva >= 90%
    sorted_indices = np.argsort(-exclusive_90_counts)

    for i in sorted_indices:
        c_name = CLASSES[i]
        s_count = class_seeds_count[c_name]
        c_90 = int(sim_90_counts[i])
        c_excl_90 = int(exclusive_90_counts[i])
        c_85 = int(sim_85_counts[i])
        c_excl_85 = int(exclusive_85_counts[i])
        pct_dataset = (c_excl_90 / total_docs) * 100

        results["classes"][c_name] = {
            "seeds": s_count,
            "sim_ge_90_total": c_90,
            "exclusive_ge_90": c_excl_90,
            "pct_dataset_ge_90": round(pct_dataset, 2),
            "sim_ge_85_total": c_85,
            "exclusive_ge_85": c_excl_85,
            "sim_ge_80_total": int(sim_80_counts[i]),
            "exclusive_ge_80": int(exclusive_80_counts[i]),
        }

        print(f"{c_name:<16} | {s_count:>8,} | {c_90:>12,} | {c_excl_90:>18,} | {pct_dataset:>9.2f}% | {c_85:>12,}")

    print("=" * 95)
    unassigned_90 = total_docs - total_with_any_90
    print(f"{'TOTAL COBERTO':<16} | {'-':<8} | {'-':<12} | {total_with_any_90:>18,} | {(total_with_any_90/total_docs)*100:>9.2f}% | {total_with_any_85:>12,}")
    print(f"{'NÃO COBERTO':<16} | {'-':<8} | {'-':<12} | {unassigned_90:>18,} | {(unassigned_90/total_docs)*100:>9.2f}% | {total_docs - total_with_any_85:>12,}")
    print("=" * 95)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResultados detalhados salvos em: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
