#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

# Garantir que a raiz do projeto esteja no sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

try:
    from scripts.clusterizacao import cluster
except ModuleNotFoundError:
    from clusterizacao import cluster


def main():
    parser = argparse.ArgumentParser(description="Unifica embeddings de partes e executa clusterização global")
    parser.add_argument("--parts", nargs="+", type=Path, required=True, help="Lista de prefixos das partes (ex: data/latent_vectors_part1 data/latent_vectors_part2)")
    parser.add_argument("--output-prefix", "-o", type=Path, default=Path("data/latent_vectors_sample_5pct"), help="Prefixo de saída consolidado")
    parser.add_argument("--k-doc", type=int, default=5, help="Número de clusters para documentos")
    parser.add_argument("--k-page", type=int, default=10, help="Número de clusters para páginas")
    args = parser.parse_args()

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    all_dino_page = []
    all_vlm_page = []
    all_dino_doc = []
    all_vlm_doc = []

    merged_docs = []
    merged_pages = []
    doc_id_offset = 0

    print("Carregando partes para consolidação...")
    for part in args.parts:
        npz_file = part.with_suffix(".npz")
        json_file = part.with_name(f"{part.stem}_metadata.json")

        if not npz_file.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {npz_file}")

        print(f"  Carregando {npz_file.name}...", flush=True)
        data = np.load(npz_file)
        all_dino_page.append(data["dino_page_embeddings"])
        all_vlm_page.append(data["vlm_page_embeddings"])
        all_dino_doc.append(data["dino_doc_embeddings"])
        all_vlm_doc.append(data["vlm_doc_embeddings"])

        if json_file.exists():
            with open(json_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            docs = meta.get("documents", [])
            pages = meta.get("pages", [])

            for d in docs:
                merged_docs.append({
                    "doc_id": d["doc_id"] + doc_id_offset,
                    "doc_name": d["doc_name"],
                    "num_pages": d["num_pages"]
                })
            for p in pages:
                merged_pages.append({
                    "doc_id": p["doc_id"] + doc_id_offset,
                    "doc_name": p["doc_name"],
                    "page_idx": p["page_idx"],
                    "text": p.get("text", "")
                })

            doc_id_offset += len(docs)

    dino_page_embs = np.vstack(all_dino_page)
    vlm_page_embs = np.vstack(all_vlm_page)
    dino_doc_embs = np.vstack(all_dino_doc)
    vlm_doc_embs = np.vstack(all_vlm_doc)

    print(f"\nConsolidação concluída:")
    print(f"  Documentos totais: {len(dino_doc_embs)} (DINO: {dino_doc_embs.shape}, VLM: {vlm_doc_embs.shape})")
    print(f"  Páginas totais:    {len(dino_page_embs)} (DINO: {dino_page_embs.shape}, VLM: {vlm_page_embs.shape})")

    print(f"\nExecutando clusterização global com KMeans (k_doc={args.k_doc}, k_page={args.k_page})...", flush=True)
    clusters_doc_dino = cluster(dino_doc_embs, args.k_doc)
    clusters_doc_vlm = cluster(vlm_doc_embs, args.k_doc)
    clusters_page_dino = cluster(dino_page_embs, args.k_page)
    clusters_page_vlm = cluster(vlm_page_embs, args.k_page)

    out_npz = args.output_prefix.with_suffix(".npz")
    np.savez_compressed(
        out_npz,
        dino_page_embeddings=dino_page_embs,
        vlm_page_embeddings=vlm_page_embs,
        dino_doc_embeddings=dino_doc_embs,
        vlm_doc_embeddings=vlm_doc_embs,
        clusters_doc_dino=clusters_doc_dino,
        clusters_doc_vlm=clusters_doc_vlm,
        clusters_page_dino=clusters_page_dino,
        clusters_page_vlm=clusters_page_vlm,
    )

    out_json = args.output_prefix.with_name(f"{args.output_prefix.stem}_metadata.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"documents": merged_docs, "pages": merged_pages}, f, ensure_ascii=False, indent=2)

    print(f"\n[Sucesso] Dataset global consolidado salvo em:")
    print(f"  - Vetores:  {out_npz}")
    print(f"  - Metadata: {out_json}")

    # Gerar visualização interativa 3D navegável
    try:
        from scripts.visualize_clusters_3d import generate_3d_visualization
        out_html = args.output_prefix.with_name(f"{args.output_prefix.stem}_3d.html")
        print(f"\nGerando visualizador 3D navegável dos clusters...")
        generate_3d_visualization(
            npz_path=out_npz,
            meta_path=out_json,
            output_html=out_html,
            method="umap",
            level="doc",
            subsample=25000,
            seed=42,
        )
        print(f"[Visualização 3D Concluída] Arquivo interativo: {out_html}")
    except Exception as e:
        print(f"[Aviso] Erro ao gerar visualização 3D automática: {e}")


if __name__ == "__main__":
    main()
