#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

# Garantir que a raiz do projeto esteja no sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gc
import numpy as np

try:
    from scripts.clusterizacao import cluster
except ModuleNotFoundError:
    from clusterizacao import cluster


def pool_document_embeddings(page_embs: np.ndarray, page_meta: list[dict], num_docs: int) -> np.ndarray:
    """Calcula embedding médio de cada documento agregando suas páginas com L2-norm."""
    doc_embs = np.zeros((num_docs, page_embs.shape[1]), dtype=np.float32)
    doc_counts = np.zeros(num_docs, dtype=np.int32)
    doc_ids = np.fromiter((m["doc_id"] for m in page_meta), dtype=np.int32, count=len(page_meta))
    np.add.at(doc_embs, doc_ids, page_embs)
    np.add.at(doc_counts, doc_ids, 1)
    doc_counts[doc_counts == 0] = 1
    doc_embs /= doc_counts[:, None]
    norms = np.linalg.norm(doc_embs, axis=-1, keepdims=True)
    return doc_embs / np.where(norms == 0, 1.0, norms)


def load_or_complete_part(part: Path, k_doc: int, k_page: int):
    """Carrega a parte ou finaliza a partir do checkpoint se a clusterização da parte foi interrompida."""
    npz_file = part.with_suffix(".npz")
    json_file = part.with_name(f"{part.stem}_metadata.json")

    if npz_file.exists() and json_file.exists():
        print(f"  Carregando parte existente: {npz_file.name}...", flush=True)
        data = np.load(npz_file)
        with open(json_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return (
            data["dino_page_embeddings"],
            data["vlm_page_embeddings"],
            data["dino_doc_embeddings"],
            data["vlm_doc_embeddings"],
            meta.get("documents", []),
            meta.get("pages", []),
        )

    # Verifica checkpoint
    ckpt_npz = part.with_name(f"{part.stem}_checkpoint.npz")
    ckpt_json = part.with_name(f"{part.stem}_checkpoint_meta.json")

    if not ckpt_npz.exists() or not ckpt_json.exists():
        raise FileNotFoundError(f"Nem arquivo final ({npz_file}) nem checkpoint ({ckpt_npz}) foram encontrados para {part}.")

    print(f"  Finalizando {part.name} a partir do checkpoint ({ckpt_npz.name})...", flush=True)
    with open(ckpt_json, "r", encoding="utf-8") as f:
        meta = json.load(f)
    docs = meta.get("documents", [])
    pages = meta.get("pages", [])

    print(f"    Carregando vetores de página ({len(pages)} páginas, {len(docs)} documentos)...", flush=True)
    ckpt_data = np.load(ckpt_npz)
    dino_page = ckpt_data["dino_page_embeddings"]
    vlm_page = ckpt_data["vlm_page_embeddings"]

    print(f"    Calculando embeddings por documento (pooling)...", flush=True)
    dino_doc = pool_document_embeddings(dino_page, pages, len(docs))
    vlm_doc = pool_document_embeddings(vlm_page, pages, len(docs))

    print(f"    Executando KMeans da parte (k_doc={k_doc}, k_page={k_page})...", flush=True)
    c_doc_dino = cluster(dino_doc, k_doc)
    c_doc_vlm = cluster(vlm_doc, k_doc)
    c_page_dino = cluster(dino_page, k_page)
    c_page_vlm = cluster(vlm_page, k_page)

    print(f"    Salvando parte finalizada em {npz_file.name}...", flush=True)
    np.savez_compressed(
        npz_file,
        dino_page_embeddings=dino_page,
        vlm_page_embeddings=vlm_page,
        dino_doc_embeddings=dino_doc,
        vlm_doc_embeddings=vlm_doc,
        clusters_doc_dino=c_doc_dino,
        clusters_doc_vlm=c_doc_vlm,
        clusters_page_dino=c_page_dino,
        clusters_page_vlm=c_page_vlm,
    )

    clean_pages = [{k: v for k, v in p.items() if k != "image"} for p in pages]
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump({"documents": docs, "pages": clean_pages}, f, ensure_ascii=False)

    print(f"    Parte {part.name} gravada com sucesso!", flush=True)
    return dino_page, vlm_page, dino_doc, vlm_doc, docs, clean_pages


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
        dino_p, vlm_p, dino_d, vlm_d, docs, pages = load_or_complete_part(part, args.k_doc, args.k_page)

        all_dino_page.append(dino_p)
        all_vlm_page.append(vlm_p)
        all_dino_doc.append(dino_d)
        all_vlm_doc.append(vlm_d)

        for d in docs:
            merged_docs.append({
                "doc_id": d["doc_id"] + doc_id_offset,
                "doc_name": d["doc_name"],
                "num_pages": d["num_pages"],
            })
        for p in pages:
            merged_pages.append({
                "doc_id": p["doc_id"] + doc_id_offset,
                "doc_name": p["doc_name"],
                "page_idx": p["page_idx"],
                "text": p.get("text", ""),
            })

        doc_id_offset += len(docs)

    dino_page_embs = np.vstack(all_dino_page)
    del all_dino_page
    gc.collect()

    vlm_page_embs = np.vstack(all_vlm_page)
    del all_vlm_page
    gc.collect()

    dino_doc_embs = np.vstack(all_dino_doc)
    del all_dino_doc
    gc.collect()

    vlm_doc_embs = np.vstack(all_vlm_doc)
    del all_vlm_doc
    gc.collect()

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
        json.dump({"documents": merged_docs, "pages": merged_pages}, f, ensure_ascii=False)

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
