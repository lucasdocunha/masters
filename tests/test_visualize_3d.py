import json
from pathlib import Path
import numpy as np

from scripts.visualize_clusters_3d import reduce_to_3d, build_3d_figure, generate_3d_visualization


def test_reduce_to_3d_pca():
    data = np.random.randn(20, 128).astype(np.float32)
    coords = reduce_to_3d(data, method="pca")
    assert coords.shape == (20, 3)


def test_reduce_to_3d_umap():
    data = np.random.randn(30, 64).astype(np.float32)
    coords = reduce_to_3d(data, method="umap", n_neighbors=5)
    assert coords.shape == (30, 3)


def test_generate_3d_visualization(tmp_path: Path):
    n_docs = 25
    dino_embs = np.random.randn(n_docs, 384).astype(np.float32)
    vlm_embs = np.random.randn(n_docs, 128).astype(np.float32)
    clusters_doc_dino = np.random.randint(0, 3, size=n_docs)
    clusters_doc_vlm = np.random.randint(0, 3, size=n_docs)

    npz_path = tmp_path / "test_clusters.npz"
    np.savez_compressed(
        npz_path,
        dino_doc_embeddings=dino_embs,
        vlm_doc_embeddings=vlm_embs,
        clusters_doc_dino=clusters_doc_dino,
        clusters_doc_vlm=clusters_doc_vlm,
    )

    meta_path = tmp_path / "test_metadata.json"
    docs_meta = [{"doc_id": i, "doc_name": f"doc_{i}_2021.pdf", "num_pages": 1} for i in range(n_docs)]
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"documents": docs_meta}, f)

    out_html = tmp_path / "test_3d.html"
    generate_3d_visualization(
        npz_path=npz_path,
        meta_path=meta_path,
        output_html=out_html,
        method="pca",
        level="doc",
    )

    assert out_html.exists()
    assert out_html.stat().st_size > 50000  # Standalone HTML com Plotly
    content = out_html.read_text(encoding="utf-8")
    assert "DINOv2" in content
    assert "ColPali" in content
