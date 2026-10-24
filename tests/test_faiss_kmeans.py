from pathlib import Path
import numpy as np
import pytest

from scripts.run_faiss_kmeans import (
    infer_cluster_taxonomy,
    run_faiss_spherical_kmeans,
    analyze_clusters,
    generate_html_gallery_k10,
    generate_clusters_comparison_dashboard,
)


def test_infer_cluster_taxonomy():
    assert infer_cluster_taxonomy("COMBUSTÍVEIS E LUBRIFICANTES", []) == "CUPOM_FISCAL_COMBUSTIVEL"
    assert infer_cluster_taxonomy("PASSAGEM AÉREA", []) == "BILHETE_AEREO_EMBARQUE"
    assert infer_cluster_taxonomy("TELEFONIA", []) == "FATURA_TELECOMUNICACAO"
    assert infer_cluster_taxonomy("HOSPEDAGEM", []) == "NOTA_FISCAL_HOSPEDAGEM"
    # Fallback with airline supplier
    assert infer_cluster_taxonomy("OUTROS", [("GOL LINHAS AEREAS", 50)]) == "BILHETE_AEREO_EMBARQUE"
    assert infer_cluster_taxonomy("OUTROS", [("CLARO S.A.", 30)]) == "FATURA_TELECOMUNICACAO"


def test_run_faiss_spherical_kmeans():
    np.random.seed(42)
    # 200 points in 16D space
    embs = np.random.randn(200, 16).astype(np.float32)
    k = 4
    labels, similarities, centroids = run_faiss_spherical_kmeans(embs, k=k, niter=5, seed=42)

    assert labels.shape == (200,)
    assert set(np.unique(labels)).issubset(set(range(k)))
    assert similarities.shape == (200,)
    assert centroids.shape == (k, 16)
    # Centroids should be unit vectors in spherical kmeans
    norms = np.linalg.norm(centroids, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_analyze_clusters():
    labels = np.array([0, 0, 1, 1], dtype=np.int32)
    sims = np.array([0.9, 0.85, 0.7, 0.75], dtype=np.float32)
    documents = [
        {"doc_name": "doc1.pdf", "doc_id": 1},
        {"doc_name": "doc2.pdf", "doc_id": 2},
        {"doc_name": "doc3.pdf", "doc_id": 3},
        {"doc_name": "doc4.pdf", "doc_id": 4},
    ]
    meta_lookup = {
        "doc1.pdf": {"subcota": "COMBUSTÍVEIS E LUBRIFICANTES", "fornecedor": "POSTO A", "ano": 2024},
        "doc2.pdf": {"subcota": "COMBUSTÍVEIS E LUBRIFICANTES", "fornecedor": "POSTO B", "ano": 2024},
        "doc3.pdf": {"subcota": "TELEFONIA", "fornecedor": "CLARO S.A.", "ano": 2023},
        "doc4.pdf": {"subcota": "TELEFONIA", "fornecedor": "VIVO", "ano": 2023},
    }

    analysis = analyze_clusters(labels, sims, documents, meta_lookup, k=2)
    assert 0 in analysis and 1 in analysis
    assert analysis[0]["total_docs"] == 2
    assert analysis[0]["inferred_taxonomy"] == "CUPOM_FISCAL_COMBUSTIVEL"
    assert analysis[1]["inferred_taxonomy"] == "FATURA_TELECOMUNICACAO"
    assert analysis[0]["distance_mean"] == pytest.approx(1.0 - 0.875, rel=1e-3)


def test_generate_html_galleries(tmp_path: Path):
    cluster_analysis = {
        0: {
            "cluster_id": 0,
            "total_docs": 10,
            "percentage": 50.0,
            "distance_mean": 0.12,
            "inferred_taxonomy": "CUPOM_FISCAL_COMBUSTIVEL",
            "dominant_subcota": "COMBUSTÍVEIS",
            "dominant_subcota_pct": 100.0,
            "top_subcotas": [("COMBUSTÍVEIS", 10, 100.0)],
            "top_fornecedores": [("POSTO X", 10, 100.0)],
            "top_years": [(2024, 10, 100.0)],
        },
        1: {
            "cluster_id": 1,
            "total_docs": 10,
            "percentage": 50.0,
            "distance_mean": 0.15,
            "inferred_taxonomy": "BILHETE_AEREO_EMBARQUE",
            "dominant_subcota": "PASSAGEM AÉREA",
            "dominant_subcota_pct": 100.0,
            "top_subcotas": [("PASSAGEM AÉREA", 10, 100.0)],
            "top_fornecedores": [("GOL", 10, 100.0)],
            "top_years": [(2024, 10, 100.0)],
        },
    }
    exported_data = {
        0: [{
            "rank": 1,
            "doc_id": 101,
            "doc_name": "101.pdf",
            "year": 2024,
            "subcota": "COMBUSTÍVEIS",
            "fornecedor": "POSTO X",
            "num_pages": 1,
            "dist_to_centroid": 0.08,
            "image": "01_101.png",
            "pdf": "01_101.pdf",
        }],
        1: [{
            "rank": 1,
            "doc_id": 102,
            "doc_name": "102.pdf",
            "year": 2024,
            "subcota": "PASSAGEM AÉREA",
            "fornecedor": "GOL",
            "num_pages": 1,
            "dist_to_centroid": 0.11,
            "image": "01_102.png",
            "pdf": "01_102.pdf",
        }],
    }

    gallery_file = tmp_path / "gallery.html"
    generate_html_gallery_k10("vlm", "ColPali VLM", 2, cluster_analysis, exported_data, gallery_file)
    assert gallery_file.exists()
    content = gallery_file.read_text(encoding="utf-8")
    assert "CUPOM_FISCAL_COMBUSTIVEL" in content
    assert "BILHETE_AEREO_EMBARQUE" in content
    assert "POSTO X" in content

    dashboard_file = tmp_path / "dashboard.html"
    generate_clusters_comparison_dashboard(2, cluster_analysis, cluster_analysis, dashboard_file)
    assert dashboard_file.exists()
    dash_content = dashboard_file.read_text(encoding="utf-8")
    assert "ColPali VLM" in dash_content
    assert "DINOv2" in dash_content
