import numpy as np
from scripts.clusterizacao import cluster, pool_document_embeddings


def test_clustering_and_pooling() -> None:
    # 2 docs: doc 0 has 2 pages, doc 1 has 1 page
    page_embs = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    page_meta = [
        {"doc_id": 0, "page_idx": 0},
        {"doc_id": 0, "page_idx": 1},
        {"doc_id": 1, "page_idx": 0},
    ]

    doc_embs = pool_document_embeddings(page_embs, page_meta, num_docs=2)
    assert doc_embs.shape == (2, 3)

    # Verifica se os embeddings de doc estão L2-normalizados
    norms = np.linalg.norm(doc_embs, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-5)

    # Teste de clusterização
    labels = cluster(page_embs, n_clusters=2)
    assert len(labels) == 3
    assert len(set(labels)) <= 2


if __name__ == "__main__":
    test_clustering_and_pooling()
    print("OK")
