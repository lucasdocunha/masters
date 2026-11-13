import gzip
import json

import numpy as np

from dados.labeling.labels_csv import (
    HEADERS, build_rows, normalize_history, part_name, read_label_rows, read_labels_csv, update_rows, write_labels_csv,
    write_labels_dir,
)
from dados.labeling.review_queues import build_review_queues, class_centroids, rank_class, runner_up

NAMES = ["a", "b"]


def test_centroides_ignoram_sem_rotulo_e_normalizam():
    vectors = np.array([[2, 0], [0, 3], [5, 5]], dtype=np.float32)
    centroids = class_centroids(vectors, np.array([0, 1, -1]), 3)
    assert np.allclose(centroids[0], [1, 0]) and np.allclose(centroids[1], [0, 1])
    assert np.allclose(centroids[2], 0)


def test_candidatos_por_similaridade_e_auditoria_pelos_menos_similares():
    sims = np.array([[0.9, 0.1], [0.5, 0.5], [0.99, 0.0], [0.2, 0.8], [0.7, 0.3]], dtype=np.float32)
    labels = np.array([0, 0, -1, 1, -1])
    usable = np.array([True, True, True, True, False])
    queues = rank_class(sims, labels, usable, 0, top_n=5, audit_n=1)
    assert queues["candidatos"].tolist() == [2, 3]  # 4 não tem documento
    assert queues["auditoria"].tolist() == [1]
    second, margin = runner_up(sims, queues["candidatos"], 0)
    assert second.tolist() == [1, 1] and np.allclose(margin, [0.99, -0.6])


def test_build_escreve_filas(tmp_path):
    vectors = np.array([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]], dtype=np.float32)
    points = [["1", "F1", "S", 2020, "u1"], ["2", "F2", "S", 2021, "u2"], ["3", "F3", "S", 2022, "u3"], None]
    summary = build_review_queues(vectors, points, {"1": "a", "3": "b"}, NAMES, tmp_path, top_n=10, audit_n=10)
    assert summary["a"] == {"rotulados": 1, "candidatos": 2, "auditoria": 1}
    queue = json.loads(gzip.decompress((tmp_path / "a.json.gz").read_bytes()))
    first = dict(zip(queue["campos"], queue["candidatos"][0]))
    assert first["id"] == "2" and first["ponto"] == 1 and first["atual"] == "" and first["fornecedor"] == "F2"
    assert [r[0] for r in queue["candidatos"]] == ["2", "3"]
    assert json.loads((tmp_path / "index.json").read_text())["classes"][0]["nome"] == "a"


def test_csv_ida_e_volta_mantem_classe_no_fim_da_linha(tmp_path):
    labels = normalize_history({"10.pdf": "nfe", "9": "uber"})
    meta = {"10": {"link": "https://x/10.pdf", "ano": 2020, "meta": {"fornecedor": 'A, "B"'}}}
    path = tmp_path / "r.csv"
    write_labels_csv(path, build_rows(labels, meta))
    lines = path.read_text().splitlines()
    assert lines[0] == ",".join(HEADERS)
    assert lines[1].startswith("9,") and lines[2].endswith(",nfe,rotulado")
    assert read_labels_csv(path) == {"9": "uber", "10": "nfe"}
    rows = update_rows(read_label_rows(path), {"10": "boleto", "11": "agua"}, {})
    assert [r[0] for r in rows] == ["10", "11"] and rows[0][1] == "https://x/10.pdf" and rows[0][-2] == "boleto"


def test_prévia_3d_usa_distância_euclidiana(tmp_path):
    vectors = np.array([[10, 0, 0], [11, 0, 0], [0, 10, 0], [10.5, 0, 0]], dtype=np.float32)
    points = [["1", "", "", 0, ""], ["2", "", "", 0, ""], ["3", "", "", 0, ""], ["4", "", "", 0, ""]]
    build_review_queues(vectors, points, {"1": "a", "2": "a", "3": "b"}, NAMES, tmp_path, space="metric-3d")
    queue = json.loads(gzip.decompress((tmp_path / "a.json.gz").read_bytes()))
    assert queue["candidatos"][0][0] == "4" and queue["candidatos"][0][2] == 1.0
    assert json.loads((tmp_path / "index.json").read_text())["espaco"] == "metric-3d"


def test_partes_por_faixa_de_doc_id_e_apaga_as_vazias(tmp_path):
    rows = build_rows({"5": "nfe", "24999": "agua", "25000": "uber", "60000": "iptu"}, {})
    (tmp_path / "parte_0009.csv").write_text("velha")
    assert write_labels_dir(tmp_path, rows) == 4
    assert sorted(p.name for p in tmp_path.glob("*.csv")) == ["parte_0000.csv", "parte_0001.csv", "parte_0002.csv"]
    assert part_name("25000") == "parte_0001.csv"
    assert read_labels_csv(tmp_path) == {"5": "nfe", "24999": "agua", "25000": "uber", "60000": "iptu"}
    assert list(read_label_rows(tmp_path)) == ["5", "24999", "25000", "60000"]
