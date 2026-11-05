import json

from PIL import Image

from dados.web.site import (
    build_envs,
    normalize_env,
    read_log_tails,
    scan_artifacts,
    publish_previews,
    build_models,
)


def test_normalize_kmeans_format() -> None:
    summary = {
        "0": {"total_docs_in_cluster": 10, "cluster_percentage": 5.0,
              "exported_samples": [{"rank": 1, "doc_name": "a.pdf", "image": "01_a.png",
                                    "dist_to_centroid": 0.1, "num_pages": 1, "year": 2020}]},
        "1": {"total_docs_in_cluster": 5, "cluster_percentage": 2.5, "exported_samples": []},
    }
    env = normalize_env("clusters", summary)
    assert env["k"] == 2
    assert env["total_docs"] == 15
    assert env["clusters"][0]["samples"][0]["doc_name"] == "a.pdf"


def test_normalize_faiss_format_orders_clusters_numerically() -> None:
    summary = {
        "analysis": {"10": {"total_docs": 3, "percentage": 1.0}, "2": {"total_docs": 7, "percentage": 2.0}},
        "samples": {"2": [{"rank": 1, "doc_name": "x.pdf", "image": "01_x.png"}]},
    }
    env = normalize_env("cluster-faiss-dino-k10", summary)
    assert [c["id"] for c in env["clusters"]] == ["2", "10"]
    assert env["clusters"][0]["total"] == 7
    assert env["label"] == "FAISS · DINOv2"


def test_build_envs_creates_webp_thumbnails(tmp_path) -> None:
    reports = tmp_path / "clusters"
    env_dir = reports / "clusters"
    (env_dir / "0").mkdir(parents=True)
    Image.new("RGB", (1000, 1400), "white").save(env_dir / "0" / "01_doc.png")
    (env_dir / "summary.json").write_text(json.dumps({
        "0": {"total_docs_in_cluster": 1, "cluster_percentage": 100.0,
              "exported_samples": [{"rank": 1, "doc_name": "doc.pdf", "image": "01_doc.png"}]}
    }))
    web = tmp_path / "web"
    envs = build_envs(reports, web)
    thumb = envs[0]["clusters"][0]["samples"][0]["thumb"]
    assert thumb == "assets/thumbs/clusters/0/01_doc.webp"
    with Image.open(web / thumb) as im:
        assert im.format == "WEBP" and im.width == 360


def test_scan_skips_png_and_publishes_small_text(tmp_path) -> None:
    data = tmp_path / "data"
    (data / "evaluation").mkdir(parents=True)
    (data / "reports").mkdir()
    (data / "evaluation" / "r.md").write_text("# ok")
    (data / "reports" / "img.png").write_bytes(b"x")
    arts = scan_artifacts(data)
    assert [a.rel for a in arts] == ["evaluation/r.md"]
    assert arts[0].category == "Resultados e avaliação"
    web = tmp_path / "web"
    assert publish_previews(arts, data, web) == 1
    assert (web / "files" / "evaluation" / "r.md").read_text() == "# ok"
    assert arts[0].preview == "files/evaluation/r.md"


def test_models_counts_labels_by_class(tmp_path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    (models / "annotations_history.json").write_text(json.dumps({"1": "nfc", "2": "nfc", "3": "boleto"}))
    (models / "head.pt").write_bytes(b"0")
    out = build_models(models, models / "annotations_history.json")
    assert out["annotations"]["total"] == 3
    assert out["annotations"]["by_class"] == {"nfc": 2, "boleto": 1}
    assert [a["name"] for a in out["artifacts"]] == ["annotations_history.json", "head.pt"]


def test_log_tails_limits_lines(tmp_path) -> None:
    (tmp_path / "slurm-1.out").write_text("\n".join(str(i) for i in range(500)))
    logs = read_log_tails(tmp_path, tmp_path / "data")
    assert logs[0]["lines"] == 500
    assert len(logs[0]["tail"]) == 200
    assert logs[0]["tail"][-1] == "499"
