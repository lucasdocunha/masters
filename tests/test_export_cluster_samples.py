from pathlib import Path
import json
from scripts.export_cluster_samples import generate_html_gallery


def test_generate_html_gallery(tmp_path: Path):
    summary = {
        "0": {
            "total_docs_in_cluster": 100,
            "cluster_percentage": 50.0,
            "exported_samples": [
                {
                    "rank": 1,
                    "doc_name": "sample.pdf",
                    "year": 2021,
                    "num_pages": 1,
                    "dist_to_centroid": 0.05,
                    "image": "01_sample.png",
                    "pdf": "01_sample.pdf",
                }
            ],
        }
    }
    out_html = tmp_path / "index.html"
    generate_html_gallery("dino", "DINOv2", summary, out_html)
    assert out_html.exists()
    content = out_html.read_text(encoding="utf-8")
    assert "DINOv2" in content
    assert "sample.pdf" in content
    assert "01_sample.png" in content
