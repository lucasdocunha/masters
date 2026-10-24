import json
from pathlib import Path
import numpy as np
import pytest

from ml_backend.model import DocumentClassifierManager, DEFAULT_CLASSES
from scripts.prepare_label_studio_tasks import (
    generate_label_studio_xml_config,
    prepare_tasks,
)


def test_classifier_manager_cold_start(tmp_path: Path):
    manager = DocumentClassifierManager(model_dir=tmp_path, retrain_interval=5)
    pred = manager.predict("doc_unknown.pdf")
    assert "class" in pred
    assert pred["model_version"] == "v0-cold-start"
    assert pred["score"] > 0.0


def test_classifier_manager_training_and_retrain_trigger(tmp_path: Path):
    manager = DocumentClassifierManager(model_dir=tmp_path, retrain_interval=3)

    # Injetar embeddings sintéticos 512-D
    np.random.seed(42)
    manager.embeddings_lookup["doc1"] = np.random.randn(512).astype(np.float32)
    manager.embeddings_lookup["doc2"] = np.random.randn(512).astype(np.float32)
    manager.embeddings_lookup["doc3"] = np.random.randn(512).astype(np.float32)
    manager.embeddings_lookup["doc4"] = np.random.randn(512).astype(np.float32)
    manager.embeddings_lookup["doc5"] = np.random.randn(512).astype(np.float32)

    # Adicionar anotações 1 e 2 (não atinge retrain_interval=3)
    r1 = manager.add_annotation("doc1.pdf", "CUPOM_FISCAL_COMBUSTIVEL")
    assert r1 is False
    assert manager.new_annotations_counter == 1

    r2 = manager.add_annotation("doc2.pdf", "BILHETE_AEREO_EMBARQUE")
    assert r2 is False
    assert manager.new_annotations_counter == 2

    # Pré-carregar mais algumas anotações para ter amostras suficientes para treinar (>=5)
    manager.annotations_history["doc4"] = "CUPOM_FISCAL_COMBUSTIVEL"
    manager.annotations_history["doc5"] = "BILHETE_AEREO_EMBARQUE"

    # Adicionar anotação 3 -> deve atingir o gatilho (counter == 3) e retreinar
    r3 = manager.add_annotation("doc3.pdf", "CUPOM_FISCAL_COMBUSTIVEL")
    assert r3 is True
    assert manager.new_annotations_counter == 0
    assert manager.train_count == 1
    assert manager.version == "v1"

    # Verificar predição pós-treino
    pred = manager.predict("doc1.pdf")
    assert pred["model_version"] == "v1"
    assert pred["class"] in ["CUPOM_FISCAL_COMBUSTIVEL", "BILHETE_AEREO_EMBARQUE"]
    assert "probabilities" in pred
    assert len(pred["probabilities"]) == 2


def test_generate_label_studio_xml_config(tmp_path: Path):
    xml_file = tmp_path / "config.xml"
    generate_label_studio_xml_config(xml_file)
    assert xml_file.exists()
    content = xml_file.read_text(encoding="utf-8")
    assert "<Choices" in content
    for c in DEFAULT_CLASSES:
        assert c in content


def test_prepare_tasks_single_page_filter(tmp_path: Path):
    meta_json = tmp_path / "meta.json"
    meta_json.write_text(
        json.dumps({
            "documents": [
                {"doc_id": 1, "doc_name": "1.pdf", "num_pages": 1, "year": 2024},
                {"doc_id": 2, "doc_name": "2.pdf", "num_pages": 3, "year": 2024},  # Multi-page (deve ignorar)
                {"doc_id": 3, "doc_name": "3.pdf", "num_pages": 1, "year": 2023},
            ]
        }),
        encoding="utf-8",
    )
    meta_lookup = tmp_path / "lookup.json"
    meta_lookup.write_text(
        json.dumps({
            "1.pdf": {"subcota": "COMBUSTÍVEIS", "fornecedor": "POSTO A", "ano": 2024},
            "3.pdf": {"subcota": "TELEFONIA", "fornecedor": "CLARO", "ano": 2023},
        }),
        encoding="utf-8",
    )
    tasks_json = tmp_path / "tasks.json"
    images_dir = tmp_path / "images"

    tasks = prepare_tasks(
        metadata_json=meta_json,
        pdf_dir=tmp_path,
        output_images_dir=images_dir,
        output_tasks_json=tasks_json,
        meta_lookup_json=meta_lookup,
        n_docs=10,
    )

    assert len(tasks) == 2  # Apenas os 2 de página única
    assert tasks[0]["data"]["doc_name"] == "1.pdf"
    assert tasks[0]["data"]["num_pages"] == 1
    assert tasks[0]["data"]["subcota_oficial"] == "COMBUSTÍVEIS"
    assert tasks[1]["data"]["doc_name"] == "3.pdf"
    assert tasks_json.exists()
