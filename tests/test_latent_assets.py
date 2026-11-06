import gzip
import json

import numpy as np
import pytest

from dados.web.latent import build_latent_assets, doc_id_from, read_points_csv

PALETTE = {"0": {"name": "Não Mapeado", "count": 2}, "1": {"name": "nfe", "count": 1}}


def _fixture(tmp_path):
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "metadata.json").write_text(json.dumps({
        "n_points": 3, "default_model": "metric", "palette": PALETTE,
        "models": {"metric": {"file": "coords_metric.bin.gz"}},
    }))
    (legacy / "coords_metric.bin.gz").write_bytes(gzip.compress(np.zeros(9, dtype=np.float32).tobytes()))
    dataset = tmp_path / "dataset.json.gz"
    dataset.write_bytes(gzip.compress(json.dumps({"docs": [
        {"id": "20", "class": "nfe", "coords_idx": 1},
        {"id": "30", "class": "Não Mapeado", "coords_idx": 2},
    ]}).encode()))
    return legacy, dataset


def _classes(out):
    return np.frombuffer(gzip.decompress((out / "classes.bin.gz").read_bytes()), dtype=np.uint8).tolist()


def test_classes_vem_do_dataset_na_ordem_dos_pontos(tmp_path) -> None:
    legacy, dataset = _fixture(tmp_path)
    out = tmp_path / "out"
    stats = build_latent_assets(legacy, dataset, out)
    assert _classes(out) == [0, 1, 0]
    assert stats == {"points": 3, "labeled": 1, "shards": 0}
    assert json.loads((out / "index.json").read_text())["shard_size"] is None


def test_contagem_divergente_aborta(tmp_path) -> None:
    legacy, dataset = _fixture(tmp_path)
    meta = json.loads((legacy / "metadata.json").read_text())
    meta["palette"]["1"]["count"] = 99
    (legacy / "metadata.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="divergem"):
        build_latent_assets(legacy, dataset, tmp_path / "out")


def test_csv_fora_de_ordem_e_rejeitado(tmp_path) -> None:
    path = tmp_path / "p.csv"
    path.write_text("id\n20\n10\n30\n")
    with pytest.raises(ValueError, match="fora da ordem"):
        read_points_csv(path, {1: "20", 2: "30"}, 3)


def test_csv_com_linhas_faltando_e_rejeitado(tmp_path) -> None:
    path = tmp_path / "p.csv"
    path.write_text("id\n10\n20\n")
    with pytest.raises(ValueError, match="3 linhas"):
        read_points_csv(path, {1: "20"}, 3)


def test_csv_gera_shards_e_reclassifica(tmp_path) -> None:
    legacy, dataset = _fixture(tmp_path)
    path = tmp_path / "p.csv"
    url = "https://www.camara.leg.br/cota-parlamentar/documentos/publ/1/2020/{}.pdf"
    path.write_text("url;fornecedor\n" + "".join(f"{url.format(i)};F{i}\n" for i in (10, 20, 30)))
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "Ano-2020.json").write_text(json.dumps({"dados": [
        {"idDocumento": 10, "descricao": "S", "ano": 2020, "fornecedor": "ignorado"}]}))
    history = tmp_path / "h.json"
    history.write_text(json.dumps({"10": "nfe", "20": "nfe", "30": "uber"}))
    out = tmp_path / "out"
    stats = build_latent_assets(legacy, dataset, out, csv_path=path, raw_dir=raw, history_path=history)
    assert stats["shards"] == 1 and stats["labeled"] == 3
    assert _classes(out) == [1, 1, 2]
    rows = json.loads(gzip.decompress((out / "meta/00000.json.gz").read_bytes()))
    assert rows[0] == ["10", "F10", "S", 2020, url.format(10)]
    assert rows[1][2] == ""


@pytest.mark.parametrize("value,expected", [("123", "123"), ("123.pdf", "123"), ("https://x/1/2020/55.PDF", "55"), ("", None)])
def test_doc_id_from(value, expected) -> None:
    assert doc_id_from(value) == expected
