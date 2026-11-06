import gzip
import json

import numpy as np
import pytest

from dados.web.latent import (
    build_latent_assets,
    doc_id_from,
    load_candidates,
    merge_corrections,
    read_points_csv,
    reconstruct_points,
)

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


def _cands(*ids):
    return [{"id": i, "url": f"u{i}", "fornecedor": "", "subcota": "", "ano": ""} for i in ids]


def test_trecho_com_tantos_pontos_quanto_candidatos_e_exato() -> None:
    points, stats = reconstruct_points(_cands("1", "2", "3", "4"), {1: "2"}, 4)
    assert [p["id"] for p in points] == ["1", "2", "3", "4"]
    assert stats["exact"] == 3 and stats["uncertain"] == 0


def test_trecho_com_candidatos_sobrando_e_incerto() -> None:
    # Pontos 1 e 2 entre as âncoras 0 e 3, com 3 candidatos: falhou um documento.
    points, stats = reconstruct_points(_cands("1", "2", "3", "4", "5"), {0: "1", 3: "5"}, 4)
    assert points[1]["id"] is None
    assert [c["id"] for c in points[1]["candidates"]] == ["2", "3"]
    assert [c["id"] for c in points[2]["candidates"]] == ["3", "4"]
    assert stats["uncertain"] == 2 and stats["listed"] == 2


def test_trecho_com_pontos_demais_fica_sem_vinculo() -> None:
    points, stats = reconstruct_points(_cands("1", "2"), {0: "1", 3: "2"}, 4)
    assert points[1] is None and points[2] is None and stats["unlinked"] == 2


def test_janela_grande_nao_lista_candidatos() -> None:
    points, stats = reconstruct_points(_cands(*map(str, range(1, 12))), {0: "1", 2: "11"}, 3)
    assert points[1] is None and stats["uncertain"] == 1 and stats["listed"] == 0


def test_ancora_fora_de_ordem_aborta() -> None:
    with pytest.raises(ValueError, match="fora da ordem"):
        reconstruct_points(_cands("1", "2", "3"), {0: "3", 1: "1"}, 3)


def test_candidatos_seguem_ano_do_arquivo_deputado_e_id_como_texto(tmp_path) -> None:
    url = "https://x/publ/{}/{}/{}.pdf"
    (tmp_path / "Ano-2014.json").write_text(json.dumps({"dados": [
        {"idDocumento": 9, "urlDocumento": url.format(2, 2014, 9)},
        {"idDocumento": 30, "urlDocumento": url.format(10, 2015, 30), "fornecedor": "F"},
        {"idDocumento": 30, "urlDocumento": url.format(10, 2015, 30), "fornecedor": "repetido"},
        {"idDocumento": 100, "urlDocumento": url.format(10, 2014, 100)},
        {"idDocumento": 5, "urlDocumento": ""},
    ]}))
    (tmp_path / "Ano-2015.json").write_text(json.dumps({"dados": [
        {"idDocumento": 1, "urlDocumento": url.format(1, 2015, 1)}]}))
    rows = load_candidates(tmp_path)
    assert [r["id"] for r in rows] == ["100", "30", "9", "1"]
    assert rows[1]["fornecedor"] == "F"


def test_build_reconstroi_sem_csv_e_grava_candidatos(tmp_path) -> None:
    legacy, dataset = _fixture(tmp_path)
    raw = tmp_path / "raw"
    raw.mkdir()
    url = "https://x/publ/1/2020/{}.pdf"
    (raw / "Ano-2020.json").write_text(json.dumps({"dados": [
        {"idDocumento": i, "urlDocumento": url.format(i), "fornecedor": f"F{i}", "descricao": "S", "ano": 2020}
        for i in (10, 20, 30)]}))
    history = tmp_path / "h.json"
    history.write_text(json.dumps({"10": "nfe", "30": "nfe"}))
    out = tmp_path / "out"
    stats = build_latent_assets(legacy, dataset, out, raw_dir=raw, history_path=history)
    assert stats["exact"] == 1 and stats["anchors"] == 2
    assert _classes(out) == [1, 0, 1]
    rows = json.loads(gzip.decompress((out / "meta/00000.json.gz").read_bytes()))
    assert rows[0] == ["10", "F10", "S", 2020, url.format(10)]


def test_correcoes_vale_a_mais_recente_e_unmapped_remove() -> None:
    events = [
        {"id": "1", "class": "uber", "ts": 2},
        {"id": "1", "class": "nfe", "ts": 1},
        {"id": "2", "class": "_unmapped", "ts": 1},
        {"id": "3", "class": "nfse", "ts": 1},
        {"id": "4", "class": "nfe", "ts": 1},
    ]
    result = merge_corrections({"1": "recibo", "2": "boleto", "4": "nfe"}, events)
    assert result["history"] == {"1": "uber", "3": "nfse", "4": "nfe"}
    assert {"id": "2", "de": "boleto", "para": None} in result["changes"]
    assert len(result["changes"]) == 3


@pytest.mark.parametrize("value,expected", [("123", "123"), ("123.pdf", "123"), ("https://x/1/2020/55.PDF", "55"), ("", None)])
def test_doc_id_from(value, expected) -> None:
    assert doc_id_from(value) == expected
