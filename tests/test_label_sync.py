import gzip
import json
from pathlib import Path

import numpy as np

from dados.config import DATA_DIR, MODELS_DIR
from dados.labeling.labels_csv import build_rows, normalize_history, read_labels_csv, write_labels_dir
from dados.labeling.sync import apply_to_parts, diff_labels, export_history, import_history_changes
from dados.web.latent import refresh_latent_classes


def _parts(tmp_path, labels):
    parts = tmp_path / "rotulos"
    write_labels_dir(parts, build_rows(labels, {}))
    return parts


def test_diff_marca_remocao_como_none():
    assert diff_labels({"1": "a", "2": "b"}, {"1": "a", "2": "c", "3": "d"}) == [
        {"id": "2", "de": "b", "para": "c"}, {"id": "3", "de": None, "para": "d"},
    ]
    assert diff_labels({"1": "a"}, {}) == [{"id": "1", "de": "a", "para": None}]


def test_importa_so_o_que_o_script_mudou_e_preserva_o_site(tmp_path):
    base = {"1": "recibo", "2": "recibo", "3": "nfe"}
    script = {"1": "recibo", "3": "nfe", "4": "agua"}            # script tirou o 2 e incluiu o 4
    parts = _parts(tmp_path, {"1": "recibo", "2": "recibo", "3": "boleto", "5": "uber"})  # site mudou 3 e 5
    log = tmp_path / "log.csv"
    applied = apply_to_parts(parts, log, import_history_changes(base, script), ts=7, points={"2": 42})
    assert [(c["id"], c["de"], c["para"]) for c in applied] == [("2", "recibo", None), ("4", None, "agua")]
    assert read_labels_csv(parts) == {"1": "recibo", "3": "boleto", "4": "agua", "5": "uber"}
    assert log.read_text().splitlines() == [
        "ts,doc_id,ponto,classe_anterior,classe_nova,modo", "7,2,42,recibo,_unmapped,script", "7,4,-1,_unmapped,agua,script",
    ]
    assert apply_to_parts(parts, log, import_history_changes(base, script)) == []


def test_export_mantem_ordem_das_chaves_e_acrescenta_no_fim(tmp_path):
    history = tmp_path / "h.json"
    history.write_text(json.dumps({"9": "nfe", "2.pdf": "agua", "5": "uber"}))
    changes = export_history({"2": "boleto", "9": "nfe", "7": "iptu"}, history)
    assert list(json.loads(history.read_text()).items()) == [("9", "nfe"), ("2", "boleto"), ("7", "iptu")]
    assert {c["id"] for c in changes} == {"2", "5", "7"}
    assert export_history({"2": "boleto", "9": "nfe", "7": "iptu"}, history) == []


def test_refresh_recalcula_classes_e_contagens_sem_tocar_coordenadas(tmp_path):
    latent = tmp_path / "latent"
    (latent / "meta").mkdir(parents=True)
    rows = [["10", "", "", 0, ""], None, ["11", "", "", 0, ""], {"c": [["12", "", "", 0, ""]]}]
    (latent / "meta/00000.json.gz").write_bytes(gzip.compress(json.dumps(rows).encode()))
    (latent / "classes.bin.gz").write_bytes(gzip.compress(bytes([1, 0, 0, 1])))
    (latent / "coords_metric.bin.gz").write_bytes(b"coords")
    index = {"n_points": 4, "classes": [{"id": 0, "name": "Não Mapeado", "hex": "#000", "count": 2}, {"id": 1, "name": "nfe", "hex": "#111", "count": 2}]}
    (latent / "index.json").write_text(json.dumps(index))
    stats = refresh_latent_classes(latent, {"10": "nfe", "11": "agua", "12": "nfe"})
    classes = np.frombuffer(gzip.decompress((latent / "classes.bin.gz").read_bytes()), dtype=np.uint8)
    assert classes.tolist() == [1, 0, 2, 0]  # ponto ambíguo (vários candidatos) fica sem rótulo
    out = json.loads((latent / "index.json").read_text())
    assert [(c["name"], c["count"]) for c in out["classes"]] == [("Não Mapeado", 2), ("nfe", 1), ("agua", 1)]
    assert stats == {"pontos": 4, "rotulados": 2, "mudaram": 2}
    assert (latent / "coords_metric.bin.gz").read_bytes() == b"coords"


def test_repositorio_partes_e_historico_so_divergem_pelo_que_o_site_salvou():
    """Toda diferença entre data/rotulos e o histórico tem de estar no log do site (falta rodar sync_labels.py).
    Se falhar, algum script mudou o histórico sem `sync_labels.py --importar-historico`."""
    parts = read_labels_csv(DATA_DIR / "rotulos")
    history = normalize_history(json.loads((MODELS_DIR / "annotations_history.json").read_text(encoding="utf-8")))
    latest: dict[str, str] = {}
    log = Path(DATA_DIR / "labeling/correcoes_site.csv")
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines()[1:]:
            fields = line.split(",")
            if len(fields) == 6:
                latest[fields[1]] = fields[4]
    unexplained = [c for c in diff_labels(history, parts) if latest.get(c["id"]) != (c["para"] or "_unmapped")]
    assert not unexplained, f"{len(unexplained)} diferenças sem registro no log, ex.: {unexplained[:5]}"
