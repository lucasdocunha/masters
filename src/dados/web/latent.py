"""Gera web/data/latent/: coordenadas, classes e metadados por ponto do espaço latente 3D.

Todos os arquivos seguem a ordem dos pontos dos coords_*.bin.gz. O vínculo ponto -> documento
só existe para os documentos com `coords_idx` no plotly_dataset, a menos que um CSV com todos os
pontos seja informado; esse CSV é sempre conferido contra os vínculos conhecidos.
"""

import csv
import gzip
import json
import re
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

SHARD_SIZE = 5000
UNMAPPED = "Não Mapeado"
ID_COLUMNS = ("idDocumento", "id_documento", "doc_id", "id", "doc_name", "name", "arquivo", "file")
URL_COLUMNS = ("urlDocumento", "url_documento", "url", "link")
INDEX_COLUMNS = ("coords_idx", "point_idx", "ponto", "index", "idx")
TEXT_COLUMNS = {"fornecedor": ("fornecedor",), "subcota": ("subcota", "descricao"), "ano": ("ano", "year")}

# Cores das classes sobre fundo escuro, sem violeta. Com 19 classes nenhuma paleta passa no
# validador de daltonismo; a identificação também vem do nome na legenda e no tooltip e de isolar a classe.
CLASS_COLORS = {
    UNMAPPED: "#858b94",
    "agua": "#6fd0f7",
    "boleto": "#4f83ff",
    "correios": "#d4ac0d",
    "cupom_fiscal": "#ff8a2b",
    "emails": "#b9e34c",
    "energia": "#12b5c9",
    "fatura": "#2fd6b0",
    "hotelaria": "#ff5c7a",
    "iptu": "#f7a8d8",
    "nfce": "#f0562e",
    "nfe": "#2f9bff",
    "nfse": "#3fd46f",
    "passagem_aerea": "#ff4fa8",
    "passagem_rodoviaria": "#7fe3a0",
    "pedagio": "#b98545",
    "recibo": "#e6b777",
    "sem_parar": "#ffc247",
    "telefonia": "#d93a36",
    "uber": "#f2ea6b",
}


def doc_id_from(value: Any) -> str | None:
    """Extrai o id numérico de um id, nome de arquivo ou URL de PDF."""
    match = re.search(r"(\d+)(?:\.pdf)?\s*$", str(value or ""), re.IGNORECASE)
    return match.group(1) if match else None


def classes_from_docs(docs: Iterable[dict], palette: dict[str, dict], n_points: int) -> np.ndarray:
    ids = {entry["name"]: int(key) for key, entry in palette.items()}
    classes = np.zeros(n_points, dtype=np.uint8)
    for doc in docs:
        classes[doc["coords_idx"]] = ids[doc["class"]]
    return classes


def check_class_counts(classes: np.ndarray, palette: dict[str, dict]) -> None:
    counts = np.bincount(classes, minlength=len(palette))
    wrong = {e["name"]: (int(counts[int(k)]), e["count"]) for k, e in palette.items() if counts[int(k)] != e["count"]}
    if wrong:
        raise ValueError(f"Contagens de classe divergem de metadata.json (obtido, esperado): {wrong}")


def _pick(row: dict[str, str], columns: Iterable[str]) -> str:
    return next((row[c].strip() for c in columns if row.get(c)), "")


def read_points_csv(path: Path, known: dict[int, str], n_points: int) -> list[dict[str, Any] | None]:
    """Lê o CSV dos pontos e aborta se ele contrariar algum vínculo ponto -> documento conhecido."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        sample = f.read(4096)
        f.seek(0)
        dialect = csv.excel_tab if "\t" in sample.splitlines()[0] else (
            csv.excel if sample.splitlines()[0].count(",") >= sample.splitlines()[0].count(";") else _Semicolon)
        reader = csv.DictReader(f, dialect=dialect)
        columns = reader.fieldnames or []
        if not set(columns) & set(ID_COLUMNS + URL_COLUMNS):
            raise ValueError(f"CSV sem coluna de id ou URL do documento. Colunas: {columns}")
        index_column = next((c for c in INDEX_COLUMNS if c in columns), None)
        points: list[dict[str, Any] | None] = [None] * n_points
        rows = 0
        for row in reader:
            idx = int(row[index_column]) if index_column else rows
            rows += 1
            if not 0 <= idx < n_points:
                raise ValueError(f"Linha {rows}: ponto {idx} fora do intervalo 0..{n_points - 1}")
            url = _pick(row, URL_COLUMNS)
            doc_id = doc_id_from(_pick(row, ID_COLUMNS) or url)
            points[idx] = {"id": doc_id, "url": url, **{k: _pick(row, c) for k, c in TEXT_COLUMNS.items()}}
    if not index_column and rows != n_points:
        raise ValueError(f"CSV sem coluna de índice precisa de {n_points} linhas; tem {rows}")
    wrong = [i for i, doc_id in known.items() if (points[i] or {}).get("id") != doc_id]
    if wrong:
        i = wrong[0]
        raise ValueError(
            f"CSV fora da ordem dos pontos: {len(wrong)} de {len(known)} vínculos conhecidos divergem "
            f"(ponto {i}: esperado {known[i]}, obtido {(points[i] or {}).get('id')})")
    return points


class _Semicolon(csv.excel):
    delimiter = ";"


def enrich_from_raw(points: list[dict[str, Any] | None], raw_dir: Path) -> None:
    """Completa campos vazios com os JSONs anuais da Câmara, um arquivo por vez."""
    pending: dict[str, dict[str, Any]] = {}
    for point in points:
        if point and point["id"] and not all(point.get(k) for k in ("url", "fornecedor", "subcota", "ano")):
            pending[point["id"]] = point
    for path in sorted(raw_dir.glob("Ano-*.json")):
        if not pending:
            return
        with open(path, encoding="utf-8") as f:
            records = json.load(f)["dados"]
        for record in records:
            point = pending.get(str(record.get("idDocumento")))
            if point:
                point["url"] = point["url"] or record.get("urlDocumento") or ""
                point["fornecedor"] = point["fornecedor"] or record.get("fornecedor") or ""
                point["subcota"] = point["subcota"] or record.get("descricao") or ""
                point["ano"] = point["ano"] or record.get("ano") or ""


def classes_from_history(points: list[dict[str, Any] | None], history: dict[str, str], palette: dict[str, dict]) -> np.ndarray:
    ids = {entry["name"]: int(key) for key, entry in palette.items()}
    for label in sorted(set(history.values()) - set(ids)):
        key = str(len(palette))
        palette[key] = {"name": label, "count": 0}
        ids[label] = int(key)
    classes = np.zeros(len(points), dtype=np.uint8)
    for i, point in enumerate(points):
        label = history.get(point["id"]) if point else None
        if label:
            classes[i] = ids[label]
    return classes


def write_shards(points: list[dict[str, Any] | None], out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    shards = 0
    for start in range(0, len(points), SHARD_SIZE):
        rows = [[p["id"], p["fornecedor"], p["subcota"], p["ano"], p["url"]] if p else None
                for p in points[start:start + SHARD_SIZE]]
        payload = json.dumps(rows, separators=(",", ":"), ensure_ascii=False).encode()
        # mtime fixo: a mesma entrada gera os mesmos bytes.
        (out_dir / f"{start // SHARD_SIZE:05d}.json.gz").write_bytes(gzip.compress(payload, 9, mtime=0))
        shards += 1
    return shards


def build_latent_assets(
    legacy_dir: Path, dataset_path: Path, out_dir: Path, *,
    csv_path: Path | None = None, raw_dir: Path | None = None, history_path: Path | None = None,
) -> dict[str, Any]:
    meta = json.loads((legacy_dir / "metadata.json").read_text(encoding="utf-8"))
    n_points = meta["n_points"]
    palette = meta["palette"]
    with gzip.open(dataset_path, "rb") as f:
        docs = json.loads(f.read())["docs"]
    classes = classes_from_docs(docs, palette, n_points)
    check_class_counts(classes, palette)

    points = None
    if csv_path:
        points = read_points_csv(csv_path, {d["coords_idx"]: str(d["id"]) for d in docs}, n_points)
        if raw_dir:
            enrich_from_raw(points, raw_dir)
        if history_path:
            classes = classes_from_history(points, json.loads(history_path.read_text(encoding="utf-8")), palette)

    out_dir.mkdir(parents=True, exist_ok=True)
    for model in meta["models"].values():
        shutil.copyfile(legacy_dir / model["file"], out_dir / model["file"])
    (out_dir / "classes.bin.gz").write_bytes(gzip.compress(classes.tobytes(), 6, mtime=0))
    if (out_dir / "meta").exists():
        shutil.rmtree(out_dir / "meta")
    shards = write_shards(points, out_dir / "meta") if points else 0

    counts = np.bincount(classes, minlength=len(palette))
    index = {
        "n_points": n_points,
        "default_model": meta["default_model"],
        "models": meta["models"],
        "classes": [
            {"id": int(key), "name": entry["name"], "hex": CLASS_COLORS.get(entry["name"], "#c8ccd2"), "count": int(counts[int(key)])}
            for key, entry in sorted(palette.items(), key=lambda kv: int(kv[0]))
        ],
        "shard_size": SHARD_SIZE if points else None,
    }
    (out_dir / "index.json").write_text(json.dumps(index, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return {"points": n_points, "labeled": int((classes > 0).sum()), "shards": shards}
