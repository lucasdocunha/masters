"""Filas de revisão por classe para a rotulagem no site (web/data/review/).

Para cada classe, o centróide é a média dos embeddings métricos (L2-normalizados) dos documentos
rotulados nela. Cada fila tem:
- `candidatos`: documentos fora da classe mais similares ao centróide (modo pasta e modo documento);
- `auditoria`: rotulados da classe menos similares ao centróide (suspeitos de erro).

A linha i dos embeddings é o ponto i do espaço latente; o documento de cada ponto vem das shards
web/data/latent/meta/ para que site e filas usem o mesmo vínculo ponto -> documento.
"""

import gzip
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

FIELDS = ["id", "ponto", "sim", "atual", "segunda", "margem", "fornecedor", "subcota", "ano", "url"]
CHUNK = 100_000


def normalize(rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float32)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return rows / norms


def class_centroids(vectors: np.ndarray, labels: np.ndarray, n_classes: int, cosine: bool = True) -> np.ndarray:
    """Centróides (n_classes, d); classes sem exemplos ficam com vetor nulo. `labels` < 0 = sem rótulo.

    Com `cosine`, média dos vetores normalizados, renormalizada; sem, média simples (espaço euclidiano)."""
    sums = np.zeros((n_classes, vectors.shape[1]), dtype=np.float64)
    counts = np.zeros(n_classes, dtype=np.int64)
    for start in range(0, len(vectors), CHUNK):
        chunk = labels[start:start + CHUNK]
        mask = chunk >= 0
        rows = np.asarray(vectors[start:start + CHUNK][mask], dtype=np.float32)
        np.add.at(sums, chunk[mask], normalize(rows) if cosine else rows)
        counts += np.bincount(chunk[mask], minlength=n_classes)
    if not cosine:
        return (sums / np.maximum(counts, 1)[:, None]).astype(np.float32)
    norms = np.linalg.norm(sums, axis=1, keepdims=True)
    return np.divide(sums, norms, out=np.zeros_like(sums), where=norms > 0).astype(np.float32)


def similarities(vectors: np.ndarray, centroids: np.ndarray, cosine: bool = True) -> np.ndarray:
    """Cosseno, ou 1 / (1 + distância euclidiana) quando `cosine` é falso. Maior = mais perto."""
    out = np.empty((len(vectors), len(centroids)), dtype=np.float32)
    for start in range(0, len(vectors), CHUNK):
        rows = np.asarray(vectors[start:start + CHUNK], dtype=np.float32)
        if cosine:
            out[start:start + CHUNK] = normalize(rows) @ centroids.T
        else:
            dist = np.sqrt(((rows[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2))
            out[start:start + CHUNK] = 1.0 / (1.0 + dist)
    return out


def rank_class(sims: np.ndarray, labels: np.ndarray, usable: np.ndarray, c: int, top_n: int, audit_n: int) -> dict[str, np.ndarray]:
    """Índices de pontos: candidatos (fora da classe, sim desc.) e auditoria (da classe, sim asc.)."""
    own = sims[:, c]
    outside = np.flatnonzero(usable & (labels != c))
    inside = np.flatnonzero(usable & (labels == c))
    top = outside[np.argsort(-own[outside], kind="stable")[:top_n]] if top_n else outside[:0]
    low = inside[np.argsort(own[inside], kind="stable")[:audit_n]] if audit_n else inside[:0]
    return {"candidatos": top, "auditoria": low}


def runner_up(sims: np.ndarray, rows: np.ndarray, c: int) -> tuple[np.ndarray, np.ndarray]:
    """Segunda classe mais próxima (excluindo c) e a margem sim_c - sim_segunda."""
    other = sims[rows].copy()
    other[:, c] = -np.inf
    second = other.argmax(axis=1)
    return second, sims[rows, c] - other[np.arange(len(rows)), second]


def load_points(meta_dir: Path) -> list[list[Any] | None]:
    """Documento de cada ponto pelas shards do site; pontos ambíguos (vários candidatos) ficam None."""
    points: list[list[Any] | None] = []
    for path in sorted(meta_dir.glob("*.json.gz")):
        for row in json.loads(gzip.decompress(path.read_bytes())):
            points.append(row if isinstance(row, list) else None)
    return points


def point_labels(points: list[list[Any] | None], labels_by_doc: dict[str, str], class_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    index = {name: i for i, name in enumerate(class_names)}
    labels = np.full(len(points), -1, dtype=np.int16)
    usable = np.zeros(len(points), dtype=bool)
    for i, point in enumerate(points):
        if point is None or not point[0]:
            continue
        usable[i] = True
        name = labels_by_doc.get(str(point[0]))
        if name in index:
            labels[i] = index[name]
    return labels, usable


def build_review_queues(
    vectors: np.ndarray, points: list[list[Any] | None], labels_by_doc: dict[str, str], class_names: list[str],
    out_dir: Path, *, top_n: int = 2000, audit_n: int = 500, space: str = "metric-128d",
) -> dict[str, dict[str, int]]:
    """`space` terminando em "-3d" usa distância euclidiana (prévia a partir das coordenadas do site)."""
    if len(vectors) != len(points):
        raise ValueError(f"{len(vectors)} embeddings para {len(points)} pontos: a ordem não confere.")
    labels, usable = point_labels(points, labels_by_doc, class_names)
    cosine = not space.endswith("-3d")
    centroids = class_centroids(vectors, labels, len(class_names), cosine)
    sims = similarities(vectors, centroids, cosine)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict[str, int]] = {}
    for c, name in enumerate(class_names):
        n_labeled = int((labels == c).sum())
        if not n_labeled:
            continue
        queues = rank_class(sims, labels, usable, c, top_n, audit_n)
        payload: dict[str, Any] = {"classe": name, "campos": FIELDS, "rotulados": n_labeled}
        for key, rows in queues.items():
            second, margin = runner_up(sims, rows, c)
            payload[key] = [
                [str(points[r][0]), int(r), round(float(sims[r, c]), 4), class_names[labels[r]] if labels[r] >= 0 else "",
                 class_names[s], round(float(m), 4), *points[r][1:5]]
                for r, s, m in zip(rows.tolist(), second.tolist(), margin.tolist())
            ]
        data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        (out_dir / f"{name}.json.gz").write_bytes(gzip.compress(data, 9, mtime=0))
        summary[name] = {"rotulados": n_labeled, "candidatos": len(payload["candidatos"]), "auditoria": len(payload["auditoria"])}

    index = {
        "gerado": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "espaco": space,
        "classes": [{"nome": name, **counts} for name, counts in summary.items()],
    }
    (out_dir / "index.json").write_text(json.dumps(index, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return summary
