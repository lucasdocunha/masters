"""Sincroniza os rótulos: data/rotulos/ (fonte da verdade, editada pelo site) e a cópia
data/models/annotations_history.json (lida e escrita pelos scripts do cluster).

- Partes -> histórico: `export_history` reescreve o JSON mantendo a ordem das chaves existentes.
- Histórico -> partes: `import_history_changes` aplica só o que mudou no histórico desde uma base
  (diff de 3 vias), então rótulos que vieram do site nunca são revertidos.
Toda mudança aplicada nas partes também entra no log data/labeling/correcoes_site.csv, no mesmo
formato do site, para o espaço latente publicado enxergá-la antes do próximo deploy.
"""

import json
import time
from pathlib import Path
from typing import Any

from dados.labeling.labels_csv import (
    doc_key, load_raw_metadata, normalize_history, read_label_rows, update_rows, write_labels_dir,
)

UNMAPPED = "_unmapped"
LOG_HEADER = "ts,doc_id,ponto,classe_anterior,classe_nova,modo"


def diff_labels(old: dict[str, str], new: dict[str, str]) -> list[dict[str, Any]]:
    """Mudanças de `old` para `new`; `para` None = documento ficou sem rótulo."""
    changes = []
    for doc_id in sorted(set(old) | set(new), key=doc_key):
        before, after = old.get(doc_id), new.get(doc_id)
        if before != after:
            changes.append({"id": doc_id, "de": before, "para": after})
    return changes


def append_log(log_path: Path, changes: list[dict[str, Any]], modo: str, ts: int, points: dict[str, int]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    new = not log_path.exists() or not log_path.read_text(encoding="utf-8").strip()
    with open(log_path, "a", encoding="utf-8") as f:
        if new:
            f.write(LOG_HEADER + "\n")
        for c in changes:
            f.write(f"{ts},{c['id']},{points.get(c['id'], -1)},{c['de'] or UNMAPPED},{c['para'] or UNMAPPED},{modo}\n")


def apply_to_parts(
    parts_dir: Path, log_path: Path, changes: list[dict[str, Any]], *, modo: str = "script",
    ts: int | None = None, points: dict[str, int] | None = None, raw_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Aplica as mudanças nas partes e no log. Devolve as que de fato mudaram algo nas partes."""
    rows = read_label_rows(parts_dir)
    labels = {doc_id: row[-2] for doc_id, row in rows.items()}
    applied = []
    for c in changes:
        current = labels.get(c["id"])
        if current == c["para"]:
            continue
        applied.append({**c, "de": current})
        if c["para"] is None:
            labels.pop(c["id"], None)
        else:
            labels[c["id"]] = c["para"]
    if not applied:
        return []
    new_ids = [c["id"] for c in applied if c["id"] not in rows and c["para"]]
    metadata = load_raw_metadata(raw_dir, new_ids) if raw_dir and raw_dir.exists() and new_ids else {}
    write_labels_dir(parts_dir, update_rows(rows, labels, metadata))
    append_log(log_path, applied, modo, int(time.time() * 1000) if ts is None else ts, points or {})
    return applied


def export_history(labels: dict[str, str], history_path: Path) -> list[dict[str, Any]]:
    """Reescreve o histórico com as classes das partes, mantendo a ordem das chaves que já existiam."""
    old_raw: dict[str, str] = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else {}
    old = normalize_history(old_raw)
    changes = diff_labels(old, labels)
    if not changes:
        return []
    out: dict[str, str] = {}
    for key in old_raw:
        doc_id = str(key).removesuffix(".pdf")
        if doc_id in labels and doc_id not in out:
            out[doc_id] = labels[doc_id]
    for doc_id in sorted(labels, key=doc_key):
        out.setdefault(doc_id, labels[doc_id])
    history_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return changes


def import_history_changes(base: dict[str, str], current: dict[str, str]) -> list[dict[str, Any]]:
    """O que um script do cluster mudou no histórico: diff entre a base (commit) e o arquivo no disco."""
    return diff_labels(normalize_history(base), normalize_history(current))


def points_by_doc(meta_dir: Path) -> dict[str, int]:
    """doc_id -> ponto do espaço latente, pelas shards publicadas (só vínculos exatos)."""
    from dados.labeling.review_queues import load_points

    return {str(p[0]): i for i, p in enumerate(load_points(meta_dir)) if p and p[0]}
