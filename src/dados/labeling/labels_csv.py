"""Rótulos versionados: data/rotulos/parte_NNNN.csv (o CSV dos documentos rotulados, dividido em partes).

É a fonte da verdade dos rótulos. O site (web/api/labels.js) edita as partes direto no GitHub, cuja API
recusa arquivos acima de ~40 MB; por isso o CSV é dividido por faixas de 25.000 doc_id (~1 MB por parte),
e cada salvamento só reenvia as partes que mudaram. O formato é um contrato com o JavaScript:
- a parte de um documento é parte_{doc_id // 25000:04d}.csv, e toda parte começa com o cabeçalho;
- uma linha por documento, ordenada por doc_id numérico, sem quebras de linha dentro dos campos;
- `classe_mapeada` e `status` são sempre os dois últimos campos e nunca precisam de aspas,
  de modo que o site troca a classe editando só o fim da linha.
"""

import csv
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

HEADERS = ["doc_id", "link_documento", "metadados_json", "ano", "classe_mapeada", "status"]
STATUS = "rotulado"
SPAN = 25_000
META_FIELDS = {
    "fornecedor": "fornecedor", "cnpjCPF": "cnpjCPF", "valorDocumento": "valorDocumento", "valorLiquido": "valorLiquido",
    "subcota": "descricao", "nomeParlamentar": "nomeParlamentar", "siglaUF": "siglaUF", "siglaPartido": "siglaPartido",
    "dataEmissao": "dataEmissao",
}


def clean(value: Any) -> str:
    return "" if value is None else " ".join(str(value).split())


def doc_key(doc_id: str) -> tuple[int, str]:
    return (int(doc_id), "") if doc_id.isdigit() else (1 << 62, doc_id)


def normalize_history(history: dict[str, str]) -> dict[str, str]:
    """Chaves do histórico de anotações podem vir com `.pdf`; aqui ficam só o id."""
    return {str(k).removesuffix(".pdf"): v for k, v in history.items() if v}


def raw_metadata(items: Iterable[dict[str, Any]], wanted: set[str]) -> Iterator[tuple[str, dict[str, Any]]]:
    """Metadados do primeiro registro de cada documento pedido num Ano-*.json."""
    for item in items:
        doc_id = str(item.get("idDocumento") or "")
        if doc_id not in wanted:
            continue
        wanted.discard(doc_id)
        yield doc_id, {
            "link": clean(item.get("urlDocumento")),
            "ano": item.get("ano") or "",
            "meta": {name: clean(item.get(key)) for name, key in META_FIELDS.items()},
        }


def load_raw_metadata(raw_dir: Path, ids: Iterable[str]) -> dict[str, dict[str, Any]]:
    wanted = set(ids)
    found: dict[str, dict[str, Any]] = {}
    for path in sorted(raw_dir.glob("Ano-*.json")):
        if not wanted:
            break
        data = json.loads(path.read_text(encoding="utf-8"))
        items = data.get("dados", []) if isinstance(data, dict) else data
        found.update(raw_metadata(items, wanted))
    return found


def make_row(doc_id: str, label: str, info: dict[str, Any] | None) -> list[str]:
    info = info or {}
    return [
        doc_id,
        info.get("link", ""),
        json.dumps(info.get("meta", {}), ensure_ascii=False, separators=(",", ":")),
        str(info.get("ano", "")),
        label,
        STATUS,
    ]


def build_rows(labels: dict[str, str], metadata: dict[str, dict[str, Any]]) -> list[list[str]]:
    return [make_row(doc_id, labels[doc_id], metadata.get(doc_id)) for doc_id in sorted(labels, key=doc_key)]


def write_labels_csv(path: Path, rows: Iterable[list[str]]) -> int:
    count = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(HEADERS)
        for row in rows:
            if any(c in row[-1] + row[-2] for c in ',"\n'):
                raise ValueError(f"classe/status inválidos para o CSV: {row[0]}")
            writer.writerow(row)
            count += 1
    return count


def part_name(doc_id: str) -> str:
    return f"parte_{int(doc_id) // SPAN:04d}.csv"


def write_labels_dir(directory: Path, rows: Iterable[list[str]]) -> int:
    """Grava as partes e apaga as que ficaram vazias. Devolve o total de linhas."""
    parts: dict[str, list[list[str]]] = {}
    for row in rows:
        parts.setdefault(part_name(row[0]), []).append(row)
    directory.mkdir(parents=True, exist_ok=True)
    for old in directory.glob("parte_*.csv"):
        if old.name not in parts:
            old.unlink()
    return sum(write_labels_csv(directory / name, part) for name, part in sorted(parts.items()))


def _files(path: Path) -> list[Path]:
    return sorted(path.glob("parte_*.csv")) if path.is_dir() else [path]


def read_labels_csv(path: Path) -> dict[str, str]:
    """Classe por doc_id, de um diretório de partes ou de um CSV único."""
    labels: dict[str, str] = {}
    for file in _files(path):
        with open(file, newline="", encoding="utf-8") as f:
            labels.update((row["doc_id"], row["classe_mapeada"]) for row in csv.DictReader(f))
    return labels


def read_label_rows(path: Path) -> dict[str, list[str]]:
    rows: dict[str, list[str]] = {}
    for file in _files(path):
        with open(file, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            rows.update((row[0], row) for row in reader)
    return rows


def update_rows(rows: dict[str, list[str]], labels: dict[str, str], metadata: dict[str, dict[str, Any]]) -> list[list[str]]:
    """Mantém as linhas existentes (e seus metadados), troca a classe e acrescenta/remova documentos."""
    out = []
    for doc_id in sorted(labels, key=doc_key):
        row = rows.get(doc_id)
        if row:
            out.append(row[:-2] + [labels[doc_id], STATUS])
        else:
            out.append(make_row(doc_id, labels[doc_id], metadata.get(doc_id)))
    return out
