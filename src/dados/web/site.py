"""Gera os dados estáticos do site de inspeção (web/).

O site é 100% estático: este módulo indexa os artefatos de ``data/`` e produz
JSONs pequenos + miniaturas WebP, para que o navegador nunca precise baixar
PNGs de 187 MB nem bins de 60 MB para navegar.

Saída em ``web/``:
    data/manifest.json   catálogo de arquivos (tipo, categoria, tamanho, data)
    data/envs.json       ambientes de clusterização normalizados + amostras
    data/models.json     artefatos de modelo + contagem de rotulagem por classe
    data/logs.json       logs encontrados e suas últimas linhas
    assets/thumbs/...    miniaturas WebP dos documentos amostrados
    files/...            cópia dos arquivos de texto pequenos (pré-visualização)
"""

from __future__ import annotations

import gzip
import json
import os
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dados.config import DATA_DIR, MODELS_DIR, REPORTS_DIR, ROOT_DIR, WEB_DIR

PREVIEW_LIMIT_BYTES = 1_500_000   # arquivos de texto maiores não são publicados
HTML_COPY_LIMIT_BYTES = 1_000_000
THUMB_WIDTH = 360
LOG_TAIL_LINES = 200

KIND_BY_EXT = {
    ".json": "JSON",
    ".csv": "CSV",
    ".tsv": "CSV",
    ".md": "Markdown",
    ".html": "HTML",
    ".htm": "HTML",
    ".png": "Imagem",
    ".jpg": "Imagem",
    ".jpeg": "Imagem",
    ".webp": "Imagem",
    ".pt": "Modelo",
    ".pth": "Modelo",
    ".joblib": "Modelo",
    ".npz": "Vetores",
    ".npy": "Vetores",
    ".log": "Log",
    ".out": "Log",
    ".err": "Log",
    ".pdf": "PDF",
    ".txt": "Texto",
    ".xml": "XML",
    ".sql": "SQL",
    ".gz": "Comprimido",
    ".zip": "Comprimido",
}
TEXT_KINDS = {"JSON", "CSV", "Markdown", "Texto", "Log", "XML", "SQL", "HTML"}

CATEGORY_BY_TOP = {
    "raw": "Dados brutos",
    "processed": "Processados",
    "models": "Modelos",
    "reports": "Clusterizações e visualizações",
    "evaluation": "Resultados e avaliação",
    "labeling": "Tarefas de rotulagem",
}

ENV_LABELS = {
    "clusters": "Clusterização base (K-Means)",
    "cluster-dino": "DINOv2 visual",
    "cluster-colpali": "ColPali documental",
    "cluster-faiss-dino-k10": "FAISS · DINOv2",
    "cluster-faiss-vlm-k10": "FAISS · VLM",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def _kind(path: Path) -> str:
    return KIND_BY_EXT.get(path.suffix.lower(), "Outro")


def _category(rel: Path) -> str:
    return CATEGORY_BY_TOP.get(rel.parts[0], "Outros") if rel.parts else "Outros"


def _cid_key(cid: str) -> tuple[int, str]:
    return (0, f"{int(cid):06d}") if str(cid).isdigit() else (1, str(cid))


# ---------------------------------------------------------------- catálogo de arquivos
@dataclass
class Artifact:
    rel: str        # caminho relativo a data/ (ex.: "reports/clusters/cluster-dino/summary.json")
    kind: str
    category: str
    size: int
    modified: str
    preview: str | None  # caminho relativo a web/ (ex.: "files/reports/...") quando publicado

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def scan_artifacts(
    data_dir: Path,
    skip_dirs: set[str] = frozenset({"raw/pdfs", "pdfs_selecionados", "rendered_pages", "pgdata"}),
) -> list[Artifact]:
    """Lista arquivos de ``data_dir`` ignorando symlinks duplicados e PNGs (vão para as miniaturas)."""
    out: list[Artifact] = []
    for dirpath, dirnames, filenames in os.walk(data_dir):
        rel_dir = Path(dirpath).relative_to(data_dir).as_posix()
        dirnames[:] = [
            d for d in dirnames
            if not Path(dirpath, d).is_symlink()
            and ((f"{rel_dir}/{d}" if rel_dir != "." else d) not in skip_dirs)
        ]
        for name in filenames:
            p = Path(dirpath, name)
            if p.is_symlink() or p.suffix.lower() in {".png"}:
                continue
            rel = p.relative_to(data_dir)
            st = p.stat()
            out.append(
                Artifact(
                    rel=rel.as_posix(),
                    kind=_kind(p),
                    category=_category(rel),
                    size=st.st_size,
                    modified=datetime.fromtimestamp(st.st_mtime, timezone.utc).date().isoformat(),
                    preview=None,
                )
            )
    return sorted(out, key=lambda a: a.rel)


def publish_previews(artifacts: list[Artifact], data_dir: Path, web_dir: Path) -> int:
    """Copia para web/files os textos pequenos; marca ``preview`` quando publicado."""
    copied = 0
    for a in artifacts:
        is_html = a.kind == "HTML"
        limit = HTML_COPY_LIMIT_BYTES if is_html else PREVIEW_LIMIT_BYTES
        if a.kind not in TEXT_KINDS or a.size > limit:
            continue
        dest = web_dir / "files" / a.rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(data_dir / a.rel, dest)
        a.preview = f"files/{a.rel}"
        copied += 1
    return copied


# ---------------------------------------------------------------- ambientes de clusterização
def _thumb_rel(env_id: str, cid: str, image: str) -> str:
    return f"assets/thumbs/{env_id}/{cid}/{Path(image).stem}.webp"


def _make_thumb(src: Path, dest: Path) -> bool:
    from PIL import Image

    if not src.exists():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im = im.convert("RGB")
        if im.width > THUMB_WIDTH:
            h = round(im.height * THUMB_WIDTH / im.width)
            im = im.resize((THUMB_WIDTH, h), Image.LANCZOS)
        im.save(dest, "WEBP", quality=72, method=4)
    return True


def normalize_env(env_id: str, summary: dict[str, Any]) -> dict[str, Any]:
    """Converte os dois formatos de summary.json para um único esquema."""
    raw_clusters: dict[str, dict[str, Any]] = {}
    if "analysis" in summary or "samples" in summary:  # formato FAISS (k10)
        analysis = summary.get("analysis", {})
        samples = summary.get("samples", {})
        for cid in set(analysis) | set(samples):
            a = analysis.get(cid, {})
            raw_clusters[cid] = {
                "total": a.get("total_docs"),
                "percent": a.get("percentage"),
                "similarity_mean": a.get("similarity_mean"),
                "taxonomy": a.get("inferred_taxonomy"),
                "dominant_subcota": a.get("dominant_subcota"),
                "dominant_pct": a.get("dominant_subcota_pct"),
                "top_subcotas": (a.get("top_subcotas") or [])[:5],
                "samples": samples.get(cid, []),
            }
    else:  # formato K-Means simples
        for cid, v in summary.items():
            if not isinstance(v, dict):
                continue
            raw_clusters[cid] = {
                "total": v.get("total_docs_in_cluster"),
                "percent": v.get("cluster_percentage"),
                "similarity_mean": None,
                "taxonomy": None,
                "dominant_subcota": None,
                "dominant_pct": None,
                "top_subcotas": [],
                "samples": v.get("exported_samples", []),
            }

    clusters = []
    for cid in sorted(raw_clusters, key=_cid_key):
        c = raw_clusters[cid]
        clusters.append({"id": cid, **{k: v for k, v in c.items() if k != "samples"}, "samples": c["samples"]})
    return {
        "id": env_id,
        "label": ENV_LABELS.get(env_id, env_id),
        "k": len(clusters),
        "total_docs": sum((c.get("total") or 0) for c in clusters),
        "clusters": clusters,
    }


def build_envs(clusters_root: Path, web_dir: Path) -> list[dict[str, Any]]:
    envs = []
    for env_dir in sorted(clusters_root.iterdir()):
        if env_dir.is_symlink() or not env_dir.is_dir():
            continue
        summary_path = env_dir / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        env = normalize_env(env_dir.name, summary)
        for c in env["clusters"]:
            for s in c["samples"]:
                image = s.get("image")
                thumb = None
                if image and _make_thumb(env_dir / c["id"] / image, web_dir / _thumb_rel(env["id"], c["id"], image)):
                    thumb = _thumb_rel(env["id"], c["id"], image)
                s["thumb"] = thumb
                s["distance"] = s.pop("dist_to_centroid", None)
                s["pages"] = s.pop("num_pages", None)
                s.pop("image", None)
                s.pop("pdf", None)
        envs.append(env)
    return envs


# ---------------------------------------------------------------- modelos e rotulagem
def build_models(models_dir: Path, history_path: Path) -> dict[str, Any]:
    artifacts = []
    for p in sorted(models_dir.glob("*")):
        if p.is_file():
            artifacts.append({
                "name": p.name,
                "size": p.stat().st_size,
                "kind": _kind(p),
                "preview": None,
            })
    by_class: Counter[str] = Counter()
    total = 0
    if history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))
        by_class.update(history.values())
        total = len(history)
    return {
        "artifacts": artifacts,
        "annotations": {"total": total, "by_class": dict(by_class.most_common())},
    }


LOG_SUFFIXES = {".log", ".out", ".err"}


def find_logs(root: Path, data_dir: Path) -> list[Path]:
    """Logs na raiz (ex.: slurm-*.out), em logs/ e em data/ (sem data/raw, que é grande)."""
    found = [p for p in root.glob("*") if p.is_file() and p.suffix.lower() in LOG_SUFFIXES]
    for base in (root / "logs", data_dir):
        if base.exists():
            for p in base.rglob("*"):
                if p.is_file() and p.suffix.lower() in LOG_SUFFIXES and not p.is_relative_to(data_dir / "raw"):
                    found.append(p)
    return sorted(set(found))


def read_log_tails(root: Path = ROOT_DIR, data_dir: Path = DATA_DIR) -> list[dict[str, Any]]:
    logs = []
    for p in find_logs(root, data_dir):
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        logs.append({
            "path": p.relative_to(root).as_posix(),
            "size": p.stat().st_size,
            "lines": len(lines),
            "tail": lines[-LOG_TAIL_LINES:],
        })
    return logs


def count_dataset_docs(dataset_path: Path) -> int | None:
    if not dataset_path.exists():
        return None
    with gzip.open(dataset_path, "rt", encoding="utf-8") as f:
        return len(json.load(f).get("docs", []))


# ---------------------------------------------------------------- orquestração
def build_site(
    data_dir: Path = DATA_DIR,
    web_dir: Path = WEB_DIR,
    reports_dir: Path = REPORTS_DIR,
    models_dir: Path = MODELS_DIR,
) -> dict[str, Any]:
    data_out = web_dir / "data"
    data_out.mkdir(parents=True, exist_ok=True)

    # limpa saídas anteriores (miniaturas e pré-visualizações) para não deixar lixo
    for stale in [web_dir / "assets" / "thumbs", web_dir / "files"]:
        if stale.exists():
            shutil.rmtree(stale)

    artifacts = scan_artifacts(data_dir)
    published = publish_previews(artifacts, data_dir, web_dir)
    envs = build_envs(reports_dir / "clusters", web_dir)
    models = build_models(models_dir, models_dir / "annotations_history.json")
    logs = read_log_tails(ROOT_DIR, data_dir)
    n_dataset_docs = count_dataset_docs(web_dir / "data" / "plotly_dataset.json.gz")

    total_samples = sum(len(c["samples"]) for e in envs for c in e["clusters"])
    total_thumbs = sum(1 for e in envs for c in e["clusters"] for s in c["samples"] if s.get("thumb"))
    manifest = {
        "generated_at": _now_iso(),
        "stats": {
            "artifacts": len(artifacts),
            "artifacts_published": published,
            "total_bytes": sum(a.size for a in artifacts),
            "environments": len(envs),
            "clusters": sum(e["k"] for e in envs),
            "sampled_docs": total_samples,
            "thumbnails": total_thumbs,
            "labeled_docs": models["annotations"]["total"],
            "dataset_docs": n_dataset_docs,
            "logs": len(logs),
        },
        "artifacts": [a.as_dict() for a in artifacts],
    }
    _write_json(data_out / "manifest.json", manifest)
    _write_json(data_out / "envs.json", {"generated_at": manifest["generated_at"], "envs": envs})
    _write_json(data_out / "models.json", models)
    _write_json(data_out / "logs.json", {"generated_at": manifest["generated_at"], "logs": logs})
    return manifest["stats"]
