#!/usr/bin/env python3
"""
Cria pasta de validação visual no Desktop para:
validacao_taxi (Recibos Manuais de Táxi / Talão de Praça / Cooperativas)
Com 100 candidatos representativos de alta diversidade de cooperativas e anos.
"""

import hashlib
import json
import os
import re
import sys
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pymupdf

BASE_DIR = Path("/home/lucas.ocunha/masters")
DATA_DIR = BASE_DIR / "data"
PDF_DIR = Path("/media/ssd2/datasets/CEAP")
RENDERED_DIR = DATA_DIR / "rendered_pages"
LOOKUP_FILE = DATA_DIR / "pdf_metadata_lookup.json"
THUMB_LARGE_DIR = Path.home() / ".cache" / "thumbnails" / "large"
DESKTOP_DIR = Path("/home/lucas.ocunha/Desktop")
DEST_DIR = DESKTOP_DIR / "validacao_taxi"

RENDERED_DIR.mkdir(parents=True, exist_ok=True)
THUMB_LARGE_DIR.mkdir(parents=True, exist_ok=True)
DEST_DIR.mkdir(parents=True, exist_ok=True)


def sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[^\w\s-]", "", name).strip()
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned[:25]


def render_pdf_to_png(pdf_path: Path, png_path: Path) -> bool:
    if png_path.exists() and png_path.stat().st_size > 0:
        return True
    try:
        doc = pymupdf.open(str(pdf_path))
        if len(doc) == 0:
            doc.close()
            return False
        page = doc[0]
        pix = page.get_pixmap(dpi=150)
        png_path.parent.mkdir(parents=True, exist_ok=True)
        pix.save(str(png_path))
        doc.close()
        return True
    except Exception as e:
        print(f"Erro renderizando {pdf_path}: {e}")
        return False


def setup_thumbnail(symlink_path: Path, png_path: Path) -> bool:
    try:
        uri = "file://" + urllib.parse.quote(str(symlink_path.absolute()))
        h = hashlib.md5(uri.encode("utf-8")).hexdigest()
        thumb_path = THUMB_LARGE_DIR / f"{h}.png"
        if thumb_path.exists():
            return True
        doc = pymupdf.open(str(png_path))
        if len(doc) == 0:
            doc.close()
            return False
        pix = doc[0].get_pixmap(dpi=72)
        pix.save(str(thumb_path))
        doc.close()
        return True
    except Exception:
        return False


def make_confirm_script() -> str:
    return """#!/usr/bin/env python3
\"\"\"
Confirma a validação dos documentos na pasta 'validacao_taxi'.
- O que você MANTIVER na pasta é registrado como classe 'taxi'.
- O que você APAGAR é desconsiderado.
Em seguida, retreina o modelo no ML backend automaticamente!
\"\"\"

import json
import subprocess
import sys
from pathlib import Path
import urllib.request

VALID_DIR = Path(__file__).resolve().parent
MASTERS_DATA = Path("/home/lucas.ocunha/masters/data")
ACCEPTED_FILE = MASTERS_DATA / "accepted_taxi.json"
LOOKUP_FILE = MASTERS_DATA / "pdf_metadata_lookup.json"
BACKEND_TRAIN_URL = "http://localhost:9090/train"

print(f"=== CONFIRMAÇÃO DE VALIDAÇÃO: TAXI ===")

png_files = sorted(VALID_DIR.glob("rank_*.png"))
if not png_files:
    print("Nenhum arquivo PNG encontrado na pasta!")
    sys.exit(1)

confirmed_ids = []
for p in png_files:
    # Extrair doc_id do final do nome: rank_XXXX_[FORN]_docid.png
    parts = p.stem.split("_")
    doc_id = parts[-1]
    confirmed_ids.append(doc_id)

print(f"Total de documentos mantidos na pasta: {len(confirmed_ids):,}")

# Carregar metadados
meta = {}
if LOOKUP_FILE.exists():
    with open(LOOKUP_FILE, "r") as f:
        meta = json.load(f)

accepted_records = []
for d_id in confirmed_ids:
    pdf_name = f"{d_id}.pdf"
    info = meta.get(pdf_name, {})
    accepted_records.append({
        "doc_id": d_id,
        "doc_name": pdf_name,
        "classe": "taxi",
        "fornecedor": info.get("fornecedor", "N/A"),
        "ano": info.get("ano"),
    })

with open(ACCEPTED_FILE, "w", encoding="utf-8") as f:
    json.dump(accepted_records, f, indent=2, ensure_ascii=False)

print(f"Registros salvos em: {ACCEPTED_FILE}")

# Enviar dados ao ML Backend
print("\\nAtualizando histórico de anotações no ML Backend...")
HISTORY_FILE = Path("/home/lucas.ocunha/masters/ml_backend/models/annotations_history.json")
if HISTORY_FILE.exists():
    with open(HISTORY_FILE, "r") as f:
        history = json.load(f)
else:
    history = {}

added = 0
for r in accepted_records:
    pdf_name = r["doc_name"]
    if pdf_name not in history or history[pdf_name] != "taxi":
        history[pdf_name] = "taxi"
        added += 1

with open(HISTORY_FILE, "w") as f:
    json.dump(history, f, indent=2)

print(f"Adicionadas {added} anotações com classe 'taxi' no histórico!")

# Disparar retreinamento
print(f"Disparando retreinamento no ML Backend ({BACKEND_TRAIN_URL})...")
try:
    req = urllib.request.Request(BACKEND_TRAIN_URL, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        res = json.loads(resp.read().decode())
        print(f"Sucesso! Modelo retreinado: {res}")
except Exception as e:
    print(f"Aviso ao chamar retreino: {e}")
    print("Você pode retreinar manualmente rodando curl -X POST http://localhost:9090/train")

print("\\n[OK] Validação de Táxi concluída com sucesso!")
"""


def main():
    print("=== SELEÇÃO DE 100 DOCUMENTOS REPRESENTATIVOS DE TÁXI ===")
    with open(LOOKUP_FILE, "r") as f:
        lookup = json.load(f)

    # Coletar candidatos de táxi
    candidates_by_supplier = {}

    for pdf_name, meta in lookup.items():
        doc_id = Path(pdf_name).stem
        forn = meta.get("fornecedor", "").strip()
        subcota = meta.get("subcota", "").strip().upper()
        forn_up = forn.upper()

        if not any(k in forn_up for k in ["TAXI", "TÁXI", "SINPETAXI"]):
            continue

        pdf_path = PDF_DIR / pdf_name
        if not pdf_path.exists():
            continue

        candidates_by_supplier.setdefault(forn, []).append({
            "doc_id": doc_id,
            "pdf_name": pdf_name,
            "fornecedor": forn,
            "ano": meta.get("ano", 2024),
            "pdf_path": pdf_path,
        })

    print(f"Fornecedores de táxi encontrados com PDF no SSD: {len(candidates_by_supplier):,}")

    # Selecionar com balanceamento e diversidade
    selected = []
    # Ordenar fornecedores por volume
    sorted_suppliers = sorted(candidates_by_supplier.keys(), key=lambda k: len(candidates_by_supplier[k]), reverse=True)

    # Pegar até 4 por fornecedor dos mais frequentes, e 1-2 dos médios
    for s in sorted_suppliers:
        docs = candidates_by_supplier[s]
        # Diversificar por ano
        docs_by_year = {}
        for d in docs:
            docs_by_year.setdefault(d["ano"], []).append(d)
        
        take = min(4, len(docs))
        picked_for_supplier = []
        for y in sorted(docs_by_year.keys(), reverse=True):
            picked_for_supplier.append(docs_by_year[y][0])
            if len(picked_for_supplier) >= take:
                break
        selected.extend(picked_for_supplier)
        if len(selected) >= 100:
            break

    selected = selected[:100]
    print(f"Total de candidatos selecionados: {len(selected)}")

    # Limpar pasta de destino
    for existing in DEST_DIR.glob("*"):
        if existing.is_file() or existing.is_symlink():
            existing.unlink()

    # Renderizar e criar symlinks
    print("\nRenderizando PDFs para PNG e gerando miniaturas do Thunar...")
    rendered_count = 0

    def process_item(idx_item):
        idx, item = idx_item
        doc_id = item["doc_id"]
        forn_clean = sanitize_filename(item["fornecedor"])
        png_cache = RENDERED_DIR / f"{doc_id}.png"

        # 1. Renderizar
        ok = render_pdf_to_png(item["pdf_path"], png_cache)
        if not ok:
            return False

        # 2. Criar Link amigável no Desktop
        symlink_name = f"rank_{idx+1:04d}_[{forn_clean}]_{doc_id}.png"
        symlink_path = DEST_DIR / symlink_name
        if symlink_path.exists() or symlink_path.is_symlink():
            symlink_path.unlink()
        symlink_path.symlink_to(png_cache)

        # 3. Miniatura do Thunar
        setup_thumbnail(symlink_path, png_cache)
        return True

    with ThreadPoolExecutor(max_workers=12) as executor:
        results = list(executor.map(process_item, enumerate(selected)))
        rendered_count = sum(1 for r in results if r)

    # Escrever scripts auxiliares
    confirm_script = DEST_DIR / "confirmar_validacao.py"
    with open(confirm_script, "w", encoding="utf-8") as f:
        f.write(make_confirm_script())
    confirm_script.chmod(0o755)

    leiame = DEST_DIR / "LEIAME.txt"
    with open(leiame, "w", encoding="utf-8") as f:
        f.write(
            "=== VALIDAÇÃO VISUAL: RECIBOS MANUAIS DE TÁXI (TALÃO DE PRAÇA) ===\n\n"
            f"Total de documentos preparados: {rendered_count}\n"
            "Representam cooperativas e talões de táxi de todo o Brasil (SINPETAXI, TÁXI LEGAL, COOPERTAXI, etc.).\n\n"
            "INSTRUÇÕES:\n"
            "1. Abra esta pasta no gerenciador de arquivos (as miniaturas já estão pré-geradas).\n"
            "2. Apague qualquer imagem que NÃO seja um recibo/talão de táxi.\n"
            "3. Quando terminar, execute no terminal:\n"
            "   python3 confirmar_validacao.py\n\n"
            "Isso salvará a classe 'taxi' e retreinará o modelo no ML Backend!\n"
        )

    print(f"\n[SUCESSO] {rendered_count} documentos de táxi preparados na pasta:")
    print(f" -> {DEST_DIR}")


if __name__ == "__main__":
    main()
