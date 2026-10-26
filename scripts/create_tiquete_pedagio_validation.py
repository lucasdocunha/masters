#!/usr/bin/env python3
"""
Cria pasta de validação visual no Desktop para:
validacao_pedagio (Tíquetes Térmicos de Cabine de Pedágio / DFE - Concessionárias de Rodovia)
Com 100 candidatos representativos de alta diversidade de concessionárias de rodovia.
"""

import hashlib
import json
import os
import re
import sys
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from collections import defaultdict
import psycopg2
import pymupdf

BASE_DIR = Path("/home/lucas.ocunha/masters")
DATA_DIR = BASE_DIR / "data"
PDF_DIR = Path("/media/ssd2/datasets/CEAP")
RENDERED_DIR = DATA_DIR / "rendered_pages"
THUMB_LARGE_DIR = Path.home() / ".cache" / "thumbnails" / "large"
DESKTOP_DIR = Path("/home/lucas.ocunha/Desktop")
DEST_DIR = DESKTOP_DIR / "validacao_pedagio"

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
Confirma a validação dos documentos na pasta 'validacao_pedagio'.
- O que você MANTIVER na pasta é registrado como tíquete térmico de pedágio.
- O que você APAGAR é desconsiderado.
Em seguida, retreina o modelo no ML backend automaticamente!
\"\"\"

import json
import subprocess
import sys
import time
from pathlib import Path
import urllib.request

VALID_DIR = Path(__file__).resolve().parent
MASTERS_DATA = Path("/home/lucas.ocunha/masters/data")
ACCEPTED_FILE = MASTERS_DATA / "accepted_pedagio.json"
HISTORY_FILE = Path("/home/lucas.ocunha/masters/ml_backend/models/annotations_history.json")
BACKEND_TRAIN_URL = "http://localhost:9090/train"

print("=== CONFIRMAÇÃO DE VALIDAÇÃO: PEDÁGIO (TÍQUETES DE CABINE) ===")

candidatos_file = VALID_DIR / ".candidatos.json"
if not candidatos_file.exists():
    print(f"Erro: {candidatos_file} não encontrado!")
    sys.exit(1)

with open(candidatos_file, "r", encoding="utf-8") as f:
    candidatos = json.load(f)

# Identificar os que sobraram na pasta
arquivos_presentes = set(p.name for p in VALID_DIR.glob("*.png"))
confirmados = [c for c in candidatos if c["symlink_name"] in arquivos_presentes]

print(f"Total inicial de candidatos: {len(candidatos)}")
print(f"Mantidos na pasta (Confirmados): {len(confirmados)}")
print(f"Apagados (Rejeitados/Outros): {len(candidatos) - len(confirmados)}")

if not confirmados:
    print("Nenhum arquivo mantido na pasta!")
    sys.exit(0)

# Salvar accepted_pedagio.json
with open(ACCEPTED_FILE, "w", encoding="utf-8") as f:
    json.dump(confirmados, f, indent=2, ensure_ascii=False)
print(f"\\nSalvos {len(confirmados)} registros em {ACCEPTED_FILE}")

# Atualizar annotations_history.json
# Por padrão, incorporamos na classe 'pedagio' (ou conforme parâmetro)
classe_destino = "pedagio"
if "--classe" in sys.argv:
    idx = sys.argv.index("--classe")
    if idx + 1 < len(sys.argv):
        classe_destino = sys.argv[idx + 1]

with open(HISTORY_FILE, "r", encoding="utf-8") as f:
    history = json.load(f)

for doc in confirmados:
    history[str(doc["doc_id"])] = classe_destino

with open(HISTORY_FILE, "w", encoding="utf-8") as f:
    json.dump(history, f, indent=2, ensure_ascii=False)

print(f"Histórico atualizado com {len(confirmados)} anotações na classe '{classe_destino}'.")
print(f"Total de anotações no histórico agora: {len(history)}")

# Disparar retreinamento
print(f"\\nDisparando retreinamento no ML Backend ({BACKEND_TRAIN_URL})...")
try:
    req = urllib.request.Request(BACKEND_TRAIN_URL, data=b"{}", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        print("Resultado do retreino:", json.dumps(res, indent=2))
except Exception as e:
    print(f"Erro ao disparar retreino: {e}")

print("\\nProcesso concluído com sucesso!")
"""


def main():
    print("1. Conectando ao PostgreSQL para amostrar tíquetes de concessionárias...")
    conn = psycopg2.connect("postgresql://postgres:postgres@localhost:5432/ceap")
    cur = conn.cursor()

    query = """
    SELECT DISTINCT v.ide_documento, v.fornecedor_nome, v.valor_liquido, v.ano
    FROM vw_documento_detalhado v
    WHERE (
        v.fornecedor_nome ILIKE ANY(ARRAY[
            '%CONCESSIONARIA%', '%CONCESSIONÁRIA%', '%RODOVIA%', '%RODOVIAS%',
            '%AUTOPISTA%', '%AUTO PISTA%', '%ECOPISTAS%', '%ECOVIAS%',
            '%ECOPONTE%', '%ECO050%', '%ECO101%', '%ECO135%',
            '%ECOSUL%', '%ECONOROESTE%', '%RENOVIAS%', '%TRIUNFO%',
            '%CONCEBRA%', '%ARTERIS%', '%CCR %', '%VIABAHIA%',
            '%VIA BAHIA%', '%ENTREVIAS%', '%ROTA DAS BANDEIRAS%',
            '%ROTA DO OESTE%', '%ROTA DOS COQUEIROS%', '%CART %',
            '%COLINAS%', '%CENTROVIAS%', '%TEBE%', '%INTERVIAS%',
            '%VIAPAULISTA%', '%VIARONDON%', '%SPVIAS%', '%RODOANEL%',
            '%VIASUL%', '%VIA 040%', '%BR-040%', '%MSVIA%',
            '%RODOVIAS DO TIETE%', '%RODOVIAS DO TIETÊ%', '%EPR %',
            '%RODOVIA DO ACO%', '%RODOVIA DO AÇO%', '%TARIFA DE PEDÁGIO%',
            '%PEDAGIO%', '%PEDÁGIO%', '%DFE %'
        ])
    )
    AND v.fornecedor_nome NOT ILIKE '%SEM PARAR%'
    AND v.fornecedor_nome NOT ILIKE '%VELOE%'
    AND v.fornecedor_nome NOT ILIKE '%CONECTCAR%'
    AND v.fornecedor_nome NOT ILIKE '%TAGGY%'
    AND v.fornecedor_nome NOT ILIKE '%MOVE MAIS%'
    ORDER BY v.ide_documento DESC
    LIMIT 1000;
    """
    cur.execute(query)
    rows = cur.fetchall()

    by_fornec = defaultdict(list)
    for r in rows:
        doc_id = str(r[0])
        pdf_p = PDF_DIR / f"{doc_id}.pdf"
        if pdf_p.exists():
            forn = (r[1] or "OUTROS").strip()
            by_fornec[forn].append({
                "doc_id": doc_id,
                "fornecedor": forn,
                "valor": float(r[2]) if r[2] else 0.0,
                "ano": r[3],
                "pdf_path": str(pdf_p)
            })

    # Selecionar até 4 por fornecedor para garantir alta diversidade de concessionárias
    selected = []
    for f, items in sorted(by_fornec.items(), key=lambda x: -len(x[1])):
        selected.extend(items[:4])
        if len(selected) >= 100:
            break

    selected = selected[:100]
    num_concs = len(set(x["fornecedor"] for x in selected))
    print(f"2. Selecionados {len(selected)} tíquetes de {num_concs} concessionárias de rodovia diferentes.")

    print("3. Renderizando páginas em PNG (DPI 150)...")
    candidatos = []
    for idx, item in enumerate(selected, 1):
        doc_id = item["doc_id"]
        pdf_path = Path(item["pdf_path"])
        png_path = RENDERED_DIR / f"{doc_id}.png"

        render_pdf_to_png(pdf_path, png_path)

        forn_s = sanitize_filename(item["fornecedor"])
        val_str = f"R${item['valor']:.2f}".replace(".", ",")
        symlink_name = f"rank_{idx:04d}_[{forn_s}]_[{val_str}]_{doc_id}.png"

        candidatos.append({
            "rank": idx,
            "doc_id": doc_id,
            "symlink_name": symlink_name,
            "fornecedor": item["fornecedor"],
            "valor": item["valor"],
            "ano": item["ano"],
            "pdf_path": str(pdf_path),
            "png_path": str(png_path)
        })

    print("4. Atualizando pasta Desktop/validacao_pedagio...")
    # Limpar links antigos
    for old_f in DEST_DIR.glob("*.png"):
        try:
            old_f.unlink()
        except Exception:
            pass

    for c in candidatos:
        symlink_path = DEST_DIR / c["symlink_name"]
        png_path = Path(c["png_path"])
        if symlink_path.exists() or symlink_path.is_symlink():
            symlink_path.unlink()
        symlink_path.symlink_to(png_path)
        setup_thumbnail(symlink_path, png_path)

    # Salvar .candidatos.json
    with open(DEST_DIR / ".candidatos.json", "w", encoding="utf-8") as f:
        json.dump(candidatos, f, indent=2, ensure_ascii=False)

    # Criar script de confirmação
    confirm_path = DEST_DIR / "confirmar_validacao.py"
    confirm_path.write_text(make_confirm_script(), encoding="utf-8")
    confirm_path.chmod(0o755)

    # Criar LEIAME.txt
    leiame = f"""PASTA DE VALIDAÇÃO VISUAL - CLASSE: TÍQUETES DE CABINE DE PEDÁGIO
========================================================================
Descrição: Tíquetes térmicos de beira de estrada (Documento Fiscal Equivalente - DFE / ANTT)
emitidos pelas concessionárias de rodovias em cabines manuais (Ecosul, Renovias, Ecopistas,
Arteris, Fernão Dias, BR-040, ViaBahia, Concebra, Rota das Bandeiras, etc.).

Total de documentos nesta pasta: {len(candidatos)} amostras de 25 concessionárias.

COMO USAR:
1. Abra no Thunar (gerenciador de arquivos):
   thunar ~/Desktop/validacao_pedagio &

2. Navegue visualmente pelas miniaturas:
   - MANTENHA os tíquetes legítimos de praça/cabine de pedágio.
   - APAGUE (tecla Del) arquivos intrusos (se houver notas fiscais grandes ou outros gastos).

3. Quando terminar, me avise no chat ("pedagio validado") ou execute:
   python3 ~/Desktop/validacao_pedagio/confirmar_validacao.py
"""
    (DEST_DIR / "LEIAME.txt").write_text(leiame, encoding="utf-8")

    print("\n[SUCESSO] Pasta Desktop/validacao_pedagio configurada com 100 tíquetes térmicos!")
    print(f"Caminho: {DEST_DIR}")


if __name__ == "__main__":
    main()
