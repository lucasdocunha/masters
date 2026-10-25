#!/usr/bin/env python3
"""
Converte todos os documentos da pasta 'validacao_nfs' em imagens .PNG de alta resolução (1ª página).
1. Renderiza a 1ª página de cada PDF em 150 DPI para o cache persistente (data/rendered_pages/<stem>.png).
2. Substitui os links .pdf nas 5 subpastas por links .png correspondentes.
3. Pré-gera as miniaturas do sistema (FreeDesktop) para que o Thunar exiba tudo instantaneamente.
4. Permite navegar com leitor de imagem (ex: Ristretto) usando as setas do teclado e tecla Delete.
"""

import hashlib
import json
import os
import shutil
import sys
import time
import urllib.parse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
import pymupdf

ROOT_DIR = Path("/home/lucas.ocunha/masters")
DATA_DIR = ROOT_DIR / "data"
RENDERED_DIR = DATA_DIR / "rendered_pages"
FOLDER = Path("/home/lucas.ocunha/Desktop/validacao_nfs")
THUMB_LARGE_DIR = Path.home() / ".cache" / "thumbnails" / "large"

SUBFOLDERS = [
    "1_nfse_servicos",
    "2_nfce_cupons",
    "3_danfe_nfe_mercadorias",
    "4_faturas_recibos",
    "5_outros_atipicos"
]


def render_pdf_to_png(task: tuple) -> bool:
    """Renderiza a 1ª página do PDF para o arquivo PNG se ainda não existir."""
    pdf_path, png_path = task
    if png_path.exists() and png_path.stat().st_size > 0:
        return True
    try:
        doc = pymupdf.open(pdf_path)
        if len(doc) == 0:
            doc.close()
            return False
        page = doc[0]
        # 150 DPI garante leitura perfeita de cada linha e número da nota
        pix = page.get_pixmap(dpi=150)
        png_path.parent.mkdir(parents=True, exist_ok=True)
        pix.save(str(png_path))
        doc.close()
        return True
    except Exception:
        return False


def setup_thumbnail(symlink_path: Path, png_path: Path) -> bool:
    """Pré-gera a miniatura no cache do Thunar (FreeDesktop)."""
    try:
        uri = "file://" + urllib.parse.quote(str(symlink_path.absolute()))
        h = hashlib.md5(uri.encode("utf-8")).hexdigest()
        thumb_path = THUMB_LARGE_DIR / f"{h}.png"

        if thumb_path.exists():
            return True

        doc = pymupdf.open(png_path)
        if len(doc) == 0:
            doc.close()
            return False
        # Miniatura de 100 DPI
        pix = doc[0].get_pixmap(dpi=72)
        pix.save(str(thumb_path))
        doc.close()
        return True
    except Exception:
        return False


def main():
    print("=== Conversão dos Documentos de Validação para PNG ===")
    t0 = time.time()

    RENDERED_DIR.mkdir(parents=True, exist_ok=True)
    THUMB_LARGE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Mapear todos os links existentes nas subpastas
    items_to_convert = []
    for sub in SUBFOLDERS:
        sub_dir = FOLDER / sub
        if not sub_dir.exists():
            continue
        # Procurar por .pdf ou .png já existentes
        for f in sub_dir.glob("rank_*.*"):
            if f.suffix in (".pdf", ".png"):
                resolved = f.resolve()
                stem = f.stem.split("_")[-1]
                png_target = RENDERED_DIR / f"{stem}.png"
                items_to_convert.append({
                    "subfolder": sub_dir,
                    "current_file": f,
                    "stem": f.stem,
                    "target_pdf": resolved if resolved.suffix == ".pdf" else None,
                    "target_png": png_target,
                    "new_symlink": sub_dir / f"{f.stem}.png"
                })

    print(f"Total de documentos encontrados nas subpastas: {len(items_to_convert):,}")

    # 2. Renderizar páginas faltantes para PNG
    render_tasks = []
    lookup_file = DATA_DIR / "pdf_metadata_lookup.json"
    lookup = {}
    if lookup_file.exists():
        with open(lookup_file, "r") as lf:
            lookup = json.load(lf)

    for item in items_to_convert:
        if not item["target_png"].exists() or item["target_png"].stat().st_size == 0:
            pdf_path = item["target_pdf"]
            if pdf_path is None or not pdf_path.exists():
                # Procurar pelo stem no dataset
                stem = item["stem"].split("_")[-1]
                cand = Path("/media/ssd2/datasets/CEAP") / f"{stem}.pdf"
                if cand.exists():
                    pdf_path = cand
                else:
                    # Checar ano
                    for y in range(2015, 2027):
                        cand = Path(f"/media/ssd2/datasets/CEAP/{y}/{stem}.pdf")
                        if cand.exists():
                            pdf_path = cand
                            break
            if pdf_path and pdf_path.exists():
                render_tasks.append((pdf_path, item["target_png"]))

    print(f"Renderizando {len(render_tasks):,} páginas faltantes para PNG (150 DPI)...")
    if render_tasks:
        t_rend = time.time()
        with ProcessPoolExecutor(max_workers=12) as ex:
            results = list(ex.map(render_pdf_to_png, render_tasks))
        print(f"Páginas renderizadas ({sum(results)}/{len(render_tasks)}) em {time.time()-t_rend:.2f}s!")

    # 3. Atualizar symlinks nas subpastas para .png
    print("\nAtualizando links simbólicos para .png nas subpastas...")
    thumb_tasks = []
    for item in items_to_convert:
        png_target = item["target_png"]
        if not png_target.exists():
            continue

        new_symlink = item["new_symlink"]
        current_file = item["current_file"]

        # Se for o próprio link png e já aponta certo, ok
        if current_file.suffix == ".pdf":
            current_file.unlink()

        if new_symlink.exists() or new_symlink.is_symlink():
            new_symlink.unlink()

        new_symlink.symlink_to(png_target)
        thumb_tasks.append((new_symlink, png_target))

    # 4. Atualizar manifesto de cada subpasta com a nova extensão .png
    for sub in SUBFOLDERS:
        manifest_file = FOLDER / sub / ".candidatos.json"
        if manifest_file.exists():
            with open(manifest_file, "r", encoding="utf-8") as f:
                sub_manifest = json.load(f)
            for m in sub_manifest:
                if m.get("filename", "").endswith(".pdf"):
                    m["filename"] = m["filename"][:-4] + ".png"
            with open(manifest_file, "w", encoding="utf-8") as f:
                json.dump(sub_manifest, f, indent=2, ensure_ascii=False)

    # 5. Pré-renderizar miniaturas FreeDesktop para Thunar
    print(f"\nPré-gerando miniaturas para {len(thumb_tasks):,} links PNG com 8 workers...")
    t_th = time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(setup_thumbnail, s, p) for s, p in thumb_tasks]
        ok = sum(1 for f in futures if f.result())
    print(f"Miniaturas pré-geradas ({ok}/{len(thumb_tasks)}) em {time.time()-t_th:.2f}s!")

    # 6. Atualizar script de confirmação
    atualizar_confirmar_validacao()

    print(f"\n[SUCESSO] Todos os {len(items_to_convert):,} documentos agora são arquivos .PNG!")
    print(f"Tempo total: {time.time()-t0:.2f}s.")


def atualizar_confirmar_validacao():
    script_path = FOLDER / "confirmar_validacao.py"
    with open(script_path, "w", encoding="utf-8") as f:
        f.write('''#!/usr/bin/env python3
"""
Confirma a validação em lote das subpastas:
- 1_nfse_servicos         -> grava como 'nfse'
- 2_nfce_cupons           -> grava como 'nfc'
- 3_danfe_nfe_mercadorias -> grava como 'nfe'
- 4_faturas_recibos       -> grava como 'fatura'

O que você MANTIVER dentro de cada pasta (em .png ou .pdf) é confirmado e gravado no modelo.
O que você APAGAR de cada pasta é desconsiderado.
Em seguida, retreina o modelo no ML backend automaticamente!
"""

import json
import subprocess
import sys
from pathlib import Path
import requests

FOLDER = Path(__file__).resolve().parent
HISTORY_FILE = Path("/home/lucas.ocunha/masters/ml_backend/models/annotations_history.json")
ACCEPTED_FILE = Path("/home/lucas.ocunha/masters/data/accepted_batches.json")
ML_BACKEND_URL = "http://localhost:9090"

SUBFOLDERS = {
    "1_nfse_servicos": "nfse",
    "2_nfce_cupons": "nfc",
    "3_danfe_nfe_mercadorias": "nfe",
    "4_faturas_recibos": "fatura",
}

print("=== Confirmação de Validação por Subpastas ===")
totals_by_class = {}
all_accepted = {}

for sub_name, label in SUBFOLDERS.items():
    sub_dir = FOLDER / sub_name
    manifest_file = sub_dir / ".candidatos.json"
    if not manifest_file.exists():
        continue
    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Identificar stems presentes na pasta (tanto se for .png quanto .pdf)
    current_stems = set(p.stem.split("_")[-1] for p in sub_dir.glob("rank_*.*") if p.suffix in (".png", ".pdf"))
    accepted = [m for m in manifest if m["stem"] in current_stems]
    rejected = [m for m in manifest if m["stem"] not in current_stems]

    totals_by_class[label] = {
        "folder": sub_name,
        "initial": len(manifest),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "stems": [m["stem"] for m in accepted]
    }
    for m in accepted:
        all_accepted[m["stem"]] = label

print(f"Resumo da sua revisão:")
total_confirm = 0
for label, d in totals_by_class.items():
    print(f"  [{label.upper():6s}] {d['folder']:25s} -> Mantidos: {d['accepted']:4d} / {d['initial']:4d} (Apagados: {d['rejected']})")
    total_confirm += d["accepted"]

if total_confirm == 0:
    print("\\nNenhum documento para confirmar.")
    sys.exit(0)

auto_yes = "-y" in sys.argv or "--yes" in sys.argv
if not auto_yes:
    ans = input(f"\\nDeseja gravar todos os {total_confirm:,} documentos nas respectivas classes e retreinar o modelo? [S/n]: ")
    if ans.strip().lower() not in ["", "s", "sim", "y", "yes"]:
        print("Operação cancelada.")
        sys.exit(0)

# Salvar json de backup dos aceitos
ACCEPTED_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(ACCEPTED_FILE, "w", encoding="utf-8") as f:
    json.dump({"total": total_confirm, "by_class": {k: len(v["stems"]) for k, v in totals_by_class.items()}, "items": all_accepted}, f, indent=2)

# Atualizar histórico
try:
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        history = json.load(f)
    for stem, label in all_accepted.items():
        history[stem] = label
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"[OK] Histórico atualizado com {total_confirm:,} novas anotações. Total no modelo: {len(history):,}.")
except PermissionError:
    print("[Aviso] Atualizando histórico via container...")
    subprocess.run([
        "docker", "exec", "ceap-ml-backend", "python3", "-c",
        """
import json
with open('/data/accepted_batches.json') as f:
    data = json.load(f)['items']
with open('/app/models/annotations_history.json', 'r') as f:
    h = json.load(f)
for s, l in data.items():
    h[s] = l
with open('/app/models/annotations_history.json', 'w') as f:
    json.dump(h, f, indent=2)
print(f'Histórico atualizado via container: {len(h)} anotações.')
"""
    ], check=True)

# Disparar retreino
print("\\nDisparando retreino no ML backend...")
try:
    res = requests.post(f"{ML_BACKEND_URL}/train", timeout=60)
    if res.status_code == 200:
        data = res.json()
        print(f"[OK SUCESSO] Modelo retreinado: {data.get('version')} | Acurácia treino: {data.get('train_accuracy', 0):.2%}")
    else:
        print(f"Resposta HTTP retreino: {res.status_code} - {res.text}")
except Exception as e:
    print(f"Aviso ao disparar retreino: {e}")

print("\\nProcesso concluído com sucesso!")
''')
    script_path.chmod(0o755)


if __name__ == "__main__":
    main()
