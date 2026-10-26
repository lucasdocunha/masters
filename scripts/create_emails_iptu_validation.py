#!/usr/bin/env python3
"""
Cria pastas de validação visual para EMAILS e IPTU com base nos centroides de embeddings.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pymupdf

BASE_DIR = Path("/home/lucas.ocunha/masters")
DATA_DIR = BASE_DIR / "data"
PDF_DIR = Path("/media/ssd2/datasets/CEAP")
RENDERED_DIR = DATA_DIR / "rendered_pages"
NPZ_FILE = DATA_DIR / "latent_vectors_sample_5pct.npz"
META_FILE = DATA_DIR / "latent_vectors_sample_5pct_metadata.json"
LOOKUP_FILE = DATA_DIR / "pdf_metadata_lookup.json"
HISTORY_FILE = BASE_DIR / "ml_backend" / "models" / "annotations_history.json"
THUMB_LARGE_DIR = Path.home() / ".cache" / "thumbnails" / "large"
DESKTOP_DIR = Path("/home/lucas.ocunha/Desktop")

RENDERED_DIR.mkdir(parents=True, exist_ok=True)
THUMB_LARGE_DIR.mkdir(parents=True, exist_ok=True)

TARGET_CLASSES = {
    "iptu": {
        "title": "IPTU (Guias DAM / Carnês / Imposto Predial Urbano)",
        "desc": "Guias de recolhimento de IPTU e taxas imobiliárias municipais.",
        "dest_dir": DESKTOP_DIR / "validacao_iptu",
        "min_sim": 0.84,
        "n_candidates": 60,
    },
    "emails": {
        "title": "EMAILS (Comprovações Digitais / Invoices por E-mail / Software)",
        "desc": "E-mails de confirmação de pagamento, assinaturas de software (Adobe, Locaweb, etc.).",
        "dest_dir": DESKTOP_DIR / "validacao_emails",
        "min_sim": 0.86,
        "n_candidates": 100,
    }
}

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
    except Exception:
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

def make_confirm_script(class_key: str, class_title: str) -> str:
    return f"""#!/usr/bin/env python3
\"\"\"
Confirma a validação dos documentos na pasta 'validacao_{class_key}'.
- O que você MANTIVER na pasta é registrado como classe '{class_key}'.
- O que você APAGAR é desconsiderado.
Em seguida, retreina o modelo no ML backend automaticamente!
\"\"\"

import json
import subprocess
import sys
import time
from pathlib import Path
import requests

FOLDER = Path(__file__).resolve().parent
MANIFEST_FILE = FOLDER / ".candidatos.json"
HISTORY_FILE = Path("/home/lucas.ocunha/masters/ml_backend/models/annotations_history.json")
ACCEPTED_FILE = Path("/home/lucas.ocunha/masters/data/accepted_{class_key}.json")
ML_BACKEND_URL = "http://localhost:9090"

if not MANIFEST_FILE.exists():
    print(f"Erro: {{MANIFEST_FILE}} não encontrado.")
    sys.exit(1)

with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
    manifest = json.load(f)

current_stems = set(p.stem.split("_")[-1] for p in FOLDER.glob("rank_*.*") if p.suffix in (".png", ".pdf"))
accepted = [m for m in manifest if m["stem"] in current_stems]
rejected = [m for m in manifest if m["stem"] not in current_stems]

print("=== Validação da Classe {class_title} ===")
print(f"Total inicial de candidatos: {{len(manifest):,}}")
print(f"  [OK] Mantidos (Confirmados como {class_key}): {{len(accepted):,}}")
print(f"  [X]  Apagados (Rejeitados/Outros):           {{len(rejected):,}}")

if not accepted:
    print("\\nNenhum arquivo para confirmar.")
    sys.exit(0)

auto_yes = "-y" in sys.argv or "--yes" in sys.argv
if not auto_yes:
    ans = input(f"\\nDeseja gravar os {{len(accepted):,}} documentos como '{class_key}' no modelo e retreinar? [S/n]: ")
    if ans.strip().lower() not in ["", "s", "sim", "y", "yes"]:
        print("Operação cancelada.")
        sys.exit(0)

accepted_stems = [m["stem"] for m in accepted]
ACCEPTED_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(ACCEPTED_FILE, "w", encoding="utf-8") as f:
    json.dump({{"total": len(accepted), "stems": accepted_stems}}, f, indent=2)

try:
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        history = json.load(f)
    for stem in accepted_stems:
        history[stem] = "{class_key}"
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"[OK] Histórico atualizado. Total agora: {{len(history):,}} anotações.")
except PermissionError:
    print("[Aviso] Atualizando histórico via container...")
    subprocess.run([
        "docker", "exec", "ceap-ml-backend", "python3", "-c",
        f\"\"\"
import json
with open('/data/accepted_{class_key}.json') as f:
    stems = set(json.load(f)['stems'])
with open('/app/models/annotations_history.json', 'r') as f:
    h = json.load(f)
for s in stems:
    h[s] = '{class_key}'
with open('/app/models/annotations_history.json', 'w') as f:
    json.dump(h, f, indent=2)
print(f'Histórico atualizado via container: {{len(h)}} anotações.')
\"\"\"
    ], check=True)

# Disparar retreino
print("\\nDisparando retreino no ML backend...")
try:
    subprocess.run(["docker", "restart", "ceap-ml-backend"], check=True)
    time.sleep(3)
    res = requests.post(f"{{ML_BACKEND_URL}}/train", timeout=60)
    if res.status_code == 200:
        data = res.json()
        print(f"[OK SUCESSO] Modelo retreinado: {{data.get('version')}} | Acurácia treino: {{data.get('train_accuracy', 0):.2%}}")
    else:
        print(f"Resposta retreino: {{res.status_code}} - {{res.text}}")
except Exception as e:
    print(f"Aviso no retreino: {{e}}")

print("\\nProcesso concluído com sucesso!")
"""

def main():
    t0 = time.time()
    print("=== CRIAÇÃO DE PASTAS DE VALIDAÇÃO: EMAILS & IPTU (POR CENTRÓIDE) ===")

    print(f"1. Carregando embeddings de {NPZ_FILE}...")
    npz = np.load(NPZ_FILE)
    combined = np.hstack([npz["dino_doc_embeddings"], npz["vlm_doc_embeddings"]]).astype(np.float32)
    norms = np.linalg.norm(combined, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    combined /= norms

    print(f"2. Carregando metadados de {META_FILE}...")
    with open(META_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)
    docs = meta["documents"]
    stem_to_idx = {Path(d["doc_name"]).stem: i for i, d in enumerate(docs)}

    lookup = {}
    if LOOKUP_FILE.exists():
        with open(LOOKUP_FILE, "r", encoding="utf-8") as f:
            lookup = json.load(f)

    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        history = json.load(f)
    labeled_stems = set(history.keys())

    already_selected_stems = set()

    for class_key, class_info in TARGET_CLASSES.items():
        print(f"\n--- Processando Classe: {class_key.upper()} ---")
        dest_dir = class_info["dest_dir"]
        dest_dir.mkdir(parents=True, exist_ok=True)

        for old_f in dest_dir.glob("rank_*.*"):
            old_f.unlink()

        seeds = [s for s, l in history.items() if l == class_key and s in stem_to_idx]
        print(f"  -> {len(seeds)} sementes encontradas no histórico.")

        if not seeds:
            print(f"Erro: Nenhuma semente para {class_key}!")
            continue

        seed_indices = [stem_to_idx[s] for s in seeds]
        centroid = np.mean(combined[seed_indices], axis=0)
        centroid /= np.linalg.norm(centroid)

        sims = np.dot(combined, centroid)
        ranked = np.argsort(-sims)

        candidates = []
        for idx in ranked:
            doc_name = docs[idx].get("doc_name", "")
            stem = Path(doc_name).stem
            if stem in labeled_stems or stem in already_selected_stems:
                continue

            pdf_path = PDF_DIR / doc_name
            if not pdf_path.exists():
                continue

            s = float(sims[idx])
            if s < class_info["min_sim"]:
                break

            m = lookup.get(doc_name, {})
            candidates.append({
                "sim": s,
                "doc_name": doc_name,
                "stem": stem,
                "pdf_path": str(pdf_path),
                "fornecedor": m.get("fornecedor", "N/A"),
                "subcota": m.get("subcota", "N/A"),
                "ano": m.get("ano", "N/A"),
            })
            already_selected_stems.add(stem)

            if len(candidates) >= class_info["n_candidates"]:
                break

        print(f"  -> {len(candidates)} candidatos mais próximos ao centróide selecionados.")
        if candidates:
            print(f"     Similaridade máxima: {candidates[0]['sim']:.4f} | mínima: {candidates[-1]['sim']:.4f}")

        # Renderizar páginas e criar links
        print("  -> Renderizando páginas e criando atalhos...")
        manifest = []
        thumb_tasks = []

        for rank, c in enumerate(candidates, 1):
            stem = c["stem"]
            png_target = RENDERED_DIR / f"{stem}.png"
            if not png_target.exists():
                render_pdf_to_png(Path(c["pdf_path"]), png_target)

            forn_tag = sanitize_filename(c["fornecedor"])
            symlink_name = f"rank_{rank:04d}_sim_{c['sim']:.4f}_[{forn_tag}]_{stem}.png"
            symlink_path = dest_dir / symlink_name

            if symlink_path.exists() or symlink_path.is_symlink():
                symlink_path.unlink()
            symlink_path.symlink_to(png_target)

            c_info = dict(c)
            c_info["rank"] = rank
            c_info["png_target"] = str(png_target)
            c_info["symlink_name"] = symlink_name
            manifest.append(c_info)
            thumb_tasks.append((symlink_path, png_target))

        # Salvar .candidatos.json
        with open(dest_dir / ".candidatos.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        # Salvar confirmar_validacao.py
        script_confirmar = dest_dir / "confirmar_validacao.py"
        with open(script_confirmar, "w", encoding="utf-8") as f:
            f.write(make_confirm_script(class_key, class_info["title"]))
        script_confirmar.chmod(0o755)

        # Salvar LEIAME.txt
        with open(dest_dir / "LEIAME.txt", "w", encoding="utf-8") as f:
            f.write(f"""PASTA DE VALIDAÇÃO VISUAL - CLASSE {class_key.upper()}
============================================================
Título: {class_info['title']}
Descrição: {class_info['desc']}

Total de candidatos mais próximos ao centróide: {len(manifest):,}

COMO USAR:
1. Abra no Thunar:
   thunar {dest_dir} &

2. Mantenha os arquivos que forem de fato da classe '{class_key}'.
3. Apague (tecla Del) arquivos intrusos.
4. Confirme executando o script:
   python3 {dest_dir}/confirmar_validacao.py
""")

        # Gerar miniaturas para o Thunar
        print("  -> Pré-gerando miniaturas para o Thunar...")
        with ThreadPoolExecutor(max_workers=16) as pool:
            futures = [pool.submit(setup_thumbnail, sym, png) for sym, png in thumb_tasks]
            done_cnt = sum(1 for fut in futures if fut.result())
        print(f"     Miniaturas geradas: {done_cnt}/{len(thumb_tasks)}")

    print(f"\n[SUCESSO] Pastas validacao_emails e validacao_iptu criadas no Desktop em {time.time()-t0:.2f}s!")

if __name__ == "__main__":
    main()
