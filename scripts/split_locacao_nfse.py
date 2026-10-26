#!/usr/bin/env python3
"""
Separa a pasta Desktop/validacao_locacao_veiculos em duas subpastas:
1. nfse_identificadas (62 documentos que são NFS-e municipais)
2. locacao_pura_contratos_faturas (35 documentos que são contratos/faturas de locação pura)
"""

import hashlib
import json
import os
import re
import shutil
import sys
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pymupdf
from rapidocr_onnxruntime import RapidOCR

BASE_DIR = Path("/home/lucas.ocunha/Desktop/validacao_locacao_veiculos")
DIR_NFSE = BASE_DIR / "nfse_identificadas"
DIR_LOCACAO = BASE_DIR / "locacao_pura_contratos_faturas"
THUMB_DIR = Path(os.path.expanduser("~/.cache/thumbnails/large"))

DIR_NFSE.mkdir(parents=True, exist_ok=True)
DIR_LOCACAO.mkdir(parents=True, exist_ok=True)

with open(BASE_DIR / ".candidatos.json") as f:
    manifest = json.load(f)

manifest_by_stem = {m["stem"]: m for m in manifest}
current_files = list(BASE_DIR.glob("rank_*.*"))

ocr = RapidOCR()

def setup_thumb(symlink_path: Path, png_target: Path):
    try:
        uri = "file://" + urllib.parse.quote(str(symlink_path.absolute()))
        h = hashlib.md5(uri.encode("utf-8")).hexdigest()
        thumb_path = THUMB_DIR / f"{h}.png"
        if thumb_path.exists():
            return True
        doc = pymupdf.open(str(png_target))
        if len(doc) == 0:
            doc.close()
            return False
        pix = doc[0].get_pixmap(dpi=72)
        pix.save(str(thumb_path))
        doc.close()
        return True
    except Exception:
        return False

manifest_nfse = []
manifest_locacao = []
thumb_tasks = []

print("Separando arquivos em subpastas...")

for f in current_files:
    stem = f.stem.split("_")[-1]
    m = manifest_by_stem.get(stem)
    if not m:
        continue
    
    # Check text
    doc = pymupdf.open(m["pdf_path"])
    txt = " ".join(page.get_text() for page in doc).lower()
    doc.close()
    
    png_path = Path(m["png_target"])
    if len(txt.strip()) < 50 and png_path.exists():
        res, _ = ocr(str(png_path))
        if res:
            txt = txt + " " + " ".join([line[1] for line in res]).lower()
            
    is_nfse = any(k in txt for k in [
        "nota fiscal de servico", "nota fiscal de serviço", "nfs-e", "nfse", 
        "prefeitura municipal", "secretaria municipal da fazenda", "issqn", "tributação municipal"
    ])
    
    target_dir = DIR_NFSE if is_nfse else DIR_LOCACAO
    new_symlink = target_dir / f.name
    
    if new_symlink.exists() or new_symlink.is_symlink():
        new_symlink.unlink()
    new_symlink.symlink_to(png_path)
    
    # Remove from root to keep root organized
    f.unlink()
    
    thumb_tasks.append((new_symlink, png_path))
    if is_nfse:
        manifest_nfse.append(m)
    else:
        manifest_locacao.append(m)

# Salvar manifestos
with open(DIR_NFSE / ".candidatos.json", "w", encoding="utf-8") as f:
    json.dump(manifest_nfse, f, indent=2, ensure_ascii=False)

with open(DIR_LOCACAO / ".candidatos.json", "w", encoding="utf-8") as f:
    json.dump(manifest_locacao, f, indent=2, ensure_ascii=False)

# Scripts de confirmação
script_nfse = f"""#!/usr/bin/env python3
import json, subprocess, sys, time
from pathlib import Path
import requests

FOLDER = Path(__file__).resolve().parent
with open(FOLDER / ".candidatos.json") as f:
    manifest = json.load(f)
current_stems = set(p.stem.split("_")[-1] for p in FOLDER.glob("rank_*.*") if p.suffix in (".png", ".pdf"))
accepted = [m["stem"] for m in manifest if m["stem"] in current_stems]

print(f"Confirmando {{len(accepted)}} documentos como NFSE...")
with open("/home/lucas.ocunha/masters/data/accepted_locacao_as_nfse.json", "w") as f:
    json.dump({{"total": len(accepted), "stems": accepted}}, f, indent=2)

h_file = Path("/home/lucas.ocunha/masters/ml_backend/models/annotations_history.json")
with open(h_file) as f:
    h = json.load(f)
for s in accepted:
    h[s] = "nfse"
with open(h_file, "w") as f:
    json.dump(h, f, indent=2)

subprocess.run(["docker", "restart", "ceap-ml-backend"], check=True)
time.sleep(3)
requests.post("http://localhost:9090/train")
print("[OK] Documentos gravados como NFSE e modelo retreinado!")
"""
with open(DIR_NFSE / "confirmar_validacao.py", "w") as f:
    f.write(script_nfse)
(DIR_NFSE / "confirmar_validacao.py").chmod(0o755)

script_locacao = f"""#!/usr/bin/env python3
import json, subprocess, sys, time
from pathlib import Path
import requests

FOLDER = Path(__file__).resolve().parent
with open(FOLDER / ".candidatos.json") as f:
    manifest = json.load(f)
current_stems = set(p.stem.split("_")[-1] for p in FOLDER.glob("rank_*.*") if p.suffix in (".png", ".pdf"))
accepted = [m["stem"] for m in manifest if m["stem"] in current_stems]

print(f"Confirmando {{len(accepted)}} documentos como LOCACAO_VEICULOS...")
with open("/home/lucas.ocunha/masters/data/accepted_locacao_veiculos.json", "w") as f:
    json.dump({{"total": len(accepted), "stems": accepted}}, f, indent=2)

h_file = Path("/home/lucas.ocunha/masters/ml_backend/models/annotations_history.json")
with open(h_file) as f:
    h = json.load(f)
for s in accepted:
    h[s] = "locacao_veiculos"
with open(h_file, "w") as f:
    json.dump(h, f, indent=2)

subprocess.run(["docker", "restart", "ceap-ml-backend"], check=True)
time.sleep(3)
requests.post("http://localhost:9090/train")
print("[OK] Documentos gravados como LOCACAO_VEICULOS e modelo retreinado!")
"""
with open(DIR_LOCACAO / "confirmar_validacao.py", "w") as f:
    f.write(script_locacao)
(DIR_LOCACAO / "confirmar_validacao.py").chmod(0o755)

# LEIAME
with open(DIR_NFSE / "LEIAME.txt", "w") as f:
    f.write(f"SUBPASTA: NFS-e Identificadas ({len(manifest_nfse)} documentos)\\nContem notas fiscais de servico municipais emitidas por locadoras regionais.\\nExecute ./confirmar_validacao.py para registrar na classe 'nfse'.\\n")

with open(DIR_LOCACAO / "LEIAME.txt", "w") as f:
    f.write(f"SUBPASTA: Locacao Pura / Contratos / Faturas ({len(manifest_locacao)} documentos)\\nContem faturas e contratos de locacao de veiculos (Localiza, Movida, etc.).\\nExecute ./confirmar_validacao.py para registrar como 'locacao_veiculos'.\\n")

# Miniaturas
print("Configurando miniaturas para o Thunar...")
with ThreadPoolExecutor(max_workers=16) as pool:
    futures = [pool.submit(setup_thumb, s, p) for s, p in thumb_tasks]
    done = sum(1 for fut in futures if fut.result())

print(f"[CONCLUÍDO] {len(manifest_nfse)} movidos para 'nfse_identificadas' e {len(manifest_locacao)} para 'locacao_pura_contratos_faturas'.")
