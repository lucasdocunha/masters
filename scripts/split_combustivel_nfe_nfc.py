#!/usr/bin/env python3
"""
Separa a pasta Desktop/validacao_combustivel em duas subpastas:
1. nfc_cupons_fiscais (Cupons Fiscais / NFC-e Modelo 65 / SAT)
2. nfe_danfe_modelo_55 (DANFE NF-e Modelo 55 - folha A4)
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
from rapidocr_onnxruntime import RapidOCR

BASE_DIR = Path("/home/lucas.ocunha/Desktop/validacao_combustivel")
DIR_NFC = BASE_DIR / "nfc_cupons_fiscais"
DIR_NFE = BASE_DIR / "nfe_danfe_modelo_55"
THUMB_DIR = Path(os.path.expanduser("~/.cache/thumbnails/large"))

DIR_NFC.mkdir(parents=True, exist_ok=True)
DIR_NFE.mkdir(parents=True, exist_ok=True)

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

manifest_nfc = []
manifest_nfe = []
thumb_tasks = []

print("Separando combustível em NFC vs NFE...")

for f in current_files:
    stem = f.stem.split("_")[-1]
    m = manifest_by_stem.get(stem)
    if not m:
        continue
    
    doc = pymupdf.open(m["pdf_path"])
    page = doc[0]
    rect = page.rect
    width, height = rect.width, rect.height
    aspect_ratio = height / width if width > 0 else 1.0
    txt = " ".join(p.get_text() for p in doc).lower()
    doc.close()
    
    png_path = Path(m["png_target"])
    if len(txt.strip()) < 50 and png_path.exists():
        res, _ = ocr(str(png_path))
        if res:
            txt = txt + " " + " ".join([line[1] for line in res]).lower()
            
    is_nfe_55 = any(k in txt for k in [
        "documento auxiliar da nota fiscal eletrônica", "modelo 55", "danfe simplificado",
        "dados do produto/serviço", "cálculo do imposto", "transportador/volumes transportados"
    ]) and not any(k in txt for k in ["nfc-e", "modelo 65", "extrato do sat", "cupom fiscal eletrônico"])
    
    is_nfc_65 = any(k in txt for k in [
        "nfc-e", "consumidor eletrônica", "modelo 65", "extrato do sat", 
        "cupom fiscal", "consulta pela chave de acesso em", "emissão por edanfe"
    ]) or (aspect_ratio > 1.8)

    # Decisão final
    if is_nfc_65 and not is_nfe_55:
        target_dir = DIR_NFC
        manifest_nfc.append(m)
    elif is_nfe_55:
        target_dir = DIR_NFE
        manifest_nfe.append(m)
    else:
        # Se largura/altura padrão A4 e tem "danfe" -> nfe, senão -> nfc
        if aspect_ratio < 1.6 and ("danfe" in txt or "chave" in txt or "&kdyh" in txt):
            target_dir = DIR_NFE
            manifest_nfe.append(m)
        else:
            target_dir = DIR_NFC
            manifest_nfc.append(m)

    new_symlink = target_dir / f.name
    if new_symlink.exists() or new_symlink.is_symlink():
        new_symlink.unlink()
    new_symlink.symlink_to(png_path)
    f.unlink()
    thumb_tasks.append((new_symlink, png_path))

# Salvar manifestos
with open(DIR_NFC / ".candidatos.json", "w", encoding="utf-8") as f:
    json.dump(manifest_nfc, f, indent=2, ensure_ascii=False)

with open(DIR_NFE / ".candidatos.json", "w", encoding="utf-8") as f:
    json.dump(manifest_nfe, f, indent=2, ensure_ascii=False)

# Scripts de confirmação
script_nfc = f"""#!/usr/bin/env python3
import json, subprocess, sys, time
from pathlib import Path
import requests

FOLDER = Path(__file__).resolve().parent
with open(FOLDER / ".candidatos.json") as f:
    manifest = json.load(f)
current_stems = set(p.stem.split("_")[-1] for p in FOLDER.glob("rank_*.*") if p.suffix in (".png", ".pdf"))
accepted = [m["stem"] for m in manifest if m["stem"] in current_stems]

print(f"Confirmando {{len(accepted)}} documentos como NFC (Cupom Fiscal / NFC-e)...")
with open("/home/lucas.ocunha/masters/data/accepted_combustivel_as_nfc.json", "w") as f:
    json.dump({{"total": len(accepted), "stems": accepted}}, f, indent=2)

h_file = Path("/home/lucas.ocunha/masters/ml_backend/models/annotations_history.json")
with open(h_file) as f:
    h = json.load(f)
for s in accepted:
    h[s] = "nfc"
with open(h_file, "w") as f:
    json.dump(h, f, indent=2)

subprocess.run(["docker", "restart", "ceap-ml-backend"], check=True)
time.sleep(3)
requests.post("http://localhost:9090/train")
print("[OK] Documentos gravados como NFC e modelo retreinado!")
"""
with open(DIR_NFC / "confirmar_validacao.py", "w") as f:
    f.write(script_nfc)
(DIR_NFC / "confirmar_validacao.py").chmod(0o755)

script_nfe = f"""#!/usr/bin/env python3
import json, subprocess, sys, time
from pathlib import Path
import requests

FOLDER = Path(__file__).resolve().parent
with open(FOLDER / ".candidatos.json") as f:
    manifest = json.load(f)
current_stems = set(p.stem.split("_")[-1] for p in FOLDER.glob("rank_*.*") if p.suffix in (".png", ".pdf"))
accepted = [m["stem"] for m in manifest if m["stem"] in current_stems]

print(f"Confirmando {{len(accepted)}} documentos como NFE (DANFE Modelo 55)...")
with open("/home/lucas.ocunha/masters/data/accepted_combustivel_as_nfe.json", "w") as f:
    json.dump({{"total": len(accepted), "stems": accepted}}, f, indent=2)

h_file = Path("/home/lucas.ocunha/masters/ml_backend/models/annotations_history.json")
with open(h_file) as f:
    h = json.load(f)
for s in accepted:
    h[s] = "nfe"
with open(h_file, "w") as f:
    json.dump(h, f, indent=2)

subprocess.run(["docker", "restart", "ceap-ml-backend"], check=True)
time.sleep(3)
requests.post("http://localhost:9090/train")
print("[OK] Documentos gravados como NFE e modelo retreinado!")
"""
with open(DIR_NFE / "confirmar_validacao.py", "w") as f:
    f.write(script_nfe)
(DIR_NFE / "confirmar_validacao.py").chmod(0o755)

# LEIAME
with open(DIR_NFC / "LEIAME.txt", "w") as f:
    f.write(f"SUBPASTA: Cupons Fiscais / NFC-e Modelo 65 ({len(manifest_nfc)} documentos)\\nContem cupons fiscais e NFC-e de postos de gasolina.\\nExecute ./confirmar_validacao.py para registrar na classe 'nfc'.\\n")

with open(DIR_NFE / "LEIAME.txt", "w") as f:
    f.write(f"SUBPASTA: DANFE NF-e Modelo 55 ({len(manifest_nfe)} documentos)\\nContem notas fiscais eletronicas (formato A4) emitidas por postos de gasolina.\\nExecute ./confirmar_validacao.py para registrar na classe 'nfe'.\\n")

print("Configurando miniaturas para o Thunar...")
with ThreadPoolExecutor(max_workers=16) as pool:
    futures = [pool.submit(setup_thumb, s, p) for s, p in thumb_tasks]
    done = sum(1 for fut in futures if fut.result())

print(f"[CONCLUÍDO] {len(manifest_nfc)} movidos para 'nfc_cupons_fiscais' e {len(manifest_nfe)} para 'nfe_danfe_modelo_55'.")
