import json
import re
import time
import shutil
import hashlib
import urllib.parse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import numpy as np
import pymupdf

ROOT_DIR = Path("/home/lucas.ocunha/masters")
DATA_DIR = ROOT_DIR / "data"
PDF_DIR = Path("/media/ssd2/datasets/CEAP")
RENDERED_DIR = DATA_DIR / "rendered_pages"
LOOKUP_FILE = DATA_DIR / "pdf_metadata_lookup.json"
HISTORY_FILE = ROOT_DIR / "ml_backend" / "models" / "annotations_history.json"
NPZ_FILE = DATA_DIR / "latent_vectors_sample_5pct.npz"
META_FILE = DATA_DIR / "latent_vectors_sample_5pct_metadata.json"

DIR_AGUA = Path("/home/lucas.ocunha/Desktop/validacao_agua")
DIR_ENERGIA = Path("/home/lucas.ocunha/Desktop/validacao_energia")
DESKTOP_ENERGIA_LINK = Path("/home/lucas.ocunha/Desktop/energia")
MASTERS_ENERGIA_LINK = ROOT_DIR / "validacao_energia"
AGUA_ENERGIA_LINK = DIR_AGUA / "energia"

THUMB_LARGE_DIR = Path.home() / ".cache" / "thumbnails" / "large"

print("=== SEPARAÇÃO AUTOMÁTICA: ÁGUA vs ENERGIA ===")

# Criar pastas
DIR_ENERGIA.mkdir(parents=True, exist_ok=True)
DIR_AGUA.mkdir(parents=True, exist_ok=True)
RENDERED_DIR.mkdir(parents=True, exist_ok=True)
THUMB_LARGE_DIR.mkdir(parents=True, exist_ok=True)

# Criar links convenientes para a pasta energia
for link in [DESKTOP_ENERGIA_LINK, MASTERS_ENERGIA_LINK, AGUA_ENERGIA_LINK]:
    if not link.exists() and not link.is_symlink():
        try:
            link.symlink_to(DIR_ENERGIA, target_is_directory=True)
        except Exception as e:
            print(f"Aviso ao criar link {link}: {e}")

# 1. Carregar dados
lookup = json.load(open(LOOKUP_FILE))
history = json.load(open(HISTORY_FILE))
manifest_agua = json.load(open(DIR_AGUA / ".candidatos.json"))

energy_patterns = [
    r"\bCEMIG\b", r"\bENERGISA\b", r"\bLIGHT\b", r"\bCPFL\b", r"\bCEEE\b",
    r"\bCOELCE\b", r"\bCOELBA\b", r"\bELETROPAULO\b", r"\bCOPEL\b", r"\bCELG\b",
    r"\bEQUATORIAL\b", r"\bCELESC\b", r"\bCELPE\b", r"\bENEL\b", r"\bEDP\b",
    r"\bESCELSA\b", r"\bELETRICIDADE\b", r"\bENERGIA\b", r"\bELETRIC", r"\bELETR[OÓ]",
    r"\bELETROBRAS\b", r"\bELETRONORTE\b", r"\bCHESF\b", r"\bFURNAS\b", r"\bAMPLA\b",
    r"\bRGE\b", r"\bNEOENERGIA\b", r"\bFOR[CÇ]A\s+E\s+LUZ\b", r"\bCERON\b", r"\bELETROACRE\b",
    r"\bCEA\b", r"\bCAEEL\b", r"\bCEB\b", r"\bDMED\b", r"\bBANDEIRANTE\b", r"\bPIRATININGA\b",
    r"\bSANTA\s+CRUZ\b", r"\bSULGIPE\b", r"\bEMG\b", r"\bENF\b", r"\bCOMPANHIA\s+ENERG[EÉ]TICA\b",
    r"\bENERSUL\b", r"\bCELPA\b", r"\bCENTRAIS\s+EL[EÉ]TRICAS\b", r"\bDISTRIBUIDORA\s+DE\s+ENERGIA\b"
]
re_energy = re.compile("|".join(energy_patterns), re.IGNORECASE)

water_patterns = [
    r"\bSABESP\b", r"\bCEDAE\b", r"\bCOPASA\b", r"\bSANEPAR\b", r"\bCOMPESA\b",
    r"\bEMBASA\b", r"\bCAGECE\b", r"\bCAESB\b", r"\bSANEAGO\b", r"\bCORSAN\b",
    r"\bCASAN\b", r"\bCAER\b", r"\bCAERN\b", r"\bCOSANPA\b", r"\bDESO\b",
    r"\bAGESPISA\b", r"\bSANESUL\b", r"\bDAE\b", r"\bSAAE\b", r"AGUAS\s+DE", r"ÁGUAS\s+DE",
    r"SANEAMENTO", r"AGUA\s+E\s+ESGOTO", r"ÁGUA\s+E\s+ESGOTO", r"AGUAS\s+GUARIROBAS",
    r"\bBRK\b", r"\bAGUAS\b", r"\bÁGUAS\b", r"\bSANEAR\b", r"\bSEMASA\b"
]
re_water = re.compile("|".join(water_patterns), re.IGNORECASE)

def sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[^\w\s-]", "", name).strip()
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned[:25]

def setup_thumbnail(symlink_path: Path, png_path: Path) -> bool:
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
        pix = doc[0].get_pixmap(dpi=72)
        pix.save(str(thumb_path))
        doc.close()
        return True
    except Exception:
        return False

# 2. Separar os 500 candidatos existentes do manifest_agua
energy_from_manifest = []
water_from_manifest = []
other_from_manifest = []

for m in manifest_agua:
    forn = m.get("fornecedor", "")
    is_e = bool(re_energy.search(forn))
    is_w = bool(re_water.search(forn))
    if is_e and not is_w:
        energy_from_manifest.append(m)
    elif is_w and not is_e:
        water_from_manifest.append(m)
    elif is_e and is_w:
        if not ("saneamento" in forn.lower() or "agua" in forn.lower() or "esgoto" in forn.lower()):
            energy_from_manifest.append(m)
        else:
            water_from_manifest.append(m)
    else:
        other_from_manifest.append(m)

print(f"Candidatos do lote atual: {len(energy_from_manifest)} Energia, {len(water_from_manifest)} Água, {len(other_from_manifest)} Outros")

# 3. Montar a pasta ENERGIA
print("\nMontando pasta Desktop/validacao_energia...")
for old_f in DIR_ENERGIA.glob("rank_*.*"):
    old_f.unlink()

manifest_energia = []
thumb_tasks_energia = []

energy_from_manifest.sort(key=lambda x: x.get("sim", 0), reverse=True)

for rank, c in enumerate(energy_from_manifest, 1):
    forn_tag = sanitize_filename(c["fornecedor"])
    filename = f"rank_{rank:04d}_sim_{c.get('sim', 0):.4f}_[{forn_tag}]_{c['stem']}.png"
    symlink_path = DIR_ENERGIA / filename
    png_target = Path(c["png_target"])
    
    if symlink_path.exists() or symlink_path.is_symlink():
        symlink_path.unlink()
    symlink_path.symlink_to(png_target)
    
    c_info = dict(c)
    c_info["rank"] = rank
    c_info["filename"] = filename
    manifest_energia.append(c_info)
    thumb_tasks_energia.append((symlink_path, png_target))

with open(DIR_ENERGIA / ".candidatos.json", "w", encoding="utf-8") as f:
    json.dump(manifest_energia, f, indent=2, ensure_ascii=False)

print(f"  -> {len(manifest_energia)} documentos de ENERGIA inseridos.")

# Script confirmar_validacao.py para Energia
script_energia = DIR_ENERGIA / "confirmar_validacao.py"
with open(script_energia, "w", encoding="utf-8") as f:
    f.write('''#!/usr/bin/env python3
"""
Confirma a validação dos documentos na pasta 'validacao_energia'.
- O que você MANTIVER na pasta é registrado como classe 'energia' (Contas de Luz / Energia Elétrica).
- O que você APAGAR é desconsiderado.
Em seguida, retreina o modelo no ML backend automaticamente!
"""

import json
import subprocess
import sys
import time
from pathlib import Path
import requests

FOLDER = Path(__file__).resolve().parent
MANIFEST_FILE = FOLDER / ".candidatos.json"
HISTORY_FILE = Path("/home/lucas.ocunha/masters/ml_backend/models/annotations_history.json")
ACCEPTED_FILE = Path("/home/lucas.ocunha/masters/data/accepted_energia.json")
ML_BACKEND_URL = "http://localhost:9090"

if not MANIFEST_FILE.exists():
    print(f"Erro: {MANIFEST_FILE} não encontrado.")
    sys.exit(1)

with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
    manifest = json.load(f)

current_stems = set(p.stem.split("_")[-1] for p in FOLDER.glob("rank_*.*") if p.suffix in (".png", ".pdf"))
accepted = [m for m in manifest if m["stem"] in current_stems]
rejected = [m for m in manifest if m["stem"] not in current_stems]

print("=== Validação da Classe ENERGIA (Contas de Luz / Eletricidade) ===")
print(f"Total inicial de candidatos: {len(manifest):,}")
print(f"  [OK] Mantidos (Confirmados como Energia): {len(accepted):,}")
print(f"  [X]  Apagados (Rejeitados/Outros):        {len(rejected):,}")

if not accepted:
    print("\\nNenhum arquivo para confirmar.")
    sys.exit(0)

auto_yes = "-y" in sys.argv or "--yes" in sys.argv
if not auto_yes:
    ans = input(f"\\nDeseja gravar os {len(accepted):,} documentos como 'energia' no modelo e retreinar? [S/n]: ")
    if ans.strip().lower() not in ["", "s", "sim", "y", "yes"]:
        print("Operação cancelada.")
        sys.exit(0)

accepted_stems = [m["stem"] for m in accepted]
ACCEPTED_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(ACCEPTED_FILE, "w", encoding="utf-8") as f:
    json.dump({"total": len(accepted), "stems": accepted_stems}, f, indent=2)

try:
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        history = json.load(f)
    for stem in accepted_stems:
        history[stem] = "energia"
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"[OK] Histórico atualizado. Total agora: {len(history):,} anotações.")
except PermissionError:
    print("[Aviso] Atualizando histórico via container...")
    subprocess.run([
        "docker", "exec", "ceap-ml-backend", "python3", "-c",
        f"""
import json
with open('/data/accepted_energia.json') as f:
    stems = set(json.load(f)['stems'])
with open('/app/models/annotations_history.json', 'r') as f:
    h = json.load(f)
for s in stems:
    h[s] = 'energia'
with open('/app/models/annotations_history.json', 'w') as f:
    json.dump(h, f, indent=2)
print(f'Histórico atualizado via container: {len(h)} anotações.')
"""
    ], check=True)

# Disparar retreino
print("\\nDisparando retreino no ML backend...")
try:
    subprocess.run(["docker", "restart", "ceap-ml-backend"], check=True)
    time.sleep(3)
    res = requests.post(f"{ML_BACKEND_URL}/train", timeout=60)
    if res.status_code == 200:
        data = res.json()
        print(f"[OK SUCESSO] Modelo retreinado: {data.get('version')} | Acurácia treino: {data.get('train_accuracy', 0):.2%}")
    else:
        print(f"Resposta retreino: {res.status_code} - {res.text}")
except Exception as e:
    print(f"Aviso no retreino: {e}")

print("\\nProcesso concluído com sucesso!")
''')
script_energia.chmod(0o755)

# LEIAME para Energia
with open(DIR_ENERGIA / "LEIAME.txt", "w", encoding="utf-8") as f:
    f.write(f"""PASTA DE VALIDAÇÃO VISUAL - CLASSE ENERGIA (CONTAS DE LUZ)
==========================================================

Esta pasta contém {len(manifest_energia):,} contas de energia elétrica separadas das contas de água.

CONCESSIONÁRIAS DE ENERGIA PRESENTES:
- CEMIG, ENEL, CPFL, COELBA, LIGHT, ENERGISA, EQUATORIAL, COPEL, CEEE, CELESC, EDP, CELG, etc.

COMO USAR:
1. Abra esta pasta no Thunar ou visualizador de imagens (Ristretto).
2. Navegue usando as setas do teclado (-> e <-).
3. Selecione e APAGUE (tecla Del) qualquer arquivo que NÃO for conta de energia elétrica.
4. Quando terminar a revisão:
   - Me avise aqui: "fiz a validação da energia, pode aplicar"
   ou
   - Execute o script na pasta:
     cd ~/Desktop/validacao_energia && ./confirmar_validacao.py
""")

# 4. Atualizar a pasta AGUA (removendo todos os arquivos de energia)
print("\nAtualizando pasta Desktop/validacao_agua (deixando apenas contas de água)...")
for old_f in DIR_AGUA.glob("rank_*.*"):
    old_f.unlink()

manifest_agua_limpo = []
thumb_tasks_agua = []

water_from_manifest.sort(key=lambda x: x.get("sim", 0), reverse=True)

for rank, c in enumerate(water_from_manifest, 1):
    forn_tag = sanitize_filename(c["fornecedor"])
    filename = f"rank_{rank:04d}_sim_{c.get('sim', 0):.4f}_[{forn_tag}]_{c['stem']}.png"
    symlink_path = DIR_AGUA / filename
    png_target = Path(c["png_target"])
    
    if symlink_path.exists() or symlink_path.is_symlink():
        symlink_path.unlink()
    symlink_path.symlink_to(png_target)
    
    c_info = dict(c)
    c_info["rank"] = rank
    c_info["filename"] = filename
    manifest_agua_limpo.append(c_info)
    thumb_tasks_agua.append((symlink_path, png_target))

with open(DIR_AGUA / ".candidatos.json", "w", encoding="utf-8") as f:
    json.dump(manifest_agua_limpo, f, indent=2, ensure_ascii=False)

print(f"  -> {len(manifest_agua_limpo)} documentos de ÁGUA mantidos na pasta.")

# 5. Garantir miniaturas para ambas as pastas
print("\nPré-gerando miniaturas para o Thunar...")
all_thumb_tasks = thumb_tasks_energia + thumb_tasks_agua
t_th = time.time()
with ThreadPoolExecutor(max_workers=16) as ex:
    futures = [ex.submit(setup_thumbnail, s, p) for s, p in all_thumb_tasks]
    ok = sum(1 for f in futures if f.result())
print(f"Miniaturas verificadas ({ok}/{len(all_thumb_tasks)}) em {time.time()-t_th:.2f}s!")

print("\n[SUCESSO] Separação concluída perfeitamente!")
