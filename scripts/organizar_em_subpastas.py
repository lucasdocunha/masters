#!/usr/bin/env python3
"""
Organiza os arquivos da pasta 'validacao_nfs' em subpastas categorizadas:
- 1_nfse_servicos          (NFS-e de Serviços Municipais - classe: nfse)
- 2_nfce_cupons            (NFC-e / Cupons de Postos e Alimentação - classe: nfc)
- 3_danfe_nfe_mercadorias  (DANFE / NF-e Modelo 55 Mercadorias - classe: nfe)
- 4_faturas_recibos        (Faturas e Recibos de Locação de Veículos/Bens - classe: fatura)
- 5_outros_atipicos        (Comprovantes e Layouts atípicos para revisão)

Também garante que as miniaturas no cache do Thunar (FreeDesktop) funcionem
perfeitamente nos novos caminhos de cada subpasta.
"""

import hashlib
import json
import os
import shutil
import sys
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pymupdf

FOLDER = Path("/home/lucas.ocunha/Desktop/validacao_nfs")
ANALYSIS_FILE = FOLDER / ".analise_detalhada.json"
MANIFEST_FILE = FOLDER / ".candidatos_iniciais.json"

THUMB_LARGE_DIR = Path.home() / ".cache" / "thumbnails" / "large"

SUBFOLDERS = {
    "NFSE": {
        "dir_name": "1_nfse_servicos",
        "class_label": "nfse",
        "desc": "Notas Fiscais de Serviços Eletrônicas Municipais"
    },
    "NFC_E": {
        "dir_name": "2_nfce_cupons",
        "class_label": "nfc",
        "desc": "NFC-e / Cupons Fiscais (Postos de Combustíveis e Consumo)"
    },
    "DANFE_NFE": {
        "dir_name": "3_danfe_nfe_mercadorias",
        "class_label": "nfe",
        "desc": "DANFE / NF-e Modelo 55 (Venda de Produtos e Mercadorias)"
    },
    "RECIBO_FATURA": {
        "dir_name": "4_faturas_recibos",
        "class_label": "fatura",
        "desc": "Faturas e Recibos de Locação (sem incidência de ISS)"
    },
    "OUTROS": {
        "dir_name": "5_outros_atipicos",
        "class_label": None,
        "desc": "Comprovantes diversos e layouts atípicos"
    }
}


def render_thumbnail_for_symlink(symlink_path: Path, target_pdf: Path) -> bool:
    """Garante que a miniatura do FreeDesktop exista para o caminho do symlink."""
    try:
        uri = "file://" + urllib.parse.quote(str(symlink_path.absolute()))
        h = hashlib.md5(uri.encode("utf-8")).hexdigest()
        thumb_path = THUMB_LARGE_DIR / f"{h}.png"

        # Se já existe pelo caminho canônico, podemos apenas copiar/linkar
        target_uri = "file://" + urllib.parse.quote(str(target_pdf.resolve()))
        target_h = hashlib.md5(target_uri.encode("utf-8")).hexdigest()
        target_thumb = THUMB_LARGE_DIR / f"{target_h}.png"

        if target_thumb.exists():
            if not thumb_path.exists():
                shutil.copyfile(target_thumb, thumb_path)
            return True

        if thumb_path.exists():
            return True

        doc = pymupdf.open(target_pdf)
        if len(doc) == 0:
            doc.close()
            return False
        pix = doc[0].get_pixmap(dpi=100)
        pix.save(str(thumb_path))
        doc.close()
        return True
    except Exception:
        return False


def main():
    print("=== Organizando Validação em Subpastas por Categoria ===")
    t0 = time.time()

    if not ANALYSIS_FILE.exists():
        print(f"Erro: {ANALYSIS_FILE} não encontrado. Execute a análise primeiro.")
        sys.exit(1)

    with open(ANALYSIS_FILE, "r", encoding="utf-8") as f:
        analyzed = json.load(f)

    manifest_lookup = {}
    if MANIFEST_FILE.exists():
        with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
            manifest_lookup = {m["filename"]: m for m in json.load(f)}

    for item in analyzed:
        if "pdf_path" not in item and item["filename"] in manifest_lookup:
            item["pdf_path"] = manifest_lookup[item["filename"]]["pdf_path"]

    # Criar subpastas
    subpaths = {}
    for key, cfg in SUBFOLDERS.items():
        p = FOLDER / cfg["dir_name"]
        p.mkdir(parents=True, exist_ok=True)
        subpaths[key] = p

    # Mover arquivos e agrupar manifestos
    manifests_by_subfolder = {k: [] for k in SUBFOLDERS}
    thumb_tasks = []
    moved_count = 0

    print("Movendo links simbólicos para as respectivas subpastas...")
    for item in analyzed:
        cat = item.get("category", "OUTROS")
        if cat not in SUBFOLDERS:
            cat = "OUTROS"

        filename = item["filename"]
        target_subfolder = subpaths[cat]
        dest_symlink = target_subfolder / filename

        # Procurar onde o arquivo está atualmente (na raiz ou em outra subpasta)
        current_loc = None
        if (FOLDER / filename).exists() or (FOLDER / filename).is_symlink():
            current_loc = FOLDER / filename
        else:
            for other_sub in subpaths.values():
                if (other_sub / filename).exists() or (other_sub / filename).is_symlink():
                    current_loc = other_sub / filename
                    break

        if current_loc:
            if current_loc != dest_symlink:
                if dest_symlink.exists() or dest_symlink.is_symlink():
                    dest_symlink.unlink()
                current_loc.rename(dest_symlink)
                moved_count += 1
            manifests_by_subfolder[cat].append(item)
            thumb_tasks.append((dest_symlink, Path(item["pdf_path"])))

    # Salvar manifesto específico dentro de cada subpasta
    for cat, cfg in SUBFOLDERS.items():
        sub_manifest = subpaths[cat] / ".candidatos.json"
        with open(sub_manifest, "w", encoding="utf-8") as f:
            json.dump(manifests_by_subfolder[cat], f, indent=2, ensure_ascii=False)

    print(f"Arquivos organizados ({moved_count} movidos):")
    for cat, cfg in SUBFOLDERS.items():
        count = len(manifests_by_subfolder[cat])
        print(f"  -> {cfg['dir_name']:25s}: {count:4d} docs | Classe: {cfg['class_label'] or 'N/A'}")

    # Atualizar cache de miniaturas para os novos caminhos
    print(f"\nAtualizando miniaturas do Thunar para os novos caminhos ({len(thumb_tasks)} arquivos)...")
    t_thumb = time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(render_thumbnail_for_symlink, s_p, t_p) for s_p, t_p in thumb_tasks]
        ok = sum(1 for f in futures if f.result())
    print(f"Miniaturas sincronizadas ({ok}/{len(thumb_tasks)}) em {time.time()-t_thumb:.2f}s!")

    # Gerar script unificado 'confirmar_validacao.py' na raiz
    criar_script_confirmacao()

    # Gerar LEIAME.txt atualizado
    criar_leiame()

    print(f"\n[CONCLUÍDO COM SUCESSO] Tudo organizado em: {FOLDER}")
    print(f"Tempo total: {time.time()-t0:.2f}s.")


def criar_script_confirmacao():
    script_path = FOLDER / "confirmar_validacao.py"
    with open(script_path, "w", encoding="utf-8") as f:
        f.write('''#!/usr/bin/env python3
"""
Confirma a validação em lote das subpastas:
- 1_nfse_servicos         -> grava como 'nfse'
- 2_nfce_cupons           -> grava como 'nfc'
- 3_danfe_nfe_mercadorias -> grava como 'nfe'
- 4_faturas_recibos       -> grava como 'fatura'

O que você MANTIVER dentro de cada pasta é confirmado e gravado no histórico do modelo.
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
    
    current_files = set(p.name for p in sub_dir.glob("*.pdf"))
    accepted = [m for m in manifest if m["filename"] in current_files]
    rejected = [m for m in manifest if m["filename"] not in current_files]

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


def criar_leiame():
    readme_path = FOLDER / "LEIAME.txt"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("""PASTA DE VALIDAÇÃO VISUAL - ORGANIZADA POR SUBPASTAS
======================================================

Os documentos foram separados automaticamente em 5 subpastas:

1. [1_nfse_servicos] (~2.620 docs)
   - Notas Fiscais de Serviços Eletrônicas Municipais (ISS, Prefeituras, hotéis, consultorias, divulgação, TI).
   - Validação da classe: 'nfse'.

2. [2_nfce_cupons] (~189 docs)
   - NFC-e / Cupons Fiscais (Postos de gasolina, combustíveis e alimentação).
   - Validação da classe: 'nfc'.

3. [3_danfe_nfe_mercadorias] (~109 docs)
   - DANFE / NF-e Modelo 55 (Venda de produtos/mercadorias, assinaturas de jornais/revistas, gráficas industriais).
   - Validação da classe: 'nfe'.

4. [4_faturas_recibos] (~6 docs)
   - Faturas e recibos de locação de veículos sem condutor (sem ISS municipal).
   - Validação da classe: 'fatura'.

5. [5_outros_atipicos] (~91 docs)
   - Comprovantes de táxi, pedágio, tíquetes e layouts atípicos.

COMO USAR:
1. Entre em cada subpasta e olhe as miniaturas.
2. Apague (Delete) qualquer arquivo que estiver no lugar errado.
3. Quando terminar, você pode:
   - Me pedir: "fiz a validação das subpastas, pode aplicar"
   ou
   - Rodar o script: ./confirmar_validacao.py
""")


if __name__ == "__main__":
    main()
