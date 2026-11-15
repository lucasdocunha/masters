#!/usr/bin/env python3
"""
Reorganiza a pasta de validação da meta 20k no Desktop.
Atuais: 18.664 rotulados. Faltam: exatamente 1.336 documentos.

Estrutura:
  Desktop/validacao_nfe_meta_20k/
    ├── 01_faltam_1336_para_20k/   (Os 1.336 documentos mais similares para bater 20k exatos)
    ├── 02_reserva_363_docs/       (363 documentos de reserva para cobrir eventuais descartes)
    ├── confirmar_validacao.py
    ├── LEIAME.txt
    └── .manifesto_meta_20k.json
"""

import hashlib
import json
import os
import re
import shutil
import time
import urllib.parse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from dados.config import DESKTOP_DIR, ROOT_DIR

DATA_DIR = ROOT_DIR / "data"
BATCHES_DIR = DATA_DIR / "validacao_batches"
TARGET_DIR = BATCHES_DIR / "validacao_nfe_meta_20k"
DESKTOP_LINK = DESKTOP_DIR / "validacao_nfe_meta_20k"
RENDERED_DIR = DATA_DIR / "rendered_pages"
THUMB_DIR = Path.home() / ".cache" / "thumbnails" / "large"
HISTORY_FILE = DATA_DIR / "models" / "annotations_history.json"
MANIFEST_FILE = TARGET_DIR / ".manifesto_meta_20k.json"


def setup_thumbnails(symlink_paths, png_path):
    for sp in symlink_paths:
        try:
            uri = "file://" + urllib.parse.quote(str(sp.absolute()))
            h = hashlib.md5(uri.encode("utf-8")).hexdigest()
            t_path = THUMB_DIR / f"{h}.png"
            if not t_path.exists() and png_path.exists():
                t_path.symlink_to(png_path.absolute())
        except Exception:
            pass


def main():
    print("=" * 70)
    print("REORGANIZANDO CANDIDATOS RESTANTES PARA ATINGIR A META DE 20.000 NF-E")
    print("=" * 70)

    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        hist = json.load(f)
    norm_hist = {k.replace(".pdf", ""): v for k, v in hist.items()}
    current_nfe = sum(1 for v in norm_hist.values() if v == "nfe")
    needed = 20000 - current_nfe
    print(f"NF-e atualmente rotuladas: {current_nfe:,}")
    print(f"Faltam para atingir 20k:   {needed:,} documentos")

    # 1. Carregar manifesto existente
    with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
        old_manifest = json.load(f)

    # Filtrar apenas os NÃO rotulados
    unlabeled = [v for k, v in old_manifest.items() if k not in norm_hist]
    unlabeled.sort(key=lambda x: -float(x.get("sim", 0.0)))
    print(f"Total de candidatos disponíveis e não rotulados: {len(unlabeled):,}")

    # Separar os exatos 1.336 mais similares e a reserva
    top_needed = unlabeled[:needed]
    reserva = unlabeled[needed:]

    # Limpar subpastas antigas
    for p in TARGET_DIR.iterdir():
        if p.is_dir() and (p.name.startswith("lote_") or p.name.startswith("01_") or p.name.startswith("02_")):
            shutil.rmtree(p)

    folder_1 = TARGET_DIR / f"01_faltam_{needed}_para_20k"
    folder_2 = TARGET_DIR / f"02_reserva_{len(reserva)}_docs"
    folder_1.mkdir(parents=True, exist_ok=True)
    folder_2.mkdir(parents=True, exist_ok=True)

    batches = [
        (folder_1, top_needed, DESKTOP_LINK / folder_1.name),
        (folder_2, reserva, DESKTOP_LINK / folder_2.name)
    ]

    new_manifest = {}
    thumb_tasks = []

    for b_dir, items, desktop_b_dir in batches:
        print(f"\nPopulando pasta: {b_dir.name} ({len(items):,} documentos)...")
        for rank, it in enumerate(items, 1):
            s = it["doc_id"]
            clean_forn = it.get("clean_forn") or re.sub(r"[^\w\s-]", "", it.get("fornecedor", "")).strip()
            clean_forn = re.sub(r"\s+", "_", clean_forn)[:24] or "DESCONHECIDO"
            val = float(it.get("valor", 0.0))
            val_str = f"R${val:.2f}".replace(".", ",")
            sim_str = f"{float(it.get('sim', 1.0)):.3f}"

            link_name = f"rank_{rank:04d}_[sim_{sim_str}]_[{clean_forn}]_[{val_str}]_{s}.png"
            target_png = RENDERED_DIR / f"{s}.png"
            batch_link = b_dir / link_name
            desktop_link_file = desktop_b_dir / link_name

            if batch_link.exists() or batch_link.is_symlink():
                batch_link.unlink()
            if target_png.exists():
                batch_link.symlink_to(target_png)

            thumb_tasks.append(([batch_link, desktop_link_file], target_png))

            new_manifest[s] = {
                "doc_id": s,
                "folder": b_dir.name,
                "rank": rank,
                "filename": link_name,
                "sim": it["sim"],
                "fornecedor": it["fornecedor"],
                "valor": val
            }

    # Miniaturas
    print("Garantindo miniaturas para navegação instantânea...")
    with ThreadPoolExecutor(max_workers=32) as ex:
        ex.map(lambda t: setup_thumbnails(t[0], t[1]), thumb_tasks)

    # Link Desktop
    if DESKTOP_LINK.is_symlink() or DESKTOP_LINK.exists():
        DESKTOP_LINK.unlink()
    DESKTOP_LINK.symlink_to(TARGET_DIR)

    # Salvar novo manifesto
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(new_manifest, f, indent=2, ensure_ascii=False)

    # Atualizar script de confirmação
    create_confirm_script(TARGET_DIR, current_nfe)

    # Atualizar LEIAME
    create_readme(TARGET_DIR, current_nfe, needed, len(reserva))

    print("\n" + "=" * 70)
    print(f"ESTRUTURA PRONTA NO DESKTOP: ~/Desktop/{DESKTOP_LINK.name}")
    print(f"  - 01_faltam_{needed}_para_20k: {len(top_needed):,} documentos")
    print(f"  - 02_reserva_{len(reserva)}_docs: {len(reserva):,} documentos")
    print("=" * 70)


def create_confirm_script(target_dir: Path, initial_nfe: int):
    script_content = f"""#!/usr/bin/env python3
\"\"\"
Confirmação dos documentos para bater os 20.000 NF-e.
Varre os arquivos PNG que NÃO foram apagados e sincroniza no annotations_history.json.
\"\"\"

import argparse
import json
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parents[2]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from dados.labeling.labels_csv import write_labels_dir

HISTORY_FILE = PROJECT_DIR / "data" / "models" / "annotations_history.json"
ROTULOS_DIR = PROJECT_DIR / "data" / "rotulos"


def main():
    parser = argparse.ArgumentParser(description="Confirma os candidatos para a meta de 20k.")
    parser.add_argument("--pasta", type=str, default=None, help="Confirmar apenas uma pasta específica")
    args = parser.parse_args()

    search_dir = BASE_DIR / args.pasta if args.pasta else BASE_DIR

    print("=== CONFIRMAÇÃO DA META DE 20.000 NF-E ===")
    png_files = list(search_dir.rglob("*.png"))
    print(f"Arquivos mantidos encontrados: {{len(png_files):,}}")

    if not png_files:
        print("[AVISO] Nenhum arquivo PNG encontrado.")
        return

    doc_ids = set()
    for p in png_files:
        m = re.search(r'_(\\d+)\\.png$', p.name)
        if m:
            doc_ids.add(m.group(1))

    print(f"IDs válidos identificados: {{len(doc_ids):,}}")

    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        history = json.load(f)

    added = 0
    for d_id in doc_ids:
        if d_id not in history or history[d_id] != "nfe":
            history[d_id] = "nfe"
            added += 1

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    total_nfe = sum(1 for v in history.values() if v == "nfe")
    print(f"\\n[SUCESSO] {{added:,}} novas anotações adicionadas como 'nfe'.")
    print(f"          Total histórico de NF-e agora: {{total_nfe:,}} / 20.000!")

    if total_nfe >= 20000:
        print("          >>> PARABÉNS! META DE 20.000 NF-E ATINGIDA COM SUCESSO! <<<")

    print("\\nSincronizando com as partes CSV em data/rotulos/...")
    write_labels_dir(history, ROTULOS_DIR)
    print("[OK] Base de rótulos CSV sincronizada!")
    print("Agora basta me avisar no chat: 'confirmei os que faltavam, pode atualizar o modelo'.")


if __name__ == "__main__":
    main()
"""
    script_path = target_dir / "confirmar_validacao.py"
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script_content)
    script_path.chmod(0o755)


def create_readme(target_dir: Path, current_nfe: int, needed: int, reserva_cnt: int):
    readme = f"""PASTA DE VALIDAÇÃO VISUAL - RETA FINAL: META 20.000 NF-E (DANFE / MODELO 55)
========================================================================================

Meta do Projeto: Atingir 20.000 documentos rotulados como NF-e.
Situação Atual:  {current_nfe:,} rotulados
Faltam:          {needed:,} documentos exatos!

ESTRUTURA DAS PASTAS:
- 01_faltam_{needed}_para_20k : Exatamente {needed:,} documentos com maior similaridade ArcFace.
                              Validando esta pasta inteira, a meta de 20.000 é 100% batida!
- 02_reserva_{reserva_cnt}_docs  : {reserva_cnt:,} documentos de reserva para caso você delete algum
                              documento duvidoso na pasta principal.

COMO REVISAR:
1. Abra a pasta '01_faltam_{needed}_para_20k' no Thunar.
2. Navegue com as setas (<- e ->).
3. Aperte DELETE se encontrar algum documento que NÃO seja DANFE.
4. Ao concluir, execute:
     cd ~/Desktop/validacao_nfe_meta_20k && ./confirmar_validacao.py
   ou simplesmente me diga no chat:
     "validei a pasta, pode aplicar"
"""
    with open(target_dir / "LEIAME.txt", "w", encoding="utf-8") as f:
        f.write(readme)


if __name__ == "__main__":
    main()
