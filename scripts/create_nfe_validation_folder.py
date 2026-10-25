#!/usr/bin/env python3
"""
Cria a pasta de validação visual em lote para a classe NFE (DANFE - Modelo 55 Mercantil da SEFAZ).
Calcula o centroide da classe Modelo 55 a partir de sementes confirmadas,
seleciona os 500 documentos mais próximos (sim >= 0.89), gera imagens .PNG em 150 DPI
e pré-gera miniaturas no cache do Thunar (FreeDesktop).
Também cria o script auxiliar 'confirmar_validacao.py'.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.parse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pymupdf

ROOT_DIR = Path("/home/lucas.ocunha/masters")
DATA_DIR = ROOT_DIR / "data"
PDF_DIR = Path("/media/ssd2/datasets/CEAP")
RENDERED_DIR = DATA_DIR / "rendered_pages"
NPZ_FILE = DATA_DIR / "latent_vectors_sample_5pct.npz"
META_FILE = DATA_DIR / "latent_vectors_sample_5pct_metadata.json"
LOOKUP_FILE = DATA_DIR / "pdf_metadata_lookup.json"
SEEDS_FILE = DATA_DIR / "seeds_modelo_55.json"
HISTORY_FILE = ROOT_DIR / "ml_backend" / "models" / "annotations_history.json"

DEST_DIR = Path("/home/lucas.ocunha/Desktop/validacao_nfe")
MASTERS_LINK = ROOT_DIR / "validacao_nfe"
THUMB_LARGE_DIR = Path.home() / ".cache" / "thumbnails" / "large"


def render_pdf_to_png(task: tuple) -> bool:
    pdf_path, png_path = task
    if png_path.exists() and png_path.stat().st_size > 0:
        return True
    try:
        doc = pymupdf.open(pdf_path)
        if len(doc) == 0:
            doc.close()
            return False
        pix = doc[0].get_pixmap(dpi=150)
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


def main(n_candidates: int = 500, min_sim: float = 0.88):
    print("=== Criação da Pasta de Validação para NFe (DANFE - Modelo 55) ===")
    t0 = time.time()

    RENDERED_DIR.mkdir(parents=True, exist_ok=True)
    THUMB_LARGE_DIR.mkdir(parents=True, exist_ok=True)
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    if not MASTERS_LINK.exists() and not MASTERS_LINK.is_symlink():
        MASTERS_LINK.symlink_to(DEST_DIR, target_is_directory=True)

    # 1. Carregar embeddings e metadados
    print(f"Carregando embeddings de {NPZ_FILE}...")
    npz = np.load(NPZ_FILE)
    combined = np.hstack([npz["dino_doc_embeddings"], npz["vlm_doc_embeddings"]]).astype(np.float32)
    norms = np.linalg.norm(combined, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    combined /= norms

    with open(META_FILE, "r", encoding="utf-8") as f:
        docs = json.load(f).get("documents", [])

    stem_to_idx = {Path(d.get("doc_name", "")).stem: idx for idx, d in enumerate(docs)}

    # 2. Carregar sementes de Modelo 55
    with open(SEEDS_FILE, "r", encoding="utf-8") as f:
        seeds = json.load(f)

    seed_stems = [s["stem"] for s in seeds if s["stem"] in stem_to_idx]
    print(f"Encontradas {len(seed_stems)} sementes confirmadas de Modelo 55.")

    if not seed_stems:
        print("Erro: Nenhuma semente de Modelo 55 encontrada.")
        sys.exit(1)

    seed_indices = [stem_to_idx[s] for s in seed_stems]
    mod55_centroid = np.mean(combined[seed_indices], axis=0)
    mod55_centroid /= np.linalg.norm(mod55_centroid)

    # 3. Calcular similaridade com todos os documentos não rotulados
    print("Calculando similaridade cosseno em relação ao centroide do Modelo 55...")
    sims = np.dot(combined, mod55_centroid)

    lookup: dict[str, Any] = {}
    if LOOKUP_FILE.exists():
        with open(LOOKUP_FILE, "r", encoding="utf-8") as f:
            lookup = json.load(f)

    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        history = json.load(f)

    labeled_stems = set(history.keys())
    candidates = []

    for idx, d in enumerate(docs):
        doc_name = d.get("doc_name", "")
        stem = Path(doc_name).stem
        if stem in labeled_stems:
            continue
        if d.get("num_pages", 1) != 1:
            continue
        pdf_path = PDF_DIR / doc_name
        if not pdf_path.exists():
            continue

        sim = float(sims[idx])
        if sim >= min_sim:
            meta = lookup.get(doc_name, {})
            candidates.append({
                "sim": sim,
                "doc_name": doc_name,
                "stem": stem,
                "pdf_path": str(pdf_path),
                "fornecedor": meta.get("fornecedor", "N/A"),
                "subcota": meta.get("subcota", "N/A"),
                "ano": meta.get("ano", "N/A"),
            })

    candidates.sort(key=lambda x: x["sim"], reverse=True)
    if n_candidates and len(candidates) > n_candidates:
        candidates = candidates[:n_candidates]

    print(f"Total de candidatos selecionados: {len(candidates):,} (sim: {candidates[-1]['sim']:.4f} a {candidates[0]['sim']:.4f})")

    # 4. Renderizar páginas faltantes para PNG
    render_tasks = []
    for c in candidates:
        png_target = RENDERED_DIR / f"{c['stem']}.png"
        c["png_target"] = str(png_target)
        if not png_target.exists() or png_target.stat().st_size == 0:
            render_tasks.append((Path(c["pdf_path"]), png_target))

    if render_tasks:
        print(f"Renderizando {len(render_tasks):,} imagens PNG (150 DPI) com 12 workers...")
        t_rend = time.time()
        with ProcessPoolExecutor(max_workers=12) as ex:
            results = list(ex.map(render_pdf_to_png, render_tasks))
        print(f"Imagens renderizadas ({sum(results)}/{len(render_tasks)}) em {time.time()-t_rend:.2f}s!")

    # 5. Criar links simbólicos na pasta Desktop/validacao_nfe
    print("Criando links simbólicos na pasta de validação...")
    for old_f in DEST_DIR.glob("rank_*.*"):
        old_f.unlink()

    initial_manifest = []
    thumb_tasks = []

    for rank, c in enumerate(candidates, 1):
        filename = f"rank_{rank:04d}_sim_{c['sim']:.4f}_{c['stem']}.png"
        symlink_path = DEST_DIR / filename
        png_target = Path(c["png_target"])

        if symlink_path.exists() or symlink_path.is_symlink():
            symlink_path.unlink()
        symlink_path.symlink_to(png_target)

        c_info = dict(c)
        c_info["rank"] = rank
        c_info["filename"] = filename
        initial_manifest.append(c_info)

        thumb_tasks.append((symlink_path, png_target))

    # Salvar manifesto
    manifest_path = DEST_DIR / ".candidatos.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(initial_manifest, f, indent=2, ensure_ascii=False)

    # 6. Pré-renderizar miniaturas FreeDesktop para Thunar
    print(f"Pré-gerando miniaturas para {len(thumb_tasks):,} arquivos com 8 workers...")
    t_th = time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(setup_thumbnail, s, p) for s, p in thumb_tasks]
        ok = sum(1 for f in futures if f.result())
    print(f"Miniaturas pré-geradas ({ok}/{len(thumb_tasks)}) em {time.time()-t_th:.2f}s!")

    # 7. Criar script confirmar_validacao.py
    script_confirmar = DEST_DIR / "confirmar_validacao.py"
    with open(script_confirmar, "w", encoding="utf-8") as f:
        f.write('''#!/usr/bin/env python3
"""
Confirma a validação dos documentos na pasta 'validacao_nfe'.
- O que você MANTIVER na pasta é registrado como classe 'nfe' (DANFE - Modelo 55).
- O que você APAGAR é desconsiderado.
Em seguida, retreina o modelo no ML backend e sincroniza com o Label Studio!
"""

import json
import subprocess
import sys
from pathlib import Path
import requests

FOLDER = Path(__file__).resolve().parent
MANIFEST_FILE = FOLDER / ".candidatos.json"
HISTORY_FILE = Path("/home/lucas.ocunha/masters/ml_backend/models/annotations_history.json")
ACCEPTED_FILE = Path("/home/lucas.ocunha/masters/data/accepted_nfe.json")
ML_BACKEND_URL = "http://localhost:9090"
LABEL_STUDIO_URL = "http://localhost:8085"

if not MANIFEST_FILE.exists():
    print(f"Erro: {MANIFEST_FILE} não encontrado.")
    sys.exit(1)

with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
    manifest = json.load(f)

current_stems = set(p.stem.split("_")[-1] for p in FOLDER.glob("rank_*.*") if p.suffix in (".png", ".pdf"))
accepted = [m for m in manifest if m["stem"] in current_stems]
rejected = [m for m in manifest if m["stem"] not in current_stems]

print("=== Validação da Classe NFE (DANFE - Modelo 55) ===")
print(f"Total inicial de candidatos: {len(manifest):,}")
print(f"  [OK] Mantidos (Confirmados como NFe/DANFE): {len(accepted):,}")
print(f"  [X]  Apagados (Rejeitados/Não-DANFE):       {len(rejected):,}")

if not accepted:
    print("\\nNenhum arquivo para confirmar.")
    sys.exit(0)

auto_yes = "-y" in sys.argv or "--yes" in sys.argv
if not auto_yes:
    ans = input(f"\\nDeseja gravar os {len(accepted):,} documentos como 'nfe' no modelo e retreinar? [S/n]: ")
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
        history[stem] = "nfe"
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"[OK] Histórico atualizado diretamente. Total agora: {len(history):,} anotações.")
except PermissionError:
    print("[Aviso] Atualizando histórico via container...")
    subprocess.run([
        "docker", "exec", "ceap-ml-backend", "python3", "-c",
        f"""
import json
with open('/data/accepted_nfe.json') as f:
    stems = set(json.load(f)['stems'])
with open('/app/models/annotations_history.json', 'r') as f:
    h = json.load(f)
for s in stems:
    h[s] = 'nfe'
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
    script_confirmar.chmod(0o755)

    # 8. Criar LEIAME.txt
    readme_path = DEST_DIR / "LEIAME.txt"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(f"""PASTA DE VALIDAÇÃO VISUAL - CLASSE NFE (DANFE - MODELO 55)
==========================================================

Esta pasta contém {len(candidates):,} documentos com alta similaridade visual e semântica com o DANFE (Modelo 55 da SEFAZ).

COMO IDENTIFICAR O DANFE (MODELO 55):
- Formato Folha A4 Retrato.
- Canhoto de recebimento no topo ("RECEBEMOS DE [EMPRESA] OS PRODUTOS...").
- Cabeçalho: "DANFE - DOCUMENTO AUXILIAR DA NOTA FISCAL ELETRÔNICA - 0-ENTRADA 1-SAÍDA".
- Código de barras com chave de 44 dígitos (dígitos 20-22 são '55').
- Quadro de cálculo do imposto (ICMS) e tabela de produtos com colunas NCM/CST/CFOP.

COMO USAR:
1. Abra esta pasta no Gerenciador de Arquivos (Thunar) ou no visualizador de imagens (Ristretto).
2. Navegue usando as setas do teclado (-> e <-).
3. Selecione e APAGUE (tecla Del) qualquer arquivo que NÃO for DANFE / Modelo 55 (ex: cupons NFC-e ou notas municipais de serviço).
4. Quando terminar a revisão:
   - Me avise aqui: "fiz a validação da nfe, pode aplicar"
   ou
   - Execute o script na pasta:
     cd ~/Desktop/validacao_nfe && ./confirmar_validacao.py
""")

    print(f"\n[SUCESSO] Pasta de validação criada em:")
    print(f"  -> {DEST_DIR}")
    print(f"Tempo total: {time.time()-t0:.2f}s.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cria pasta de validação visual para NF-e (Modelo 55)")
    parser.add_argument("--count", type=int, default=500, help="Quantidade de candidatos (default: 500)")
    parser.add_argument("--min-sim", type=float, default=0.88, help="Similaridade mínima (default: 0.88)")
    args = parser.parse_args()
    main(n_candidates=args.count, min_sim=args.min_sim)
