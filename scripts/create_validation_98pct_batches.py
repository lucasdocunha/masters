#!/usr/bin/env python3
"""
Gera pastas de validação no Desktop para TODOS os 73.731 documentos com similaridade >= 98%
no espaço métrico ArcFace 128-D.

Estrutura gerada:
  Desktop/validacao_98pct/
    ├── agua/                     (29 docs)
    ├── fatura/                   (12 docs)
    ├── passagem_rodoviaria/      (104 docs)
    ├── passagem_aerea/           (119 docs)
    ├── telefonia/                (158 docs)
    ├── energia/                  (198 docs)
    ├── pedagio/                  (266 docs)
    ├── nfe/                      (3.908 docs em lotes de 2k)
    │   ├── lote_01_2000/
    │   └── lote_02_1908/
    ├── recibo/                   (4.759 docs em lotes de 2.5k)
    │   ├── lote_01_2500/
    │   └── lote_02_2259/
    ├── nfse/                     (9.300 docs em lotes de 2k)
    ├── nfc/                      (26.962 docs em lotes de 2k)
    ├── uber/                     (27.916 docs em lotes de 2k)
    ├── confirmar_todos_98pct.py
    └── LEIAME.txt
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
from typing import Dict, List

import numpy as np
import psycopg
import pymupdf
import torch

BASE_DIR = Path("/home/lucas.ocunha/masters")
DATA_DIR = BASE_DIR / "data"
NPZ_PATH = DATA_DIR / "latent_vectors_full.npz"
META_PATH = DATA_DIR / "latent_vectors_full_metadata.json"
HISTORY_PATH = BASE_DIR / "ml_backend" / "models" / "annotations_history.json"
MODEL_PATH = BASE_DIR / "ml_backend" / "models" / "metric_projection_head.pt"
RENDERED_DIR = DATA_DIR / "rendered_pages"
PDF_DIR = Path("/media/ssd2/datasets/CEAP")
DESKTOP_DIR = Path("/home/lucas.ocunha/Desktop")
TARGET_ROOT = DESKTOP_DIR / "validacao_98pct"
THUMB_DIR = Path.home() / ".cache" / "thumbnails" / "large"

sys.path.append(str(BASE_DIR))
from scripts.train_metric_learning import MetricProjectionNet

RENDERED_DIR.mkdir(parents=True, exist_ok=True)
THUMB_DIR.mkdir(parents=True, exist_ok=True)
TARGET_ROOT.mkdir(parents=True, exist_ok=True)

BATCH_THRESHOLD = 2500
SUB_BATCH_SIZE = 2000


def sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[^\w\s-]", "", name).strip()
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned[:22]


def render_pdf_to_png(pdf_path: Path, png_path: Path, dpi: int = 100) -> bool:
    if png_path.exists() and png_path.stat().st_size > 0:
        return True
    try:
        doc = pymupdf.open(str(pdf_path))
        if len(doc) == 0:
            doc.close()
            return False
        page = doc[0]
        pix = page.get_pixmap(dpi=dpi)
        png_path.parent.mkdir(parents=True, exist_ok=True)
        pix.save(str(png_path))
        doc.close()
        return True
    except Exception:
        return False


def setup_thumbnail(symlink_path: Path, png_path: Path) -> bool:
    try:
        uri = "file://" + urllib.parse.quote(str(symlink_path.absolute()))
        thumb_hash = hashlib.md5(uri.encode("utf-8")).hexdigest()
        thumb_path = THUMB_DIR / f"{thumb_hash}.png"
        if not thumb_path.exists() and png_path.exists():
            thumb_path.symlink_to(png_path.absolute())
        return True
    except Exception:
        return False


def process_single_candidate(item: dict) -> bool:
    doc_id = item["ide_documento"]
    pdf_path = PDF_DIR / f"{doc_id}.pdf"
    png_path = RENDERED_DIR / f"{doc_id}.png"
    symlink_path = item["symlink_path"]

    if not pdf_path.exists():
        return False

    if not render_pdf_to_png(pdf_path, png_path, dpi=100):
        return False

    try:
        if symlink_path.exists() or symlink_path.is_symlink():
            symlink_path.unlink()
        symlink_path.symlink_to(png_path.absolute())
        setup_thumbnail(symlink_path, png_path)
        return True
    except Exception:
        return False


def main():
    t0 = time.time()
    print("==================================================================")
    print(" GERAÇÃO DE PASTAS DE VALIDAÇÃO (73K AMOSTRAS COM >= 98% SIMILARIDADE)")
    print("==================================================================")

    print("\n1. Carregando representações latentes e anotações...")
    npz = np.load(NPZ_PATH)
    dino = npz["dino_doc_embeddings"]
    vlm = npz["vlm_doc_embeddings"]
    combined = np.hstack([dino, vlm]).astype(np.float32)
    norms = np.linalg.norm(combined, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    combined /= norms
    n_docs = len(combined)

    with open(META_PATH, "r", encoding="utf-8") as f:
        docs = json.load(f)["documents"]
    with open(HISTORY_PATH, "r", encoding="utf-8") as f:
        history = json.load(f)

    known_classes = sorted(list(set(history.values())))
    labeled_stems = set(history.keys())
    unmapped_mask = np.array([Path(d["doc_name"]).stem not in labeled_stems for d in docs])
    unmapped_indices = np.where(unmapped_mask)[0]

    print(f"   Total de documentos: {n_docs:,} | Não mapeados: {len(unmapped_indices):,}")

    print("\n2. Projetando no espaço métrico ArcFace 128-D na GPU...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MetricProjectionNet(in_features=512, hidden_dim=256, out_features=128).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    class_centroids = []
    for c in known_classes:
        c_indices = [i for i, d in enumerate(docs) if history.get(Path(d["doc_name"]).stem) == c]
        v_t = torch.from_numpy(combined[c_indices]).float().to(device)
        with torch.no_grad():
            proj = model(v_t).cpu().numpy()
        mean_v = proj.mean(axis=0)
        mean_v /= np.linalg.norm(mean_v)
        class_centroids.append(mean_v)
    class_centroids = np.array(class_centroids)

    batch_size = 65536
    proj_all = np.empty((n_docs, 128), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, n_docs, batch_size):
            e = min(s + batch_size, n_docs)
            out = model(torch.from_numpy(combined[s:e]).to(device))
            proj_all[s:e] = out.cpu().numpy()

    print("\n3. Identificando todos os candidatos não mapeados com similaridade >= 0.98...")
    sims_unmapped_matrix = np.dot(proj_all[unmapped_indices], class_centroids.T)
    max_sims = np.max(sims_unmapped_matrix, axis=1)
    best_classes = np.argmax(sims_unmapped_matrix, axis=1)

    qual_98 = max_sims >= 0.98
    qual_doc_indices = unmapped_indices[qual_98]
    qual_sims = max_sims[qual_98]
    qual_class_ids = best_classes[qual_98]

    selected_by_class: Dict[str, List[dict]] = {c: [] for c in known_classes}
    all_selected_ids = []

    for d_idx, sim_val, c_id in zip(qual_doc_indices, qual_sims, qual_class_ids):
        stem = Path(docs[d_idx]["doc_name"]).stem
        try:
            doc_id = int(stem)
        except ValueError:
            continue
        c_name = known_classes[c_id]
        selected_by_class[c_name].append({
            "ide_documento": doc_id,
            "classe": c_name,
            "similaridade": float(sim_val),
        })
        all_selected_ids.append(doc_id)

    # Ordenar por similaridade decrescente dentro de cada classe
    total_candidates = 0
    for c_name in known_classes:
        selected_by_class[c_name].sort(key=lambda x: -x["similaridade"])
        for r_idx, it in enumerate(selected_by_class[c_name], 1):
            it["rank"] = r_idx
        cnt = len(selected_by_class[c_name])
        total_candidates += cnt
        if cnt > 0:
            print(f"   - {c_name:22s}: {cnt:6,d} docs (simil: {selected_by_class[c_name][0]['similaridade']:.3f} -> {selected_by_class[c_name][-1]['similaridade']:.3f})")

    print(f"\n   Total de candidatos >= 98%: {total_candidates:,}")

    # 4. Consultar Metadados no PostgreSQL em Lote
    print(f"\n4. Consultando metadados fiscais no PostgreSQL para {len(all_selected_ids):,} documentos...")
    meta_db = {}
    try:
        with psycopg.connect("postgresql://postgres:postgres@localhost:5432/ceap") as conn:
            with conn.cursor() as cur:
                chunk_size = 5000
                for i in range(0, len(all_selected_ids), chunk_size):
                    chunk = all_selected_ids[i:i + chunk_size]
                    cur.execute(
                        """
                        SELECT ide_documento, fornecedor_nome, valor_liquido, data_emissao
                        FROM vw_documento_detalhado
                        WHERE ide_documento = ANY(%s)
                        """,
                        (chunk,)
                    )
                    for r in cur.fetchall():
                        meta_db[r[0]] = {
                            "fornecedor": r[1] or "DESCONHECIDO",
                            "valor": float(r[2]) if r[2] is not None else 0.0,
                            "data": str(r[3]) if r[3] else "",
                        }
        print(f"   Metadados encontrados para {len(meta_db):,} documentos.")
    except Exception as e:
        print(f"   [AVISO] Erro consultando banco ({e}). Continuando com nomes padrão.")

    # 5. Estruturar Pastas e Lotes
    print("\n5. Estruturando pastas, sub-lotes e atalhos...")
    tasks_to_render = []

    # Processar primeiro as classes menores para estarem disponíveis imediatamente
    sorted_classes = sorted(known_classes, key=lambda c: len(selected_by_class[c]))

    for c in sorted_classes:
        items = selected_by_class[c]
        if not items:
            continue

        c_dir = TARGET_ROOT / c
        c_dir.mkdir(parents=True, exist_ok=True)

        if len(items) <= BATCH_THRESHOLD:
            # Pasta direta
            for it in items:
                doc_id = it["ide_documento"]
                sim_str = f"{it['similaridade']:.3f}"
                db_info = meta_db.get(doc_id, {"fornecedor": "FORNECEDOR", "valor": 0.0})
                clean_forn = sanitize_filename(db_info["fornecedor"])
                val_str = f"R${db_info['valor']:.2f}".replace(".", ",")
                link_name = f"rank_{it['rank']:05d}_[sim_{sim_str}]_[{clean_forn}]_[{val_str}]_{doc_id}.png"
                it["symlink_path"] = c_dir / link_name
                tasks_to_render.append(it)
        else:
            # Dividir em lotes de SUB_BATCH_SIZE (ex: 2.000)
            n_lotes = (len(items) + SUB_BATCH_SIZE - 1) // SUB_BATCH_SIZE
            for lote_idx in range(n_lotes):
                sub_items = items[lote_idx * SUB_BATCH_SIZE : (lote_idx + 1) * SUB_BATCH_SIZE]
                lote_dir = c_dir / f"lote_{lote_idx+1:02d}_{len(sub_items)}"
                lote_dir.mkdir(parents=True, exist_ok=True)
                for it in sub_items:
                    doc_id = it["ide_documento"]
                    sim_str = f"{it['similaridade']:.3f}"
                    db_info = meta_db.get(doc_id, {"fornecedor": "FORNECEDOR", "valor": 0.0})
                    clean_forn = sanitize_filename(db_info["fornecedor"])
                    val_str = f"R${db_info['valor']:.2f}".replace(".", ",")
                    link_name = f"rank_{it['rank']:05d}_[sim_{sim_str}]_[{clean_forn}]_[{val_str}]_{doc_id}.png"
                    it["symlink_path"] = lote_dir / link_name
                    tasks_to_render.append(it)

        # Criar script de confirmação na pasta da classe
        create_class_confirm_script(c_dir, c)

    create_master_confirm_script(TARGET_ROOT)
    create_readme_file(TARGET_ROOT, selected_by_class)

    # 6. Renderizar em Paralelo com 32 Workers
    print(f"\n6. Renderizando {len(tasks_to_render):,} páginas (DPI 100) com 32 workers...")
    t_render0 = time.time()
    rendered_success = 0
    last_print = time.time()

    with ThreadPoolExecutor(max_workers=32) as executor:
        for idx, res in enumerate(executor.map(process_single_candidate, tasks_to_render), 1):
            if res:
                rendered_success += 1
            if time.time() - last_print >= 10.0 or idx == len(tasks_to_render):
                elapsed = time.time() - t_render0
                speed = idx / elapsed if elapsed > 0 else 0
                eta_s = (len(tasks_to_render) - idx) / speed if speed > 0 else 0
                print(f"   Progresso: {idx:6,d}/{len(tasks_to_render):,d} ({idx/len(tasks_to_render)*100:5.1f}%) | Velocidade: {speed:.1f} pags/s | ETA: {eta_s/60:.1f} min")
                last_print = time.time()

    print(f"\n=== SUCESSO TOTAL EM {time.time()-t0:.1f} SEGUNDOS! ===")
    print(f"Pasta de validação pronta em: {TARGET_ROOT}")


def create_class_confirm_script(class_dir: Path, class_name: str):
    script_content = f"""#!/usr/bin/env python3
\"\"\"
Confirma a validação para a classe '{class_name}'.
Varre todos os arquivos mantidos (na raiz ou em sub-lotes),
adiciona ao annotations_history.json e retreina o ML Backend.
\"\"\"

import json
import re
from pathlib import Path
import requests

CLASS_NAME = "{class_name}"
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = Path("/home/lucas.ocunha/masters")
HISTORY_FILE = ROOT_DIR / "ml_backend" / "models" / "annotations_history.json"
ML_BACKEND_URL = "http://localhost:9090"


def main():
    print(f"=== CONFIRMANDO VALIDAÇÃO PARA CLASSE: {{CLASS_NAME}} ===")
    png_files = list(CURRENT_DIR.rglob("*.png"))
    print(f"Total de documentos mantidos na pasta/lotes: {{len(png_files):,}}")

    if not png_files:
        print("Nenhum arquivo PNG encontrado.")
        return

    doc_ids = []
    for p in png_files:
        m = re.search(r'_(\\d+)\\.png$', p.name)
        if m:
            doc_ids.append(m.group(1))

    print(f"IDs válidos identificados: {{len(doc_ids):,}}")

    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        history = json.load(f)

    added = 0
    for d_id in doc_ids:
        if d_id not in history:
            history[d_id] = CLASS_NAME
            added += 1

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"Adicionadas {{added:,}} novas anotações ao annotations_history.json (Total histórico: {{len(history):,}})")

    print(f"Disparando retreino no ML Backend ({{ML_BACKEND_URL}})...")
    try:
        resp = requests.post(f"{{ML_BACKEND_URL}}/train", timeout=60)
        print("Resposta do ML Backend:", resp.json())
    except Exception as e:
        print(f"Aviso ao contatar ML Backend: {{e}}")

    print(f"=== VALIDAÇÃO DE {{CLASS_NAME}} CONCLUÍDA COM SUCESSO! ===")


if __name__ == "__main__":
    main()
"""
    script_path = class_dir / "confirmar_validacao.py"
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script_content)
    script_path.chmod(0o755)


def create_master_confirm_script(target_root: Path):
    script_content = f"""#!/usr/bin/env python3
\"\"\"
Script Mestre: Varre TODAS as pastas e lotes em validacao_98pct,
atualiza annotations_history.json em lote com todos os arquivos mantidos
e retreina o classificador do ML Backend.
\"\"\"

import json
import re
from pathlib import Path
import requests

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = Path("/home/lucas.ocunha/masters")
HISTORY_FILE = ROOT_DIR / "ml_backend" / "models" / "annotations_history.json"
ML_BACKEND_URL = "http://localhost:9090"


def main():
    print("=== CONFIRMAÇÃO MESTRE DE TODAS AS CLASSES (>= 98%) ===")
    
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        history = json.load(f)

    initial_count = len(history)

    for class_folder in sorted(CURRENT_DIR.iterdir()):
        if not class_folder.is_dir() or class_folder.name.startswith("."):
            continue

        class_name = class_folder.name
        png_files = list(class_folder.rglob("*.png"))
        if not png_files:
            continue

        class_added = 0
        for p in png_files:
            m = re.search(r'_(\\d+)\\.png$', p.name)
            if m:
                d_id = m.group(1)
                if d_id not in history:
                    history[d_id] = class_name
                    class_added += 1

        print(f"  - {{class_name:22s}}: {{len(png_files):6,d}} mantidos (+{{class_added:6,d}} novos)")

    total_added = len(history) - initial_count
    print(f"\\nTotal de novos rótulos a incorporar: {{total_added:,}}")

    if total_added > 0:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        print(f"annotations_history.json atualizado! (Total: {{len(history):,}})")

        print(f"Disparando retreino no ML Backend ({{ML_BACKEND_URL}})...")
        try:
            resp = requests.post(f"{{ML_BACKEND_URL}}/train", timeout=90)
            print("Resposta do ML Backend:", resp.json())
        except Exception as e:
            print(f"Aviso ao contatar ML Backend: {{e}}")
    else:
        print("Nenhum novo documento para adicionar.")

    print("=== PROCESSO FINALIZADO COM SUCESSO! ===")


if __name__ == "__main__":
    main()
"""
    script_path = target_root / "confirmar_todos_98pct.py"
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script_content)
    script_path.chmod(0o755)


def create_readme_file(target_root: Path, selected_by_class: Dict[str, List[dict]]):
    lines = [
        "# Validação Visual Massiva de Alta Certeza (Similaridade >= 98% no ArcFace)",
        "",
        "Esta pasta contém todos os 73.731 documentos do CEAP que apresentam similaridade >= 98%",
        "com os centróides aprendidos pelo modelo ArcFace Metric Learning.",
        "",
        "Para evitar travamentos no Nautilus/Gerenciador de Arquivos, classes com mais de 2.500",
        "documentos foram automaticamente particionadas em sub-lotes de ~2.000 imagens.",
        "",
        "## Quantidades por Classe e Organização:",
        "",
        "| Classe | Quantidade (>= 98%) | Estrutura |",
        "| :--- | :---: | :--- |",
    ]
    for c, items in sorted(selected_by_class.items(), key=lambda x: -len(x[1])):
        if len(items) == 0:
            continue
        if len(items) <= BATCH_THRESHOLD:
            desc = "Pasta direta"
        else:
            n_lotes = (len(items) + SUB_BATCH_SIZE - 1) // SUB_BATCH_SIZE
            desc = f"{n_lotes} lotes de ~2.000"
        lines.append(f"| `{c}` | **{len(items):,d}** | {desc} |")

    lines.extend([
        "",
        "## Como Validar:",
        "1. Abra cada pasta/lote no gerenciador de arquivos (Nautilus). Miniaturas em alta velocidade.",
        "2. Os arquivos estão nomeados do mais similar para o menos similar:",
        "   `rank_00001_[sim_0.995]_[FORNECEDOR]_[VALOR]_[DOC_ID].png`",
        "3. Se vir algum arquivo que não pertença à classe, basta apagá-lo.",
        "4. Para confirmar:",
        "   - **Por classe:** entre na pasta da classe e execute `python3 confirmar_validacao.py`",
        "   - **Tudo de uma vez:** execute nesta pasta `python3 confirmar_todos_98pct.py`",
    ])

    with open(target_root / "LEIAME.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
