#!/usr/bin/env python3
"""
Gera pastas de validação no Desktop para TODAS as 14 classes do CEAP
com até 2.000 amostras por classe com similaridade >= 0.90 no espaço métrico (ArcFace 128-D).

Estrutura gerada:
  Desktop/validacao_90pct/
    ├── agua/                     (até 2.000 docs)
    ├── emails/                   (2 docs)
    ├── energia/                  (até 2.000 docs)
    ├── fatura/                   (448 docs)
    ├── iptu/                     (760 docs)
    ├── nfc/                      (até 2.000 docs)
    ├── nfe/                      (até 2.000 docs)
    ├── nfse/                     (até 2.000 docs)
    ├── passagem_aerea/           (até 2.000 docs)
    ├── passagem_rodoviaria/      (1.379 docs)
    ├── pedagio/                  (1.627 docs)
    ├── recibo/                   (até 2.000 docs)
    ├── telefonia/                (até 2.000 docs)
    ├── uber/                     (até 2.000 docs)
    ├── confirmar_todas_classes.py
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
from typing import Dict, List, Tuple

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
TARGET_ROOT = DESKTOP_DIR / "validacao_90pct"
THUMB_DIR = Path.home() / ".cache" / "thumbnails" / "large"

sys.path.append(str(BASE_DIR))
from scripts.train_metric_learning import MetricProjectionNet

RENDERED_DIR.mkdir(parents=True, exist_ok=True)
THUMB_DIR.mkdir(parents=True, exist_ok=True)
TARGET_ROOT.mkdir(parents=True, exist_ok=True)


def sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[^\w\s-]", "", name).strip()
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned[:25]


def render_pdf_to_png(pdf_path: Path, png_path: Path, dpi: int = 120) -> bool:
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

    if not render_pdf_to_png(pdf_path, png_path):
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
    print(" GERAÇÃO DE PASTAS DE VALIDAÇÃO (>= 90% SIMILARIDADE NO ARCFACE)")
    print("==================================================================")

    # 1. Carregar embeddings e metadados
    print("\n1. Carregando representações latentes...")
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
    print(f"   Total de documentos: {n_docs:,} | Rótulos conhecidos: {len(labeled_stems):,} em {len(known_classes)} classes")

    # 2. Carregar Modelo Metric Learning e Projetar
    print("\n2. Projetando no espaço métrico ArcFace 128-D...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MetricProjectionNet(in_features=512, hidden_dim=256, out_features=128).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    # Centróides das classes
    class_centroids = {}
    with torch.no_grad():
        for c in known_classes:
            c_indices = [i for i, d in enumerate(docs) if history.get(Path(d["doc_name"]).stem) == c]
            if c_indices:
                v_t = torch.from_numpy(combined[c_indices]).float().to(device)
                proj = model(v_t).cpu().numpy()
                mean_v = proj.mean(axis=0)
                mean_v /= np.linalg.norm(mean_v)
                class_centroids[c] = mean_v

    # Projeção em lotes de todos os 1.34M
    batch_size = 65536
    proj_all = np.empty((n_docs, 128), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, n_docs, batch_size):
            e = min(s + batch_size, n_docs)
            out = model(torch.from_numpy(combined[s:e]).to(device))
            proj_all[s:e] = out.cpu().numpy()

    # 3. Filtrar candidatos não mapeados com similaridade >= 0.90
    print("\n3. Selecionando até 2.000 amostras mais similares por classe...")
    unmapped_mask = np.array([Path(d["doc_name"]).stem not in labeled_stems for d in docs])
    unmapped_indices = np.where(unmapped_mask)[0]

    selected_by_class: Dict[str, List[dict]] = {}
    all_selected_ids = []

    for c, cent in class_centroids.items():
        sims_unmapped = np.dot(proj_all[unmapped_indices], cent)
        qual_mask = sims_unmapped >= 0.90
        qual_sub_idx = np.where(qual_mask)[0]
        qual_sims = sims_unmapped[qual_sub_idx]

        # Ordenar do mais similar (maior confiança) para o menor
        sorted_order = np.argsort(-qual_sims)[:2000]
        final_doc_indices = unmapped_indices[qual_sub_idx[sorted_order]]
        final_sims = qual_sims[sorted_order]

        c_items = []
        for r_idx, (d_idx, sim_val) in enumerate(zip(final_doc_indices, final_sims), 1):
            stem = Path(docs[d_idx]["doc_name"]).stem
            try:
                doc_id = int(stem)
            except ValueError:
                continue
            c_items.append({
                "rank": r_idx,
                "ide_documento": doc_id,
                "classe": c,
                "similaridade": float(sim_val),
            })
            all_selected_ids.append(doc_id)

        selected_by_class[c] = c_items
        print(f"   - {c:22s}: {len(c_items):5,d} selecionados (simil. máx: {c_items[0]['similaridade']:.3f} | mín: {c_items[-1]['similaridade']:.3f})" if c_items else f"   - {c:22s}: 0 selecionados")

    total_candidates = sum(len(v) for v in selected_by_class.values())
    print(f"\n   Total consolidado de candidatos para validação: {total_candidates:,}")

    # 4. Consultar Metadados no PostgreSQL em Lote
    print(f"\n4. Consultando metadados fiscais no PostgreSQL para {len(all_selected_ids):,} documentos...")
    meta_db = {}
    try:
        with psycopg.connect("postgresql://postgres:postgres@localhost:5432/ceap") as conn:
            with conn.cursor() as cur:
                # Query em chunks de 5.000 para não estourar parâmetros
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

    # 5. Criar Pastas e Links Simbólicos
    print("\n5. Renderizando páginas e criando atalhos com miniaturas...")
    tasks_to_render = []

    for c, items in selected_by_class.items():
        c_dir = TARGET_ROOT / c
        c_dir.mkdir(parents=True, exist_ok=True)

        for item in items:
            doc_id = item["ide_documento"]
            sim_str = f"{item['similaridade']:.3f}"
            db_info = meta_db.get(doc_id, {"fornecedor": "FORNECEDOR", "valor": 0.0})
            clean_forn = sanitize_filename(db_info["fornecedor"])
            val_str = f"R${db_info['valor']:.2f}".replace(".", ",")
            
            # Formato do nome do arquivo
            link_name = f"rank_{item['rank']:04d}_[sim_{sim_str}]_[{clean_forn}]_[{val_str}]_{doc_id}.png"
            item["symlink_path"] = c_dir / link_name
            tasks_to_render.append(item)

        # Criar script de confirmação específico da classe
        create_class_confirm_script(c_dir, c)

    # Executar renderização em paralelo com 32 threads
    print(f"   Processando {len(tasks_to_render):,} páginas com 32 workers...")
    t_render0 = time.time()
    rendered_success = 0
    with ThreadPoolExecutor(max_workers=32) as executor:
        for res in executor.map(process_single_candidate, tasks_to_render):
            if res:
                rendered_success += 1

    print(f"   Renderização e symlinks concluídos em {time.time()-t_render0:.1f}s ({rendered_success:,}/{len(tasks_to_render):,} OK)")

    # 6. Salvar manifesto geral e script de confirmação mestre
    manifest_file = TARGET_ROOT / ".manifest_candidatos.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        # Serializar tirando symlink_path que é Path
        serializable = {
            c: [
                {k: v for k, v in it.items() if k != "symlink_path"}
                for it in items
            ]
            for c, items in selected_by_class.items()
        }
        json.dump(serializable, f, indent=2, ensure_ascii=False)

    create_master_confirm_script(TARGET_ROOT, known_classes)
    create_readme_file(TARGET_ROOT, selected_by_class)

    print(f"\n=== SUCESSO TOTAL EM {time.time()-t0:.1f} SEGUNDOS! ===")
    print(f"Pasta de validação disponível em: {TARGET_ROOT}")


def create_class_confirm_script(class_dir: Path, class_name: str):
    script_content = f"""#!/usr/bin/env python3
\"\"\"
Confirma a validação dos documentos mantidos nesta pasta para a classe '{class_name}'.
Adiciona todos os documentos restantes em annotations_history.json e retreina o ML Backend.
\"\"\"

import json
import re
from pathlib import Path
import psycopg
import requests

CLASS_NAME = "{class_name}"
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = Path("/home/lucas.ocunha/masters")
HISTORY_FILE = ROOT_DIR / "ml_backend" / "models" / "annotations_history.json"
ML_BACKEND_URL = "http://localhost:9090"


def main():
    print(f"=== CONFIRMANDO VALIDAÇÃO PARA CLASSE: {{CLASS_NAME}} ===")
    png_files = list(CURRENT_DIR.glob("*.png"))
    print(f"Documentos mantidos na pasta: {{len(png_files)}}")

    if not png_files:
        print("Nenhum arquivo PNG encontrado na pasta.")
        return

    doc_ids = []
    for p in png_files:
        m = re.search(r'_(\\d+)\\.png$', p.name)
        if m:
            doc_ids.append(m.group(1))

    print(f"IDs válidos identificados: {{len(doc_ids)}}")

    # 1. Atualizar annotations_history.json
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        history = json.load(f)

    added = 0
    for d_id in doc_ids:
        if d_id not in history:
            history[d_id] = CLASS_NAME
            added += 1

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"Adicionadas {{added}} novas anotações ao annotations_history.json (Total histórico: {{len(history)}})")

    # 2. Notificar ML Backend para Retreino
    print(f"Disparando retreino no ML Backend ({{ML_BACKEND_URL}})...")
    try:
        resp = requests.post(f"{{ML_BACKEND_URL}}/train", timeout=30)
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


def create_master_confirm_script(target_root: Path, known_classes: List[str]):
    script_content = f"""#!/usr/bin/env python3
\"\"\"
Script Mestre de Confirmação:
Varre todas as subpastas em validacao_90pct, coleta os arquivos mantidos
para cada classe, atualiza annotations_history.json em lote e retreina o ML Backend.
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
    print("=== CONFIRMAÇÃO MESTRE DE TODAS AS SUBPASTAS DE VALIDAÇÃO (>= 90%) ===")
    
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        history = json.load(f)

    initial_count = len(history)
    added_by_class = {{}}

    for subfolder in sorted(CURRENT_DIR.iterdir()):
        if not subfolder.is_dir() or subfolder.name.startswith("."):
            continue

        class_name = subfolder.name
        png_files = list(subfolder.glob("*.png"))
        if not png_files:
            continue

        doc_ids = []
        for p in png_files:
            m = re.search(r'_(\\d+)\\.png$', p.name)
            if m:
                doc_ids.append(m.group(1))

        class_added = 0
        for d_id in doc_ids:
            if d_id not in history:
                history[d_id] = class_name
                class_added += 1

        added_by_class[class_name] = {{
            "mantidos": len(png_files),
            "novos_rotulos": class_added
        }}
        print(f"  - {{class_name:22s}}: {{len(png_files):5d}} mantidos (+{{class_added:5d}} novos)")

    total_added = len(history) - initial_count
    print(f"\\nTotal de novos rótulos a incorporar: {{total_added:,}}")

    if total_added > 0:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        print(f"annotations_history.json atualizado com sucesso! (Total: {{len(history):,}})")

        print(f"Disparando retreino no ML Backend ({{ML_BACKEND_URL}})...")
        try:
            resp = requests.post(f"{{ML_BACKEND_URL}}/train", timeout=60)
            print("Resposta do ML Backend:", resp.json())
        except Exception as e:
            print(f"Aviso ao contatar ML Backend: {{e}}")
    else:
        print("Nenhum novo documento para adicionar.")

    print("=== PROCESSO FINALIZADO COM SUCESSO! ===")


if __name__ == "__main__":
    main()
"""
    script_path = target_root / "confirmar_todas_classes.py"
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script_content)
    script_path.chmod(0o755)


def create_readme_file(target_root: Path, selected_by_class: Dict[str, List[dict]]):
    lines = [
        "# Validação Visual de Amostras de Alta Confiança (Similaridade >= 90% no ArcFace)",
        "",
        "Esta pasta contém subpastas para as 14 classes ativas do CEAP, com até 2.000 amostras",
        "de documentos que ainda NÃO estavam mapeados, ordenados por similaridade com o centróide da classe.",
        "",
        "## Subpastas e Quantidades Geradas:",
        "",
        "| Classe | Amostras Selecionadas (>= 90%) |",
        "| :--- | :---: |",
    ]
    for c, items in sorted(selected_by_class.items(), key=lambda x: -len(x[1])):
        lines.append(f"| `{c}` | **{len(items):,d}** |")

    lines.extend([
        "",
        "## Como Validar:",
        "1. Abra as subpastas no Gerenciador de Arquivos (Nautilus). As miniaturas já estão pré-geradas no cache.",
        "2. Os arquivos estão nomeados como:",
        "   `rank_0001_[sim_0.985]_[NOME_FORNECEDOR]_[VALOR]_[ID_DOCUMENTO].png`",
        "3. **Remova** (delete ou mova para lixeira) quaisquer arquivos que forem falsos positivos ou não pertencerem à classe.",
        "4. Para confirmar:",
        "   - **Por classe individual:** entre na pasta da classe e execute `python3 confirmar_validacao.py`",
        "   - **Todas as classes de uma vez:** execute nesta pasta `python3 confirmar_todas_classes.py`",
        "5. O sistema atualizará automaticamente `annotations_history.json` e disparará o retreino do ML Backend.",
    ])

    readme_path = target_root / "LEIAME.txt"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
