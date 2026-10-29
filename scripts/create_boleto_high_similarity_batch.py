#!/home/lucas.ocunha/masters/.venv_ocr/bin/python3
"""
Gera lote de validação de Boletos Bancários com Alta Similaridade baseados no novo centróide
e nas 58 sementes reais confirmadas pelo usuário em:
  - masters/data/validacao_batches/validacao_boleto_1000/CONFIRMADOS/ (22 docs)
  - masters/data/validacao_batches/validacao_boleto_alta_similaridade/CONFIRMADOS/ (36 docs)

Organiza os candidatos em 3 subpastas estruturadas:
  1_quase_identicos_sementes_98_99pct/  (>= 98% a alguma semente direta)
  2_centroide_97pct/                    (>= 97% em relação ao novo centróide)
  3_centroide_96pct/                    (>= 96% em relação ao novo centróide, top 450)
  CONFIRMADOS/                          (pasta com os 36 boletos já confirmados preservados)
  manifest.json
  confirmar_validacao.py
"""

import os
import sys
import re
import json
import time
import shutil
import hashlib
import urllib.parse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pymupdf
import cv2

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
cv2.setNumThreads(1)

THUMB_LARGE_DIR = Path.home() / ".cache" / "thumbnails" / "large"
THUMB_NORMAL_DIR = Path.home() / ".cache" / "thumbnails" / "normal"
THUMB_LARGE_DIR.mkdir(parents=True, exist_ok=True)
THUMB_NORMAL_DIR.mkdir(parents=True, exist_ok=True)


def render_pdf_page(task):
    stem, pdf_p_str, png_p_str = task
    try:
        png_p = Path(png_p_str)
        pdf_p = Path(pdf_p_str)
        if not png_p.exists():
            if not pdf_p.exists():
                return False
            doc = pymupdf.open(str(pdf_p))
            if len(doc) == 0:
                doc.close()
                return False
            page = doc[0]
            pix = page.get_pixmap(dpi=110, alpha=False)
            png_p.parent.mkdir(parents=True, exist_ok=True)
            pix.save(str(png_p))
            doc.close()
        return True
    except Exception:
        return False


def render_freedesktop_thumb(png_file: Path) -> bool:
    try:
        uri = "file://" + urllib.parse.quote(str(png_file.resolve()))
        h = hashlib.md5(uri.encode("utf-8")).hexdigest()
        thumb_p = THUMB_LARGE_DIR / f"{h}.png"
        if thumb_p.exists():
            return True
        img = cv2.imread(str(png_file))
        if img is None:
            return False
        h_orig, w_orig = img.shape[:2]
        scale = 256.0 / max(h_orig, w_orig)
        resized = cv2.resize(img, (int(w_orig * scale), int(h_orig * scale)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(thumb_p), resized)
        return True
    except Exception:
        return False


def main():
    t0 = time.time()
    base_dir = Path("/home/lucas.ocunha/masters")
    data_dir = base_dir / "data"
    pdf_dir = Path("/media/ssd2/datasets/CEAP")
    rendered_dir = data_dir / "rendered_pages"
    rendered_dir.mkdir(parents=True, exist_ok=True)

    dest_dir = data_dir / "validacao_batches" / "validacao_boleto_alta_similaridade"
    conf_dir1 = data_dir / "validacao_batches" / "validacao_boleto_1000" / "CONFIRMADOS"
    conf_dir2 = dest_dir / "CONFIRMADOS"

    print("==================================================================")
    print(" REGENERAÇÃO DE CANDIDATOS A BOLETO (NOVO CENTRÓIDE DE 58 SEMENTES)")
    print("==================================================================")

    # 1. Carregar todas as 58 sementes validadas
    stems1 = [f.stem for f in conf_dir1.glob("*.png")] if conf_dir1.exists() else []
    stems2 = [f.stem for f in conf_dir2.glob("*.png")] if conf_dir2.exists() else []
    conf_stems = sorted(list(set(stems1 + stems2)))
    conf_stems_set = set(conf_stems)
    print(f"1. Sementes confirmadas agregadas: {len(conf_stems)} boletos 100% validados")

    # 2. Carregar metadados e DINOv2
    print("2. Carregando representações DINOv2 e metadados de documentos...")
    meta_p = data_dir / "latent_vectors_full_metadata.json"
    with open(meta_p, "r", encoding="utf-8") as f:
        meta = json.load(f)
    docs = meta["documents"]
    n_docs = len(docs)
    stem_to_idx = {Path(d["doc_name"]).stem: i for i, d in enumerate(docs)}

    indices_conf = [stem_to_idx[s] for s in conf_stems if s in stem_to_idx]

    lookup_p = data_dir / "pdf_metadata_lookup.json"
    with open(lookup_p, "r", encoding="utf-8") as f:
        lookup = json.load(f)

    history_p = base_dir / "ml_backend" / "models" / "annotations_history.json"
    with open(history_p, "r", encoding="utf-8") as f:
        history = json.load(f)

    npz = np.load(data_dir / "latent_vectors_full.npz", mmap_mode="r")
    dino = npz["dino_doc_embeddings"]

    # 3. Calcular Novo Centróide de 58 sementes
    d_conf = dino[indices_conf].astype(np.float32)
    d_conf /= np.linalg.norm(d_conf, axis=1, keepdims=True)
    centroid = d_conf.mean(axis=0)
    centroid /= np.linalg.norm(centroid)

    # 4. Similaridade global em chunks
    print("3. Calculando similaridades com novo centróide e sementes individuais...")
    chunk_size = 100000
    sims_centroid = np.zeros(n_docs, dtype=np.float32)
    sims_max_seed = np.zeros(n_docs, dtype=np.float32)
    best_seed_idx = np.zeros(n_docs, dtype=np.int32)

    for start in range(0, n_docs, chunk_size):
        end = min(start + chunk_size, n_docs)
        sub_d = dino[start:end].astype(np.float32)
        norms = np.linalg.norm(sub_d, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        sub_d /= norms

        sims_centroid[start:end] = np.dot(sub_d, centroid)
        all_sims = np.dot(sub_d, d_conf.T)
        sims_max_seed[start:end] = np.max(all_sims, axis=1)
        best_seed_idx[start:end] = np.argmax(all_sims, axis=1)

    # 5. Filtrar e classificar candidatos nas 3 faixas
    tier1 = []  # >= 0.98 sementes
    tier2 = []  # >= 0.97 centróide (excluindo tier1)
    tier3 = []  # >= 0.96 centróide (excluindo tier1 e tier2)

    # Bloquear fornecedores de pedágio/tags
    blocked_forn = ["SEM PARAR", "CGMP", "CONECTCAR", "VELOE", "MOVE MAIS", "AUTOPISTA", "RODOVIA", "CCR"]

    for i in range(n_docs):
        dname = docs[i]["doc_name"]
        stem = Path(dname).stem
        if stem in conf_stems_set:
            continue

        # Excluir documentos marcados expressamente com outra classe definitiva
        ann_cls = history.get(stem)
        if ann_cls and ann_cls not in ["boleto"]:
            continue

        info = lookup.get(dname, {})
        forn = info.get("fornecedor", "").upper()
        subc = info.get("subcota", "").upper()

        if any(b in forn for b in blocked_forn) or "PEDÁGIO" in subc or "PEDAGIO" in subc:
            continue

        sim_c = float(sims_centroid[i])
        sim_s = float(sims_max_seed[i])
        s_stem = conf_stems[best_seed_idx[i]]

        if sim_s >= 0.98:
            tier1.append((i, stem, dname, sim_s, sim_c, s_stem, forn, subc))
        elif sim_c >= 0.97:
            tier2.append((i, stem, dname, sim_s, sim_c, s_stem, forn, subc))
        elif sim_c >= 0.96:
            tier3.append((i, stem, dname, sim_s, sim_c, s_stem, forn, subc))

    tier1.sort(key=lambda x: -x[3])
    tier2.sort(key=lambda x: -x[4])
    tier3.sort(key=lambda x: -x[4])

    # Limitar tier3 para manter o lote ágil para o usuário
    tier3 = tier3[:450]

    print(f"\nCandidatos identificados:")
    print(f"  • Tier 1 (>= 98% a sementes confirmadas): {len(tier1)} documentos")
    print(f"  • Tier 2 (>= 97% ao novo centróide):     {len(tier2)} documentos")
    print(f"  • Tier 3 (>= 96% ao novo centróide):     {len(tier3)} documentos (top 450)")
    total_selected = len(tier1) + len(tier2) + len(tier3)
    print(f"  Total selecionado para conferência:       {total_selected} documentos")

    # 6. Preparar pastas de destino (limpar apenas as pastas de candidatos, preservar CONFIRMADOS)
    dest_tier1 = dest_dir / "1_quase_identicos_sementes_98_99pct"
    dest_tier2 = dest_dir / "2_centroide_97pct"
    dest_tier3 = dest_dir / "3_centroide_96pct"
    dest_confirmados = dest_dir / "CONFIRMADOS"

    for d in [dest_tier1, dest_tier2, dest_tier3]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    dest_confirmados.mkdir(parents=True, exist_ok=True)

    # 7. Renderização rápida paralela via ThreadPoolExecutor
    all_candidates = []
    for item in tier1:
        all_candidates.append((dest_tier1, "tier1_semente_98pct", item))
    for item in tier2:
        all_candidates.append((dest_tier2, "tier2_centroide_97pct", item))
    for item in tier3:
        all_candidates.append((dest_tier3, "tier3_centroide_96pct", item))

    print(f"\n4. Renderizando páginas em PNG (110 DPI) para {total_selected} docs...")
    render_tasks = []
    for target_dir, tier_tag, item in all_candidates:
        idx, stem, dname, sim_s, sim_c, s_stem, forn, subc = item
        pdf_path = pdf_dir / dname
        png_path = rendered_dir / f"{stem}.png"
        render_tasks.append((stem, str(pdf_path), str(png_path)))

    t_render_start = time.time()
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(render_pdf_page, render_tasks))
    print(f"   Renderização concluída em {time.time()-t_render_start:.1f}s!")

    # 8. Criar Hardlinks e compor manifesto
    manifest = []
    png_links_created = []

    for target_dir, tier_tag, item in all_candidates:
        idx, stem, dname, sim_s, sim_c, s_stem, forn, subc = item
        src_png = rendered_dir / f"{stem}.png"
        dst_png = target_dir / f"{stem}.png"

        if src_png.exists() and not dst_png.exists():
            os.link(src_png, dst_png)
            png_links_created.append(dst_png)

        info = lookup.get(dname, {})
        manifest.append({
            "doc_id": stem,
            "doc_name": dname,
            "tier": tier_tag,
            "folder": target_dir.name,
            "sim_seed": round(sim_s, 4),
            "nearest_seed": s_stem,
            "sim_centroid": round(sim_c, 4),
            "fornecedor": info.get("fornecedor", ""),
            "valor_documento": info.get("valor_documento", ""),
            "data_emissao": info.get("data_emissao", ""),
            "subcota": info.get("subcota", ""),
        })

    with open(dest_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # 9. Pré-renderizar miniaturas do FreeDesktop para Nautilus / Thunar
    print(f"5. Gerando miniaturas FreeDesktop para carregamento instantâneo...")
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(render_freedesktop_thumb, png_links_created))

    # 10. Atualizar script confirmar_validacao.py
    confirm_script_code = """#!/usr/bin/env python3
\"\"\"
Confirma os boletos validados pelo usuário nesta pasta e os integra ao annotations_history.json.
\"\"\"
import json
import sys
from pathlib import Path

BASE_DIR = Path("/home/lucas.ocunha/masters")
HISTORY_FILE = BASE_DIR / "ml_backend" / "models" / "annotations_history.json"
CURRENT_DIR = Path(__file__).resolve().parent
CONFIRMADOS_DIR = CURRENT_DIR / "CONFIRMADOS"

def main():
    print("=== CONFIRMAÇÃO DO LOTE DE BOLETOS DE ALTA SIMILARIDADE ===")
    
    include_all = "-a" in sys.argv
    if include_all:
        png_files = list(CURRENT_DIR.rglob("*.png"))
    else:
        png_files = list(CONFIRMADOS_DIR.glob("*.png"))
        if not png_files:
            print(f"Nenhum arquivo PNG encontrado na pasta: {CONFIRMADOS_DIR.name}")
            print("Dica: mova os boletos aprovados para a pasta CONFIRMADOS/ ou use -a para confirmar todos.")
            return

    stems = sorted(list(set(p.stem for p in png_files)))
    print(f"Total de boletos a confirmar: {len(stems)}")

    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        history = json.load(f)

    added = 0
    already = 0
    for s in stems:
        if s not in history:
            history[s] = "boleto"
            added += 1
        elif history[s] == "boleto":
            already += 1
        else:
            print(f"  Aviso: doc {s} estava como '{history[s]}', atualizando para 'boleto'...")
            history[s] = "boleto"
            added += 1

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    print(f"\\n[SUCESSO] {added} novos boletos confirmados e salvos em annotations_history.json!")
    print(f"Total atualizado no histórico: {len(history):,} anotações.")

if __name__ == "__main__":
    main()
"""
    confirm_path = dest_dir / "confirmar_validacao.py"
    with open(confirm_path, "w", encoding="utf-8") as f:
        f.write(confirm_script_code)
    confirm_path.chmod(0o755)

    print("\n" + "=" * 66)
    print("   NOVO LOTE DE BOLETOS REGERADO COM SUCESSO!                  ")
    print("=" * 66)
    print(f"Destino: {dest_dir}")
    print(f"  ├── 1_quase_identicos_sementes_98_99pct: {len(tier1)} imagens")
    print(f"  ├── 2_centroide_97pct:                   {len(tier2)} imagens")
    print(f"  ├── 3_centroide_96pct:                   {len(tier3)} imagens")
    print(f"  ├── CONFIRMADOS/                         ({len(stems2)} boletos já confirmados preservados)")
    print(f"  ├── manifest.json                        ({len(manifest)} itens)")
    print(f"  └── confirmar_validacao.py               (script de confirmação automática)")
    print(f"Tempo total: {time.time()-t0:.1f} segundos\n")


if __name__ == "__main__":
    main()
