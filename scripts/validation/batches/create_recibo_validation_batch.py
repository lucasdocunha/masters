#!/home/lucas.ocunha/masters/.venv_ocr/bin/python3
"""
Criação do Lote Massivo de Validação da Classe RECIBO (42.518 Documentos >= 97% ArcFace).
Divide em 2 sub-clusters:
  1. Táxi / Refeição / Alimentação / Pequenos Comprovantes (~32.8k)
  2. Imóveis / Locação / Condomínio / Manutenção de Escritório (~9.7k)

Organiza em sub-lotes para fluidez no Thunar/Nautilus e pré-gera miniaturas no cache do sistema.
"""

import os
import sys
import re
import json
import time
import hashlib
import urllib.parse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from collections import Counter
import pymupdf
import numpy as np
from sklearn.cluster import KMeans

BASE_DIR = Path("/home/lucas.ocunha/masters")
DATA_DIR = BASE_DIR / "data"
PDF_DIR = Path("/media/ssd2/datasets/CEAP")
RENDERED_DIR = DATA_DIR / "rendered_pages"
DESKTOP_DIR = Path("/home/lucas.ocunha/Desktop")
EMBS_PATH = DATA_DIR / "latent_vectors_metric_learned_128d.npy"
META_PATH = DATA_DIR / "latent_vectors_full_metadata.json"
HIST_PATH = DATA_DIR / "models" / "annotations_history.json"
LOOKUP_FILE = DATA_DIR / "pdf_metadata_lookup.json"
TARGET_DIR = DATA_DIR / "validacao_batches" / "validacao_recibo"
DESKTOP_LINK = DESKTOP_DIR / "validacao_recibo"
THUMB_DIR = Path.home() / ".cache" / "thumbnails" / "large"

RENDERED_DIR.mkdir(parents=True, exist_ok=True)
TARGET_DIR.mkdir(parents=True, exist_ok=True)
THUMB_DIR.mkdir(parents=True, exist_ok=True)


def sanitize_str(val: str, max_len: int = 20) -> str:
    cleaned = re.sub(r"[^\w\s-]", "", str(val or "")).strip()
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned[:max_len] or "SEM_NOME"


def render_pdf_worker(task):
    doc_id, pdf_path_str, png_path_str = task
    png_p = Path(png_path_str)
    if png_p.exists() and png_p.stat().st_size > 0:
        return True
    try:
        pdf_p = Path(pdf_path_str)
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


def setup_thumbnail_fast(task):
    symlink_path_str, png_path_str = task
    try:
        sym_p = Path(symlink_path_str)
        png_p = Path(png_path_str)
        uri = "file://" + urllib.parse.quote(str(sym_p.absolute()))
        thumb_hash = hashlib.md5(uri.encode("utf-8")).hexdigest()
        thumb_path = THUMB_DIR / f"{thumb_hash}.png"
        if not thumb_path.exists() and png_p.exists():
            thumb_path.symlink_to(png_p.absolute())
        return True
    except Exception:
        return False


def make_cluster_confirm_script(folder_name: str, cluster_label: str) -> str:
    return f"""#!/home/lucas.ocunha/masters/.venv_ocr/bin/python3
\"\"\"
Script de Confirmação para {cluster_label}.
Varre todos os arquivos mantidos nesta pasta e sub-lotes,
e os registra em annotations_history.json como 'recibo'.
\"\"\"

import os
import re
import sys
import json
from pathlib import Path

BASE_DIR = Path("/home/lucas.ocunha/masters")
HISTORY_FILE = BASE_DIR / "data" / "models" / "annotations_history.json"
CURRENT_DIR = Path(__file__).resolve().parent

def main():
    print("=" * 70)
    print(" CONFIRMAÇÃO DE RECIBOS: {cluster_label}")
    print("=" * 70)

    png_files = list(CURRENT_DIR.rglob("*.png"))
    stems = set()
    for p in png_files:
        m = re.search(r'(\\d{{7,8}})\\.png$', p.name)
        if m:
            stems.add(m.group(1))

    print(f"Arquivos PNG encontrados na pasta: {{len(png_files):,}}")
    print(f"Documentos únicos identificados  : {{len(stems):,}}")

    if not stems:
        print("Nenhum documento encontrado.")
        return

    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        history = json.load(f)

    prev_total = len(history)
    prev_recibo = sum(1 for v in history.values() if v == "recibo")

    added = 0
    for s in stems:
        if s not in history or history[s] != "recibo":
            history[s] = "recibo"
            added += 1

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    new_total = len(history)
    new_recibo = sum(1 for v in history.values() if v == "recibo")

    print(f"\\n✓ Novos recibos incorporados: {{added:,}}")
    print(f"✓ Total anterior de recibo   : {{prev_recibo:,}}")
    print(f"✓ Novo total de recibo       : {{new_recibo:,}}")
    print(f"✓ Total geral de anotações   : {{new_total:,}}")
    print("=" * 70)
    print(" Confirmação concluída com sucesso!")
    print("=" * 70)

if __name__ == "__main__":
    main()
"""


def make_master_confirm_script() -> str:
    return r"""#!/home/lucas.ocunha/masters/.venv_ocr/bin/python3
""" + """\"\"\"
Script Mestre de Confirmação:
Varre AMBOS os clusters de recibo (táxi/alimentação e imóveis/locação),
registra todos os documentos mantidos em annotations_history.json como 'recibo'
e sincroniza com o banco de dados PostgreSQL.
\"\"\"

import os
import re
import sys
import json
from pathlib import Path
import psycopg

BASE_DIR = Path("/home/lucas.ocunha/masters")
HISTORY_FILE = BASE_DIR / "data" / "models" / "annotations_history.json"
YEAR_FILE = BASE_DIR / "data" / "pdf_to_year.json"
CURRENT_DIR = Path(__file__).resolve().parent
DB_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/ceap")

def main():
    print("=" * 72)
    print(" CONFIRMAÇÃO MESTRE: TODOS OS RECIBOS MANTIDOS NAS PASTAS")
    print("=" * 72)

    png_files = list(CURRENT_DIR.rglob("*.png"))
    stems = set()
    for p in png_files:
        m = re.search(r'(\\d{7,8})\\.png$', p.name)
        if m:
            stems.add(m.group(1))

    print(f"Arquivos PNG mantidos pelo usuário : {len(png_files):,}")
    print(f"Total de documentos únicos         : {len(stems):,}")

    if not stems:
        print("Nenhum arquivo PNG encontrado.")
        return

    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        history = json.load(f)

    prev_total = len(history)
    prev_recibo = sum(1 for v in history.values() if v == "recibo")

    added = 0
    confirmed_list = sorted(list(stems))
    for s in confirmed_list:
        if s not in history or history[s] != "recibo":
            history[s] = "recibo"
            added += 1

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    new_total = len(history)
    new_recibo = sum(1 for v in history.values() if v == "recibo")

    print(f"\\n1. Histórico Atualizado:")
    print(f"   ✓ Novos recibos adicionados: {added:,}")
    print(f"   ✓ Total de recibo          : {new_recibo:,}")
    print(f"   ✓ Total geral de anotações : {new_total:,}")

    # Sincronização PostgreSQL
    print(f"\\n2. Sincronizando com PostgreSQL...")
    pdf_to_year = {}
    if YEAR_FILE.exists():
        with open(YEAR_FILE, "r", encoding="utf-8") as f:
            pdf_to_year = json.load(f)

    try:
        with psycopg.connect(DB_URL) as conn:
            with conn.cursor() as cur:
                rows = []
                for s in confirmed_list:
                    doc_id = int(s)
                    ano = pdf_to_year.get(f"{s}.pdf", 2024)
                    caminho = f"/media/ssd2/datasets/CEAP/{s}.pdf"
                    rows.append((doc_id, ano, caminho, "baixado", "recibo"))

                batch_size = 2000
                for i in range(0, len(rows), batch_size):
                    chunk = rows[i:i + batch_size]
                    cur.executemany(\"\"\"
                        INSERT INTO documento (ide_documento, ano, caminho_arquivo, status, classe_predita)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (ide_documento) DO UPDATE SET
                            status = 'baixado',
                            classe_predita = 'recibo';
                    \"\"\", chunk)
                conn.commit()
        print(f"   ✓ {len(rows):,} documentos sincronizados na tabela 'documento'.")
    except Exception as e:
        print(f"   [AVISO] Erro PostgreSQL: {e}")

    print("\\n" + "=" * 72)
    print(" SUCESSO: RECIBOS INTEGRADOS AO DATASET!")
    print("=" * 72)

if __name__ == "__main__":
    main()
\"\"\"
"""


def main():
    t0 = time.time()
    print("=" * 74)
    print(" CRIAÇÃO DO LOTE DE VALIDAÇÃO: CLASSE RECIBO (42.518 CANDIDATOS >= 97%)")
    print("=" * 74)

    print("\n1. Carregando embeddings ArcFace 128-D e histórico de rótulos...")
    lat = np.load(EMBS_PATH)
    norms = np.linalg.norm(lat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    lat /= norms

    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)["documents"]
    with open(HIST_PATH, "r", encoding="utf-8") as f:
        history = json.load(f)
    with open(LOOKUP_FILE, "r", encoding="utf-8") as f:
        lookup = json.load(f)

    stems = [Path(d["doc_name"]).stem for d in meta]
    labeled_stems = set(history.keys())

    # 2. Encontrar recibos conhecidos e particionar em 2 clusters
    print("\n2. Agrupando amostras conhecidas de recibo em 2 clusters no ArcFace...")
    recibo_indices = [i for i, s in enumerate(stems) if history.get(s) == "recibo"]
    print(f"   Total de sementes conhecidas: {len(recibo_indices)}")

    recibo_vecs = lat[recibo_indices]
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(recibo_vecs)
    labels = kmeans.labels_

    subcotas_0 = [lookup.get(meta[recibo_indices[i]]["doc_name"], {}).get("subcota", "") for i, l in enumerate(labels) if l == 0]
    is_c0_taxi = "TÁXI" in " ".join(subcotas_0).upper() or "ALIMENTAÇÃO" in " ".join(subcotas_0).upper()

    c1_seed_indices = [recibo_indices[i] for i, l in enumerate(labels) if l == (0 if is_c0_taxi else 1)]
    c2_seed_indices = [recibo_indices[i] for i, l in enumerate(labels) if l == (1 if is_c0_taxi else 0)]

    cent_c1 = lat[c1_seed_indices].mean(axis=0); cent_c1 /= np.linalg.norm(cent_c1)
    cent_c2 = lat[c2_seed_indices].mean(axis=0); cent_c2 /= np.linalg.norm(cent_c2)

    print(f"   Centróide 1 (Táxi/Alimentação): {len(c1_seed_indices)} sementes")
    print(f"   Centróide 2 (Imóveis/Locação) : {len(c2_seed_indices)} sementes")

    # 3. Calcular similaridades para todos os não mapeados
    print("\n3. Calculando similaridades para os documentos não mapeados...")
    unmapped_indices = [i for i, s in enumerate(stems) if s not in labeled_stems]
    print(f"   Total de não mapeados analisados: {len(unmapped_indices):,}")

    sims_c1 = np.dot(lat, cent_c1)
    sims_c2 = np.dot(lat, cent_c2)

    candidates_c1 = []
    candidates_c2 = []

    for i in unmapped_indices:
        s1 = float(sims_c1[i])
        s2 = float(sims_c2[i])
        stem = stems[i]
        doc_name = meta[i]["doc_name"]
        info = lookup.get(doc_name, {})
        forn = info.get("fornecedor", "SEM_FORN")
        sub = info.get("subcota", "SEM_SUB")
        val = info.get("valorLiquido", "0")

        if s1 >= 0.97:
            candidates_c1.append((s1, stem, doc_name, forn, sub, val))
        elif s2 >= 0.97:
            candidates_c2.append((s2, stem, doc_name, forn, sub, val))

    candidates_c1.sort(key=lambda x: -x[0])
    candidates_c2.sort(key=lambda x: -x[0])

    print(f"   ✓ Candidatos Cluster 1 (Táxi/Alimentação): {len(candidates_c1):,}")
    print(f"   ✓ Candidatos Cluster 2 (Imóveis/Locação) : {len(candidates_c2):,}")
    print(f"   ✓ Total Geral Selecionado (>= 97%)      : {len(candidates_c1) + len(candidates_c2):,}")

    # 4. Renderização em paralelo das páginas PNG faltantes
    all_selected = candidates_c1 + candidates_c2
    render_tasks = []
    missing_cnt = 0
    for _, stem, doc_name, _, _, _ in all_selected:
        png_path = RENDERED_DIR / f"{stem}.png"
        if not png_path.exists():
            pdf_path = PDF_DIR / doc_name
            render_tasks.append((stem, str(pdf_path), str(png_path)))
            missing_cnt += 1

    print(f"\n4. Verificação de renderização:")
    print(f"   • PNGs já em cache: {len(all_selected) - missing_cnt:,}")
    print(f"   • PNGs a renderizar: {missing_cnt:,}")

    if render_tasks:
        print(f"   Renderizando {missing_cnt:,} páginas em paralelo com 24 workers...")
        t_rend = time.time()
        with ProcessPoolExecutor(max_workers=24) as pool:
            done = 0
            for res in pool.map(render_pdf_worker, render_tasks, chunksize=32):
                done += 1
                if done % 2000 == 0 or done == len(render_tasks):
                    rate = done / (time.time() - t_rend)
                    print(f"     [{done:,}/{len(render_tasks):,}] renderizados ({rate:.1f} págs/s)...", flush=True)

    # 5. Criar estrutura de diretórios e atalhos simbólicos
    print("\n5. Organizando pastas e gerando atalhos simbólicos...")
    dir_c1 = TARGET_DIR / "1_taxi_e_alimentacao"
    dir_c2 = TARGET_DIR / "2_imoveis_e_locacao"
    dir_c1.mkdir(parents=True, exist_ok=True)
    dir_c2.mkdir(parents=True, exist_ok=True)

    thumb_tasks = []
    manifest_records = []

    def populate_cluster_folder(candidates, cluster_root_dir, cluster_id, batch_size=3000):
        # Separar por faixas
        tier_99 = [c for c in candidates if c[0] >= 0.99]
        tier_98 = [c for c in candidates if 0.98 <= c[0] < 0.99]
        tier_97 = [c for c in candidates if 0.97 <= c[0] < 0.98]

        # Tier 99: pasta direta
        folder_99 = cluster_root_dir / f"1_alta_certeza_99pct_{len(tier_99)}"
        folder_99.mkdir(parents=True, exist_ok=True)
        for rank, c in enumerate(tier_99, 1):
            create_symlink_and_task(c, rank, folder_99, cluster_id, "99pct")

        # Tier 98: particionar em sublotes
        parent_98 = cluster_root_dir / f"2_faixa_98pct_{len(tier_98)}"
        parent_98.mkdir(parents=True, exist_ok=True)
        n_batches_98 = max(1, (len(tier_98) + batch_size - 1) // batch_size)
        for b_idx in range(n_batches_98):
            chunk = tier_98[b_idx * batch_size:(b_idx + 1) * batch_size]
            sub_folder = parent_98 / f"lote_{b_idx+1:02d}_{len(chunk)}"
            sub_folder.mkdir(parents=True, exist_ok=True)
            for rank, c in enumerate(chunk, b_idx * batch_size + 1):
                create_symlink_and_task(c, rank, sub_folder, cluster_id, "98pct")

        # Tier 97: particionar em sublotes
        parent_97 = cluster_root_dir / f"3_faixa_97pct_{len(tier_97)}"
        parent_97.mkdir(parents=True, exist_ok=True)
        n_batches_97 = max(1, (len(tier_97) + batch_size - 1) // batch_size)
        for b_idx in range(n_batches_97):
            chunk = tier_97[b_idx * batch_size:(b_idx + 1) * batch_size]
            sub_folder = parent_97 / f"lote_{b_idx+1:02d}_{len(chunk)}"
            sub_folder.mkdir(parents=True, exist_ok=True)
            for rank, c in enumerate(chunk, b_idx * batch_size + 1):
                create_symlink_and_task(c, rank, sub_folder, cluster_id, "97pct")

    def create_symlink_and_task(item, rank, dest_folder, cluster_id, tier):
        sim, stem, doc_name, forn, sub, val = item
        png_source = RENDERED_DIR / f"{stem}.png"
        forn_tag = sanitize_str(forn, 18)
        val_tag = sanitize_str(val, 8).replace(".", "_")
        sym_name = f"rank_{rank:05d}_sim_{sim:.3f}_[{forn_tag}]_R${val_tag}_{stem}.png"
        sym_path = dest_folder / sym_name

        if sym_path.exists() or sym_path.is_symlink():
            sym_path.unlink()
        try:
            sym_path.symlink_to(png_source)
            thumb_tasks.append((str(sym_path), str(png_source)))
        except Exception:
            pass

        manifest_records.append({
            "doc_id": stem,
            "filename": doc_name,
            "cluster": cluster_id,
            "tier": tier,
            "similarity": round(sim, 4),
            "fornecedor": forn,
            "subcota": sub,
            "valor": val,
            "symlink": str(sym_path)
        })

    print("   Popula sub-pastas do Cluster 1 (Táxi / Alimentação)...")
    populate_cluster_folder(candidates_c1, dir_c1, "cluster_1_taxi_alimentacao", batch_size=3000)

    print("   Popula sub-pastas do Cluster 2 (Imóveis / Locação)...")
    populate_cluster_folder(candidates_c2, dir_c2, "cluster_2_imoveis_locacao", batch_size=2500)

    # 6. Salvar Scripts de Confirmação e LEIAME
    print("\n6. Gerando scripts de confirmação e documentação...")
    with open(dir_c1 / "confirmar_taxi_alimentacao.py", "w", encoding="utf-8") as f:
        f.write(make_cluster_confirm_script("1_taxi_e_alimentacao", "Cluster 1 (Táxi, Alimentação e Pequenos Recibos)"))
    (dir_c1 / "confirmar_taxi_alimentacao.py").chmod(0o755)

    with open(dir_c2 / "confirmar_imoveis_locacao.py", "w", encoding="utf-8") as f:
        f.write(make_cluster_confirm_script("2_imoveis_e_locacao", "Cluster 2 (Imóveis, Condomínio e Locação)"))
    (dir_c2 / "confirmar_imoveis_locacao.py").chmod(0o755)

    with open(TARGET_DIR / "confirmar_todos_recibos.py", "w", encoding="utf-8") as f:
        f.write(make_master_confirm_script())
    (TARGET_DIR / "confirmar_todos_recibos.py").chmod(0o755)

    with open(TARGET_DIR / "manifest_recibo_42k.json", "w", encoding="utf-8") as f:
        json.dump(manifest_records, f, indent=2, ensure_ascii=False)

    with open(TARGET_DIR / "LEIAME.txt", "w", encoding="utf-8") as f:
        f.write(f"""LOTE DE VALIDAÇÃO VISUAL MASSIVA - CLASSE RECIBO (>= 97% ARCFACE)
========================================================================
Total Geral de Candidatos: {len(all_selected):,} documentos
- Cluster 1 (Táxi / Refeição / Alimentação / Comprovantes Avulsos): {len(candidates_c1):,} docs
- Cluster 2 (Manutenção Escritório / Condomínio / Imóveis / Locação): {len(candidates_c2):,} docs

COMO VALIDAR NO GERENCIADOR DE ARQUIVOS (THUNAR / NAUTILUS):
1. Abra as pastas de seu interesse:
   thunar {TARGET_DIR}/1_taxi_e_alimentacao/1_alta_certeza_99pct_{len([c for c in candidates_c1 if c[0]>=0.99])} &
   thunar {TARGET_DIR}/2_imoveis_e_locacao/1_alta_certeza_99pct_{len([c for c in candidates_c2 if c[0]>=0.99])} &

2. Miniaturas ultra-rápidas pré-geradas no cache do sistema.
3. Se vir algum intruso (ex: NF-e ou cupom térmico evidente), basta apagá-lo (Delete).
4. Para confirmar:
   - Para confirmar apenas o Cluster 1:
     python3 {dir_c1}/confirmar_taxi_alimentacao.py
   - Para confirmar apenas o Cluster 2:
     python3 {dir_c2}/confirmar_imoveis_locacao.py
   - Para confirmar TODOS os mantidos em ambas as pastas:
     python3 {TARGET_DIR}/confirmar_todos_recibos.py
""")

    # 7. Criar link no Desktop
    if DESKTOP_LINK.exists() or DESKTOP_LINK.is_symlink():
        DESKTOP_LINK.unlink()
    DESKTOP_LINK.symlink_to(TARGET_DIR)
    print(f"   ✓ Atalho no Desktop atualizado: {DESKTOP_LINK} -> {TARGET_DIR}")

    # 8. Pré-gerar thumbnails no cache ~/.cache/thumbnails/large
    print(f"\n7. Pré-gerando {len(thumb_tasks):,} miniaturas no cache do sistema...")
    with ThreadPoolExecutor(max_workers=32) as pool:
        done_t = sum(1 for res in pool.map(setup_thumbnail_fast, thumb_tasks) if res)
    print(f"   ✓ {done_t:,} miniaturas associadas com sucesso!")

    print("\n" + "=" * 74)
    print("      LOTE MASSIVO DE RECIBOS GERADO COM SUCESSO!")
    print("=" * 74)
    print(f"Tempo total: {time.time() - t0:.1f} segundos")
    print(f"Localização: {DESKTOP_LINK}")
    print("=" * 74)


if __name__ == "__main__":
    main()
