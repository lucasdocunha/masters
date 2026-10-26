import json
import shutil
import time
import hashlib
import urllib.parse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pymupdf

ROOT_DIR = Path("/home/lucas.ocunha/masters")
DATA_DIR = ROOT_DIR / "data"
DEST_DIR = Path("/home/lucas.ocunha/Desktop/validacao_nfe")
THUMB_LARGE_DIR = Path.home() / ".cache" / "thumbnails" / "large"
RENDERED_DIR = DATA_DIR / "rendered_pages"
SEEDS_FILE = DATA_DIR / "seeds_modelo_55.json"
REJECTED_FILE = DATA_DIR / "rejected_modelo_55.json"
HISTORY_FILE = ROOT_DIR / "ml_backend" / "models" / "annotations_history.json"
LOOKUP_FILE = DATA_DIR / "pdf_metadata_lookup.json"
NPZ_FILE = DATA_DIR / "latent_vectors_sample_5pct.npz"
META_FILE = DATA_DIR / "latent_vectors_sample_5pct_metadata.json"

print("=== ATUALIZAÇÃO DO CENTROIDE E GERAÇÃO DA PASTA NFE ===")

# 1. Carregar embeddings e metadados
print("Carregando embeddings...")
npz = np.load(NPZ_FILE)
comb = np.hstack([npz["dino_doc_embeddings"], npz["vlm_doc_embeddings"]]).astype(np.float32)
norms = np.linalg.norm(comb, axis=1, keepdims=True)
norms[norms == 0] = 1.0
comb /= norms

with open(META_FILE, "r", encoding="utf-8") as f:
    docs = json.load(f)["documents"]
stem_to_idx = {Path(d["doc_name"]).stem: idx for idx, d in enumerate(docs)}

lookup = {}
if LOOKUP_FILE.exists():
    with open(LOOKUP_FILE, "r", encoding="utf-8") as f:
        lookup = json.load(f)

# 2. Consolidar sementes confirmadas
# A. Sementes originais
seeds77 = json.load(open(SEEDS_FILE)) if SEEDS_FILE.exists() else []
all_seeds = {s["stem"]: s for s in seeds77}

# B. Sementes confirmadas na subpasta 3_danfe_nfe_mercadorias
danfe107_file = Path("/home/lucas.ocunha/Desktop/validacao_nfs/3_danfe_nfe_mercadorias/.candidatos.json")
if danfe107_file.exists():
    for d in json.load(open(danfe107_file)):
        s = d["stem"]
        if s not in all_seeds:
            meta = lookup.get(f"{s}.pdf", {})
            all_seeds[s] = {
                "stem": s,
                "doc_name": f"{s}.pdf",
                "fornecedor": meta.get("fornecedor", d.get("fornecedor", "N/A")),
                "subcota": meta.get("subcota", d.get("subcota", "N/A")),
                "ano": meta.get("ano", "N/A"),
                "reason": "Confirmado lote NFS subpasta 3_danfe"
            }

# C. Sementes validadas da pasta validacao_nfe (ranks 1 a 306 mantidos)
old_manifest_path = DEST_DIR / ".candidatos.json"
manifest_306 = json.load(open(old_manifest_path))[:306] if old_manifest_path.exists() else []
existing_pngs = set(p.name for p in DEST_DIR.glob("rank_*.png"))

kept_306 = [m for m in manifest_306 if m["filename"] in existing_pngs]
deleted_306 = [m for m in manifest_306 if m["filename"] not in existing_pngs]

for k in kept_306:
    s = k["stem"]
    if s not in all_seeds:
        meta = lookup.get(k["doc_name"], {})
        all_seeds[s] = {
            "stem": s,
            "doc_name": k["doc_name"],
            "fornecedor": k.get("fornecedor", meta.get("fornecedor", "N/A")),
            "subcota": k.get("subcota", meta.get("subcota", "N/A")),
            "ano": k.get("ano", meta.get("ano", "N/A")),
            "reason": f"Confirmado revisao visual rank {k.get('rank', 'N/A')}"
        }

print(f"Total de sementes confirmadas: {len(all_seeds)}")

# Salvar backup e novo seeds_modelo_55.json
shutil.copyfile(SEEDS_FILE, DATA_DIR / "seeds_modelo_55.backup_77.json")
with open(SEEDS_FILE, "w", encoding="utf-8") as f:
    json.dump(list(all_seeds.values()), f, indent=2, ensure_ascii=False)
print(f"Salvo novo seeds_modelo_55.json com {len(all_seeds)} sementes!")

# Salvar rejeitados
rejected_stems = [m["stem"] for m in deleted_306]
with open(REJECTED_FILE, "w", encoding="utf-8") as f:
    json.dump(deleted_306, f, indent=2, ensure_ascii=False)
print(f"Salvo rejected_modelo_55.json com {len(deleted_306)} documentos rejeitados.")

# 3. Calcular novo centroide
seed_indices = [stem_to_idx[s] for s in all_seeds if s in stem_to_idx]
new_centroid = np.mean(comb[seed_indices], axis=0)
new_centroid /= np.linalg.norm(new_centroid)
print(f"Novo centroide calculado a partir de {len(seed_indices)} embeddings!")

# 4. Encontrar candidatos com similaridade >= 0.90
history = json.load(open(HISTORY_FILE)) if HISTORY_FILE.exists() else {}
excluded = set(history.keys()).union(all_seeds.keys()).union(rejected_stems)

sims = np.dot(comb, new_centroid)

candidates = []
for idx, d in enumerate(docs):
    doc_name = d.get("doc_name", "")
    stem = Path(doc_name).stem
    if stem in excluded or d.get("num_pages", 1) != 1:
        continue
    pdf_path = Path("/media/ssd2/datasets/CEAP") / doc_name
    if not pdf_path.exists():
        continue
    s = float(sims[idx])
    if s >= 0.90:
        meta = lookup.get(doc_name, {})
        candidates.append({
            "sim": s,
            "doc_name": doc_name,
            "stem": stem,
            "pdf_path": str(pdf_path),
            "fornecedor": meta.get("fornecedor", "N/A"),
            "subcota": meta.get("subcota", "N/A"),
            "ano": meta.get("ano", "N/A"),
            "png_target": str(RENDERED_DIR / f"{stem}.png")
        })

candidates.sort(key=lambda x: x["sim"], reverse=True)
print(f"Total de candidatos bem proximos (sim >= 0.90): {len(candidates):,} (de {candidates[-1]['sim']:.4f} a {candidates[0]['sim']:.4f})")

# 5. Limpar pasta DEST_DIR e criar novos links simbólicos
for old_f in DEST_DIR.glob("rank_*.*"):
    old_f.unlink()

initial_manifest = []
thumb_tasks = []

def setup_thumbnail(symlink_path, png_path):
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
with open(DEST_DIR / ".candidatos.json", "w", encoding="utf-8") as f:
    json.dump(initial_manifest, f, indent=2, ensure_ascii=False)

print(f"Criados {len(candidates)} links simbólicos.")

# 6. Gerar miniaturas
t_th = time.time()
with ThreadPoolExecutor(max_workers=16) as ex:
    futures = [ex.submit(setup_thumbnail, s, p) for s, p in thumb_tasks]
    ok = sum(1 for f in futures if f.result())
print(f"Miniaturas geradas ({ok}/{len(thumb_tasks)}) em {time.time()-t_th:.2f}s!")

# 7. Atualizar LEIAME
readme_path = DEST_DIR / "LEIAME.txt"
with open(readme_path, "w", encoding="utf-8") as f:
    f.write(f"""PASTA DE VALIDAÇÃO VISUAL - CLASSE NFE (DANFE - MODELO 55) - LOTE REFINADO
========================================================================

Esta pasta foi gerada a partir do NOVO CENTROIDE refinado com as 408 sementes confirmadas.
Contém {len(candidates):,} documentos altamente similares (similaridade cosseno >= 0.9000).

COMO IDENTIFICAR O DANFE (MODELO 55):
- Formato Folha A4 Retrato.
- Canhoto de recebimento no topo ("RECEBEMOS DE [EMPRESA] OS PRODUTOS...").
- Cabeçalho: "DANFE - DOCUMENTO AUXILIAR DA NOTA FISCAL ELETRÔNICA - 0-ENTRADA 1-SAÍDA".
- Código de barras com chave de 44 dígitos (dígitos 20-22 são "55").
- Quadro de cálculo do imposto (ICMS) e tabela de produtos com colunas NCM/CST/CFOP.

COMO USAR:
1. Abra esta pasta no Gerenciador de Arquivos (Thunar) ou no visualizador de imagens (Ristretto).
2. Navegue usando as setas do teclado (-> e <-).
3. Selecione e APAGUE (tecla Del) qualquer arquivo que NÃO for DANFE / Modelo 55.
4. Quando terminar a revisão:
   - Me avise aqui: "fiz a validação da nfe, pode aplicar"
   ou
   - Execute o script na pasta:
     cd ~/Desktop/validacao_nfe && ./confirmar_validacao.py
""")

print("\n[SUCESSO] Pasta validacao_nfe totalmente atualizada com o novo lote refinado!")
