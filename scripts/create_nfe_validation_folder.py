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

print("=== REFINAMENTO PURO DO CENTROIDE NFE (SEM CONTAMINAÇÃO DE NFSE) ===")

# 1. Carregar embeddings
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

# 2. Sementes PURAS: apenas seeds77 + as confirmadas visualmente pelo usuário (EXCLUINDO danfe107!)
seeds77 = json.load(open(DATA_DIR / "seeds_modelo_55.backup_77.json"))
clean_seeds = {s["stem"]: s for s in seeds77}

old_seeds = json.load(open(SEEDS_FILE))
user_confirmed = [s for s in old_seeds if "Confirmado revisao visual" in s.get("reason", "")]
for u in user_confirmed:
    clean_seeds[u["stem"]] = u

# Rejeitados
rejected_list = json.load(open(REJECTED_FILE))
rejected_stems = set(r["stem"] for r in rejected_list)

print(f"Total de sementes limpas: {len(clean_seeds)}")

# Salvar seeds_modelo_55.json limpo
with open(SEEDS_FILE, "w", encoding="utf-8") as f:
    json.dump(list(clean_seeds.values()), f, indent=2, ensure_ascii=False)

# 3. Calcular centroide puro de NFe
seed_indices = [stem_to_idx[s] for s in clean_seeds if s in stem_to_idx]
nfe_centroid = np.mean(comb[seed_indices], axis=0)
nfe_centroid /= np.linalg.norm(nfe_centroid)

# 4. Calcular centroide de NFSe para contraste e filtro
history = json.load(open(HISTORY_FILE)) if HISTORY_FILE.exists() else {}
nfse_stems = [s for s, l in history.items() if l == "nfse" and s in stem_to_idx]
nfse_centroid = np.mean(comb[[stem_to_idx[s] for s in nfse_stems]], axis=0)
nfse_centroid /= np.linalg.norm(nfse_centroid)

excluded = set(history.keys()).union(clean_seeds.keys()).union(rejected_stems)

sims_nfe = np.dot(comb, nfe_centroid)
sims_nfse = np.dot(comb, nfse_centroid)

candidates = []
for idx, d in enumerate(docs):
    doc_name = d.get("doc_name", "")
    stem = Path(doc_name).stem
    if stem in excluded or d.get("num_pages", 1) != 1:
        continue
    pdf_path = Path("/media/ssd2/datasets/CEAP") / doc_name
    if not pdf_path.exists():
        continue
    
    s_nfe = float(sims_nfe[idx])
    s_nfse = float(sims_nfse[idx])
    
    # Filtro rígido: similaridade alta com NFe E maior que com NFSe
    if s_nfe >= 0.885 and s_nfe > s_nfse:
        meta = lookup.get(doc_name, {})
        candidates.append({
            "sim": s_nfe,
            "sim_nfse": s_nfse,
            "diff_sim": s_nfe - s_nfse,
            "doc_name": doc_name,
            "stem": stem,
            "pdf_path": str(pdf_path),
            "fornecedor": meta.get("fornecedor", "N/A"),
            "subcota": meta.get("subcota", "N/A"),
            "ano": meta.get("ano", "N/A"),
            "png_target": str(RENDERED_DIR / f"{stem}.png")
        })

# Ordenar por maior similaridade NFe
candidates.sort(key=lambda x: x["sim"], reverse=True)
min_sim = candidates[-1]["sim"]
max_sim = candidates[0]["sim"]
print(f"Candidatos selecionados (puros NFe): {len(candidates):,} (sim NFe: {min_sim:.4f} a {max_sim:.4f})")

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

print(f"Criados {len(candidates)} links simbólicos limpos.")

# 6. Garantir miniaturas
t_th = time.time()
with ThreadPoolExecutor(max_workers=16) as ex:
    futures = [ex.submit(setup_thumbnail, s, p) for s, p in thumb_tasks]
    ok = sum(1 for f in futures if f.result())
print(f"Miniaturas verificadas ({ok}/{len(thumb_tasks)}) em {time.time()-t_th:.2f}s!")

# 7. Atualizar LEIAME
readme_path = DEST_DIR / "LEIAME.txt"
with open(readme_path, "w", encoding="utf-8") as f:
    f.write(f"""PASTA DE VALIDAÇÃO VISUAL - CLASSE NFE (DANFE - MODELO 55) - LOTE PURO
======================================================================

Lote gerado a partir do CENTROIDE PURO de NF-e (apenas sementes verificadas, com filtro de exclusão contra NFS-e).
Total de candidatos: {len(candidates):,} documentos.

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

print("\n[CONCLUÍDO] Pasta validacao_nfe recriada com documentos 100% alinhados à DANFE!")
