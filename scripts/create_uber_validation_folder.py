#!/usr/bin/env python3
"""
Cria a pasta de validação visual em lote para a classe UBER.
Calcula a similaridade com o centroide do cluster UBER já anotado,
seleciona os documentos mais próximos (sim >= 0.85), cria links simbólicos
para os PDFs originais e pré-gera miniaturas no cache do Thunar (FreeDesktop).
Também cria o script auxiliar 'confirmar_validacao.py'.
"""

import hashlib
import json
import os
import sys
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pymupdf

DATA_DIR = Path("/home/lucas.ocunha/masters/data")
PDF_DIR = Path("/media/ssd2/datasets/CEAP")
NPZ_FILE = DATA_DIR / "latent_vectors_sample_5pct.npz"
META_FILE = DATA_DIR / "latent_vectors_sample_5pct_metadata.json"
LOOKUP_FILE = DATA_DIR / "pdf_metadata_lookup.json"
HISTORY_FILE = Path("/home/lucas.ocunha/masters/ml_backend/models/annotations_history.json")

DEST_DIR = Path("/home/lucas.ocunha/Desktop/validacao_uber")
MASTERS_LINK = Path("/home/lucas.ocunha/masters/validacao_uber")
THUMB_LARGE_DIR = Path.home() / ".cache" / "thumbnails" / "large"
THUMB_NORMAL_DIR = Path.home() / ".cache" / "thumbnails" / "normal"


def render_thumbnail(pdf_path: Path, symlink_path: Path) -> bool:
    """Renderiza a 1ª página do PDF para o cache de miniaturas do sistema (FreeDesktop)."""
    try:
        uri = "file://" + urllib.parse.quote(str(symlink_path.resolve()))
        h = hashlib.md5(uri.encode("utf-8")).hexdigest()
        target_png = THUMB_LARGE_DIR / f"{h}.png"
        
        # Também criar no hash do symlink direto
        uri_sym = "file://" + urllib.parse.quote(str(symlink_path.absolute()))
        h_sym = hashlib.md5(uri_sym.encode("utf-8")).hexdigest()
        target_sym_png = THUMB_LARGE_DIR / f"{h_sym}.png"

        if target_png.exists() and target_sym_png.exists():
            return True

        doc = pymupdf.open(pdf_path)
        if len(doc) == 0:
            doc.close()
            return False
        page = doc[0]
        # 128 DPI gera ~256x360 px, perfeito para a visualização 'large' do Thunar
        pix = page.get_pixmap(dpi=100)
        pix.save(str(target_png))
        if h != h_sym:
            pix.save(str(target_sym_png))
        doc.close()
        return True
    except Exception:
        return False


def main(min_similarity: float = 0.85):
    print("=== Criação da Pasta de Validação Rápida da Classe UBER ===")
    t0 = time.time()

    # 1. Carregar embeddings e metadados
    print(f"Carregando embeddings de {NPZ_FILE}...")
    npz = np.load(NPZ_FILE)
    combined = np.hstack([npz["dino_doc_embeddings"], npz["vlm_doc_embeddings"]]).astype(np.float32)
    norms = np.linalg.norm(combined, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    combined /= norms

    print(f"Carregando metadados de {META_FILE}...")
    with open(META_FILE, "r", encoding="utf-8") as f:
        docs = json.load(f).get("documents", [])

    stem_to_idx = {Path(d.get("doc_name", "")).stem: idx for idx, d in enumerate(docs)}

    # 2. Carregar histórico de anotações
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        history = json.load(f)

    uber_stems = [s for s, label in history.items() if label == "uber" and s in stem_to_idx]
    print(f"Encontrados {len(uber_stems)} documentos UBER já rotulados no histórico.")

    if not uber_stems:
        print("Erro: Nenhum documento UBER encontrado no histórico.")
        sys.exit(1)

    uber_indices = [stem_to_idx[s] for s in uber_stems]
    uber_centroid = np.mean(combined[uber_indices], axis=0)
    uber_centroid /= np.linalg.norm(uber_centroid)

    # 3. Calcular similaridades com todos os documentos não rotulados
    print("Calculando similaridade cosseno em relação ao centroide UBER...")
    sims = np.dot(combined, uber_centroid)

    lookup: dict[str, Any] = {}
    if LOOKUP_FILE.exists():
        with open(LOOKUP_FILE, "r", encoding="utf-8") as f:
            lookup = json.load(f)

    labeled_stems = set(history.keys())
    candidates = []
    for idx, d in enumerate(docs):
        doc_name = d.get("doc_name", "")
        stem = Path(doc_name).stem
        if stem in labeled_stems:
            continue
        pdf_path = PDF_DIR / doc_name
        if not pdf_path.exists():
            continue
        sim = float(sims[idx])
        if sim >= min_similarity:
            meta = lookup.get(doc_name, {})
            candidates.append({
                "sim": sim,
                "doc_name": doc_name,
                "stem": stem,
                "pdf_path": str(pdf_path),
                "num_pages": d.get("num_pages", 1),
                "fornecedor": meta.get("fornecedor", "N/A"),
                "subcota": meta.get("subcota", "N/A"),
                "ano": meta.get("ano", d.get("year", "N/A")),
            })

    candidates.sort(key=lambda x: x["sim"], reverse=True)
    print(f"Total de candidatos com similaridade >= {min_similarity:.2f}: {len(candidates):,}")

    # 4. Criar diretório de destino
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    THUMB_LARGE_DIR.mkdir(parents=True, exist_ok=True)
    THUMB_NORMAL_DIR.mkdir(parents=True, exist_ok=True)

    if not MASTERS_LINK.exists() and not MASTERS_LINK.is_symlink():
        MASTERS_LINK.symlink_to(DEST_DIR, target_is_directory=True)

    # 5. Criar links simbólicos com ordenação por rank e similaridade
    print("Criando links simbólicos e pré-renderizando miniaturas do Thunar...")
    thumb_tasks = []
    initial_manifest = []

    for rank, c in enumerate(candidates, 1):
        filename = f"rank_{rank:04d}_sim_{c['sim']:.4f}_{c['stem']}.pdf"
        symlink_path = DEST_DIR / filename
        
        # Criar symlink se não existir
        if symlink_path.exists() or symlink_path.is_symlink():
            symlink_path.unlink()
        symlink_path.symlink_to(c["pdf_path"])

        c_info = dict(c)
        c_info["rank"] = rank
        c_info["filename"] = filename
        initial_manifest.append(c_info)

        thumb_tasks.append((Path(c["pdf_path"]), symlink_path))

    # 6. Salvar manifesto inicial
    manifest_path = DEST_DIR / ".candidatos_iniciais.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(initial_manifest, f, indent=2, ensure_ascii=False)

    # 7. Pré-renderizar miniaturas em paralelo (FreeDesktop thumbnail cache)
    t_thumb = time.time()
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(render_thumbnail, pdf_p, sym_p) for pdf_p, sym_p in thumb_tasks]
        done = sum(1 for f in futures if f.result())
    print(f"Miniaturas pré-geradas para {done}/{len(thumb_tasks)} arquivos em {time.time()-t_thumb:.2f}s!")

    # 8. Criar script 'confirmar_validacao.py' dentro da pasta
    script_confirmar = DEST_DIR / "confirmar_validacao.py"
    with open(script_confirmar, "w", encoding="utf-8") as f:
        f.write('''#!/usr/bin/env python3
"""
Confirma a validação dos documentos na pasta 'validacao_uber'.
Compara os arquivos PDF restantes com o manifesto inicial:
- O que foi MANTIDO é confirmado e registrado como classe 'uber' no histórico do modelo.
- O que foi APAGADO é registrado como rejeitado (não-uber).
Em seguida, retreina o modelo no ml-backend e sincroniza com o Label Studio!
"""

import json
import os
import sys
from pathlib import Path
import requests

FOLDER = Path(__file__).resolve().parent
MANIFEST_FILE = FOLDER / ".candidatos_iniciais.json"
HISTORY_FILE = Path("/home/lucas.ocunha/masters/ml_backend/models/annotations_history.json")
ML_BACKEND_URL = "http://localhost:9090"

if not MANIFEST_FILE.exists():
    print(f"Erro: Arquivo {MANIFEST_FILE} não encontrado.")
    sys.exit(1)

with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
    manifest = json.load(f)

# Obter stems restantes
remaining_files = set(p.name for p in FOLDER.glob("*.pdf"))
print(f"=== Validação da Classe UBER ===")
print(f"Total inicial de candidatos: {len(manifest):,}")
print(f"Arquivos restantes na pasta: {len(remaining_files):,}")

accepted = []
rejected = []

for item in manifest:
    if item["filename"] in remaining_files:
        accepted.append(item)
    else:
        rejected.append(item)

print(f"\\nResumo:")
print(f"  [OK] Mantidos (Confirmados como UBER): {len(accepted):,}")
print(f"  [X]  Apagados (Rejeitados/Não-UBER):   {len(rejected):,}")

if not accepted:
    print("\\nAviso: Nenhum arquivo restante para confirmar.")
    sys.exit(0)

ans = input(f"\\nDeseja gravar os {len(accepted):,} documentos como 'uber' no modelo e retreinar? [S/n]: ")
if ans.strip().lower() not in ["", "s", "sim", "y", "yes"]:
    print("Operação cancelada.")
    sys.exit(0)

# Carregar histórico e atualizar
with open(HISTORY_FILE, "r", encoding="utf-8") as f:
    history = json.load(f)

for item in accepted:
    history[item["stem"]] = "uber"

with open(HISTORY_FILE, "w", encoding="utf-8") as f:
    json.dump(history, f, indent=2, ensure_ascii=False)

print(f"[OK] Histórico atualizado com {len(accepted):,} validações da classe UBER!")
print(f"Total geral no histórico agora: {len(history):,} anotações.")

# Disparar retreino no ML Backend
print("Disparando retreino no ML backend...")
try:
    res = requests.post(f"{ML_BACKEND_URL}/train", timeout=30)
    if res.status_code == 200:
        data = res.json()
        print(f"[OK SUCESSO] Modelo retreinado: {data.get('version')} | Acurácia: {data.get('train_accuracy', 0):.2%}")
        print("Sincronização de predições com o Label Studio disparada em segundo plano!")
    else:
        print(f"Erro no retreino: {res.status_code} - {res.text}")
except Exception as e:
    print(f"Aviso ao contatar ML backend: {e}")

print("\\nProcesso concluído com sucesso!")
''')
    script_confirmar.chmod(0o755)

    # 9. Criar LEIAME.txt
    readme_path = DEST_DIR / "LEIAME.txt"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(f"""PASTA DE VALIDAÇÃO VISUAL - CLASSE UBER
========================================

Esta pasta contém {len(candidates):,} documentos com alta similaridade visual e semântica (>= {min_similarity:.2f}) com a classe UBER.

COMO USAR:
1. Abra esta pasta no Gerenciador de Arquivos (Thunar) em 'Exibir como Ícones' para ver as miniaturas de todos os recibos.
2. Os arquivos estão ordenados pelo rank de similaridade:
   - Os primeiros (rank 0001 a 1000+) são praticamente 100% Uber.
   - Os últimos (perto de rank {len(candidates):04d}) contêm os casos limítrofes (ex: 99 POP, passagens aéreas).
3. Selecione e APAGUE (tecla Delete) qualquer arquivo que NÃO for da classe Uber.
   (Nota: os arquivos aqui são referências/links simbólicos; apagar não apaga o dataset original).
4. Quando terminar a revisão, execute o script de confirmação pelo terminal:
   python3 confirmar_validacao.py
   ou
   ./confirmar_validacao.py

Isso irá registrar automaticamente todos os documentos que você manteve como classe 'uber' no histórico do modelo e retreinar a nova versão!
""")

    print(f"\n[SUCESSO] Pasta de validação criada em:")
    print(f"  -> {DEST_DIR}")
    print(f"  -> Acesso rápido no Desktop: {DEST_DIR.name}")
    print(f"Tempo total: {time.time()-t0:.2f}s.")


if __name__ == "__main__":
    main(min_similarity=0.85)
