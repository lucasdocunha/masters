#!/usr/bin/env python3
"""
Cria a pasta de validação visual em lote para a classe NFS (Nota Fiscal de Serviços - NFSe).
Calcula a similaridade com o centroide da classe NFSe já anotada,
seleciona os documentos mais próximos (sim >= 0.88), cria links simbólicos
para os PDFs originais e pré-gera miniaturas no cache do Thunar (FreeDesktop).
Também cria o script auxiliar 'confirmar_validacao.py'.
"""

import argparse
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

DEST_DIR = Path("/home/lucas.ocunha/Desktop/validacao_nfs")
MASTERS_LINK = Path("/home/lucas.ocunha/masters/validacao_nfs")
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
        # 100 DPI gera ~200x300 px, ideal para a visualização 'large' do Thunar
        pix = page.get_pixmap(dpi=100)
        pix.save(str(target_png))
        if h != h_sym:
            pix.save(str(target_sym_png))
        doc.close()
        return True
    except Exception:
        return False


def main(min_similarity: float = 0.88, max_candidates: int | None = None):
    print("=== Criação da Pasta de Validação Rápida da Classe NFS (NFSe) ===")
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

    nfse_stems = [s for s, label in history.items() if label == "nfse" and s in stem_to_idx]
    print(f"Encontrados {len(nfse_stems)} documentos NFSe já rotulados no histórico.")

    if not nfse_stems:
        print("Erro: Nenhum documento NFSe encontrado no histórico para calcular centroide.")
        sys.exit(1)

    nfse_indices = [stem_to_idx[s] for s in nfse_stems]
    nfse_centroid = np.mean(combined[nfse_indices], axis=0)
    nfse_centroid /= np.linalg.norm(nfse_centroid)

    # 3. Calcular similaridades com todos os documentos não rotulados
    print("Calculando similaridade cosseno em relação ao centroide NFSe...")
    sims = np.dot(combined, nfse_centroid)

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
    if max_candidates is not None and len(candidates) > max_candidates:
        candidates = candidates[:max_candidates]

    print(f"Total de candidatos com similaridade >= {min_similarity:.2f}: {len(candidates):,}")

    # 4. Criar diretório de destino
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    THUMB_LARGE_DIR.mkdir(parents=True, exist_ok=True)
    THUMB_NORMAL_DIR.mkdir(parents=True, exist_ok=True)

    if not MASTERS_LINK.exists() and not MASTERS_LINK.is_symlink():
        MASTERS_LINK.symlink_to(DEST_DIR, target_is_directory=True)

    # Limpar symlinks antigos na pasta se existirem
    for old_f in DEST_DIR.glob("*.pdf"):
        if old_f.is_symlink() or old_f.is_file():
            old_f.unlink()

    # 5. Criar links simbólicos com ordenação por rank e similaridade
    print("Criando links simbólicos e preparando miniaturas...")
    thumb_tasks = []
    initial_manifest = []

    for rank, c in enumerate(candidates, 1):
        filename = f"rank_{rank:04d}_sim_{c['sim']:.4f}_{c['stem']}.pdf"
        symlink_path = DEST_DIR / filename
        
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
    print(f"Pré-renderizando miniaturas para {len(thumb_tasks):,} arquivos com 8 workers...")
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
Confirma a validação dos documentos na pasta 'validacao_nfs'.
Compara os arquivos PDF restantes com o manifesto inicial:
- O que foi MANTIDO é confirmado e registrado como classe 'nfse' no histórico do modelo.
- O que foi APAGADO é registrado como rejeitado (não-nfse).
Em seguida, retreina o modelo no ml-backend e sincroniza com o Label Studio!
"""

import json
import os
import subprocess
import sys
from pathlib import Path
import requests

FOLDER = Path(__file__).resolve().parent
MANIFEST_FILE = FOLDER / ".candidatos_iniciais.json"
HISTORY_FILE = Path("/home/lucas.ocunha/masters/ml_backend/models/annotations_history.json")
ACCEPTED_FILE = Path("/home/lucas.ocunha/masters/data/accepted_nfs.json")
ML_BACKEND_URL = "http://localhost:9090"
LABEL_STUDIO_URL = "http://localhost:8085"

if not MANIFEST_FILE.exists():
    print(f"Erro: Arquivo {MANIFEST_FILE} não encontrado.")
    sys.exit(1)

with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
    manifest = json.load(f)

# Obter arquivos restantes
remaining_files = set(p.name for p in FOLDER.glob("*.pdf"))
print(f"=== Validação da Classe NFS (Nota Fiscal de Serviços - NFSe) ===")
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
print(f"  [OK] Mantidos (Confirmados como NFSe): {len(accepted):,}")
print(f"  [X]  Apagados (Rejeitados/Não-NFSe):   {len(rejected):,}")

if not accepted:
    print("\\nAviso: Nenhum arquivo restante para confirmar.")
    sys.exit(0)

auto_yes = "-y" in sys.argv or "--yes" in sys.argv
if not auto_yes:
    ans = input(f"\\nDeseja gravar os {len(accepted):,} documentos como 'nfse' no modelo e retreinar? [S/n]: ")
    if ans.strip().lower() not in ["", "s", "sim", "y", "yes"]:
        print("Operação cancelada.")
        sys.exit(0)

# Salvar json com os aceitos e deletados
accepted_stems = [item["stem"] for item in accepted]
rejected_stems = [item["stem"] for item in rejected]

ACCEPTED_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(ACCEPTED_FILE, "w", encoding="utf-8") as f:
    json.dump({"accepted": accepted_stems, "rejected": rejected_stems}, f, indent=2)
print(f"[OK] Dados salvos em {ACCEPTED_FILE}")

# Carregar histórico e atualizar
try:
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        history = json.load(f)
    for stem in accepted_stems:
        history[stem] = "nfse"
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"[OK] Histórico atualizado diretamente. Total agora: {len(history):,} anotações.")
except PermissionError:
    print("[Aviso] Permissão negada para atualizar histórico diretamente. Atualizando via container...")
    subprocess.run([
        "docker", "exec", "ceap-ml-backend", "python3", "-c",
        f"""
import json
with open('/data/accepted_nfs.json') as f:
    acc = set(json.load(f)['accepted'])
with open('/app/models/annotations_history.json', 'r') as f:
    h = json.load(f)
for s in acc:
    h[s] = 'nfse'
with open('/app/models/annotations_history.json', 'w') as f:
    json.dump(h, f, indent=2)
print(f'Histórico atualizado via container: {len(h)} anotações.')
"""
    ], check=True)

# Disparar retreino no ML Backend
print("\\nDisparando retreino no ML backend...")
try:
    res = requests.post(f"{ML_BACKEND_URL}/train", timeout=60)
    if res.status_code == 200:
        data = res.json()
        print(f"[OK SUCESSO] Modelo retreinado: {data.get('version')} | Acurácia treino: {data.get('train_accuracy', 0):.2%}")
        print("Sincronização de predições com o Label Studio disparada em segundo plano!")
    else:
        print(f"Erro no retreino via HTTP: {res.status_code} - {res.text}. Tentando via docker...")
        subprocess.run(["docker", "exec", "ceap-ml-backend", "python3", "-c",
            "from model import DocumentClassifierManager; from pathlib import Path; "
            "m = DocumentClassifierManager(Path('/app/models'), Path('/data/latent_vectors_sample_5pct.npz'), Path('/data/latent_vectors_sample_5pct_metadata.json'), 50); "
            "print('Resultado treino:', m.train())"
        ], check=True)
except Exception as e:
    print(f"Aviso ao contatar ML backend: {e}")

print("\\nProcesso concluído com sucesso!")
''')
    script_confirmar.chmod(0o755)

    # 9. Criar LEIAME.txt
    readme_path = DEST_DIR / "LEIAME.txt"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(f"""PASTA DE VALIDAÇÃO VISUAL - CLASSE NFS (NOTA FISCAL DE SERVIÇOS - NFSE)
========================================================================

Esta pasta contém {len(candidates):,} documentos com alta similaridade visual e semântica (>= {min_similarity:.2f}) com a classe NFSe.

COMO USAR:
1. Abra esta pasta no Gerenciador de Arquivos (Thunar) em 'Exibir como Ícones' para ver as miniaturas das notas fiscais.
2. Os arquivos estão ordenados pelo rank de similaridade:
   - Os primeiros (rank 0001 a 1500+) são praticamente 100% NFS-e (Prefeituras municipais, serviços, hospedagem, divulgação, consultoria).
   - Os últimos (perto de rank {len(candidates):04d}) podem conter casos limítrofes (ex: cupons fiscais NFC-e de postos ou DANFE de mercadorias).
3. Selecione e APAGUE (tecla Delete) qualquer arquivo que NÃO for Nota Fiscal de Serviços (NFS-e).
   (Nota: os arquivos aqui são referências/links simbólicos; apagar NÃO apaga o arquivo original no dataset).
4. Quando terminar a revisão:
   - Você pode pedir para o agente: "fiz a validação da NFS, pode aplicar"
   ou
   - Executar o script de confirmação pelo terminal:
     python3 confirmar_validacao.py

Isso irá registrar automaticamente todos os documentos mantidos como classe 'nfse' no histórico do modelo e retreinar a nova versão!
""")

    print(f"\n[SUCESSO] Pasta de validação criada em:")
    print(f"  -> {DEST_DIR}")
    print(f"  -> Acesso rápido no Desktop: {DEST_DIR.name}")
    print(f"Tempo total: {time.time()-t0:.2f}s.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cria pasta de validação visual para NFS-e")
    parser.add_argument("--min-sim", type=float, default=0.88, help="Similaridade mínima (default: 0.88)")
    parser.add_argument("--max", type=int, default=None, help="Limite máximo de candidatos")
    args = parser.parse_args()
    main(min_similarity=args.min_sim, max_candidates=args.max)
