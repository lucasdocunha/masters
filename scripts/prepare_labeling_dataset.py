#!/usr/bin/env python3
"""
Prepara o dataset otimizado para o Estúdio de Rotulação e Tabela CSV na Web App:
1. Extrai mapeamento de URLs oficiais da Câmara para os 1.34M documentos
2. Carrega as anotações existentes e incorpora os novos lotes:
   - boleto (735 confirmados + 265 candidatos de bancos/condomínios)
   - hotelaria (126 confirmados + 874 candidatos)
   - correios (976 confirmados + 24 candidatos)
   - pedagio (81 confirmados) e sem_parar (919 candidatos)
3. Calcula centróides de cada classe no espaço 128-D ArcFace
4. Seleciona os Top 250 candidatos não-mapeados mais próximos da fronteira de cada cluster
5. Salva web_app/data/labeling_dataset.json.gz (~850 KB)
6. Atualiza web_app/data/metadata.json com a paleta completa
"""

import os
import sys
import glob
import json
import gzip
import time
from pathlib import Path
import numpy as np

BASE_DIR = Path("/home/lucas.ocunha/masters")
DATA_DIR = BASE_DIR / "data"
WEB_DATA_DIR = BASE_DIR / "web_app" / "data"
WEB_DATA_DIR.mkdir(parents=True, exist_ok=True)

EXTENDED_PALETTE = {
    "Não Mapeado": {"hex": "#52525b", "rgb": [82, 82, 91]},
    "nfe": {"hex": "#3b82f6", "rgb": [59, 130, 246]},                 # Azul
    "nfse": {"hex": "#22c55e", "rgb": [34, 197, 94]},                # Verde
    "cupom_fiscal": {"hex": "#f97316", "rgb": [249, 115, 22]},        # Laranja
    "nfce": {"hex": "#f43f5e", "rgb": [244, 63, 94]},                # Coral/Rose
    "recibo": {"hex": "#a855f7", "rgb": [168, 85, 247]},             # Roxo
    "uber": {"hex": "#eab308", "rgb": [234, 179, 8]},                # Amarelo
    "energia": {"hex": "#06b6d4", "rgb": [6, 182, 212]},              # Ciano
    "agua": {"hex": "#38bdf8", "rgb": [56, 189, 248]},                # Azul celeste
    "telefonia": {"hex": "#ef4444", "rgb": [239, 68, 68]},            # Vermelho
    "passagem_aerea": {"hex": "#ec4899", "rgb": [236, 72, 153]},      # Rosa
    "passagem_rodoviaria": {"hex": "#10b981", "rgb": [16, 185, 129]}, # Esmeralda
    "pedagio": {"hex": "#b45309", "rgb": [180, 83, 9]},               # Âmbar escuro
    "sem_parar": {"hex": "#f59e0b", "rgb": [245, 158, 11]},           # Dourado
    "hotelaria": {"hex": "#e11d48", "rgb": [225, 29, 72]},            # Vinho/Carmim
    "correios": {"hex": "#ca8a04", "rgb": [202, 138, 4]},             # Mostarda Correios
    "boleto": {"hex": "#6366f1", "rgb": [99, 102, 241]},              # Índigo Boleto
    "iptu": {"hex": "#8b5cf6", "rgb": [139, 92, 246]},                # Violeta
    "fatura": {"hex": "#14b8a6", "rgb": [20, 184, 166]},              # Teal
    "emails": {"hex": "#84cc16", "rgb": [132, 204, 22]},              # Lima
}


def build_or_load_url_cache() -> dict:
    cache_path = DATA_DIR / "stem_to_url.json"
    if cache_path.exists():
        print("   Carregando cache de URLs da Câmara...", flush=True)
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    print("   Extraindo URLs dos JSONs Ano-*.json...", flush=True)
    t0 = time.time()
    urls = {}
    for p in sorted(glob.glob(str(DATA_DIR / "raw" / "Ano-*.json"))):
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        items = data.get("dados", []) if isinstance(data, dict) else data
        for item in items:
            u = item.get("urlDocumento")
            if u and u.lower().endswith(".pdf"):
                stem = Path(u).stem
                if stem not in urls:
                    urls[stem] = u

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(urls, f)
    print(f"   URLs mapeadas: {len(urls):,} em {time.time()-t0:.1f}s", flush=True)
    return urls


def load_all_batches():
    """Carrega anotações prévias e os 4 lotes validados."""
    history_p = BASE_DIR / "ml_backend" / "models" / "annotations_history.json"
    annotations = {}
    if history_p.exists():
        with open(history_p, "r", encoding="utf-8") as f:
            annotations = json.load(f)

    # Lote de Boletos Reais
    boleto_man = Path("/home/lucas.ocunha/Desktop/validacao_boleto_1000/manifest.json")
    if boleto_man.exists():
        with open(boleto_man, "r", encoding="utf-8") as f:
            b_data = json.load(f)
        for m in b_data:
            did = str(m.get("doc_id"))
            if did not in annotations:
                annotations[did] = "boleto"

    # Lote de Hotelaria
    hot_man = Path("/home/lucas.ocunha/Desktop/validacao_hotelaria_1000/manifest.json")
    if hot_man.exists():
        with open(hot_man, "r", encoding="utf-8") as f:
            h_data = json.load(f)
        for m in h_data:
            did = str(m.get("doc_id"))
            if did not in annotations:
                annotations[did] = "hotelaria"

    # Lote de Correios
    cor_man = Path("/home/lucas.ocunha/Desktop/validacao_correios_1000/manifest.json")
    if cor_man.exists():
        with open(cor_man, "r", encoding="utf-8") as f:
            c_data = json.load(f)
        for m in c_data:
            did = str(m.get("doc_id"))
            if did not in annotations:
                annotations[did] = "correios"

    # Lote de Pedágio & Sem Parar
    ped_man = Path("/home/lucas.ocunha/Desktop/validacao_pedagio_1000/manifest.json")
    if ped_man.exists():
        with open(ped_man, "r", encoding="utf-8") as f:
            p_data = json.load(f)
        for m in p_data:
            did = str(m.get("doc_id"))
            if did not in annotations:
                # Se confirmado ocr de praça/eixo = pedagio, senão sem_parar
                if m.get("ocr_match"):
                    annotations[did] = "pedagio"
                else:
                    annotations[did] = "sem_parar"

    return annotations


def main():
    print("=== PREPARAÇÃO DO DATASET DE ROTULAÇÃO ATIVA & TABELA CSV ===")
    t0 = time.time()

    # 1. Cache de URLs
    print("1. Processando URLs oficiais da Câmara...")
    urls = build_or_load_url_cache()

    # 2. Metadados e Embeddings 128-D
    print("2. Carregando embeddings 128-D e metadados de documentos...")
    meta_p = DATA_DIR / "latent_vectors_full_metadata.json"
    embs_p = DATA_DIR / "latent_vectors_metric_learned_128d.npy"
    lookup_p = DATA_DIR / "pdf_metadata_lookup.json"

    with open(meta_p, "r", encoding="utf-8") as f:
        meta = json.load(f)
    docs = meta["documents"]
    n_docs = len(docs)
    stems = [Path(d["doc_name"]).stem for d in docs]
    stem_to_idx = {s: i for i, s in enumerate(stems)}

    with open(lookup_p, "r", encoding="utf-8") as f:
        lookup = json.load(f)

    embs = np.load(embs_p)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs /= norms

    # 3. Anotações consolidadas
    print("3. Consolidando anotações com os novos lotes validados...")
    annotations = load_all_batches()
    print(f"   Total de documentos anotados disponíveis: {len(annotations):,}")

    # Salvar anotações atualizadas
    with open(BASE_DIR / "ml_backend" / "models" / "annotations_history.json", "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)

    # 4. Calcular Centróides de cada classe
    print("4. Calculando centróides para cada cluster no espaço 128-D...")
    known_classes = sorted(list(set(annotations.values())))
    centroids = {}
    class_indices = {}

    for c in known_classes:
        idxs = [stem_to_idx[s] for s, lbl in annotations.items() if lbl == c and s in stem_to_idx]
        if len(idxs) > 0:
            c_vec = embs[idxs].mean(axis=0)
            c_vec /= np.linalg.norm(c_vec)
            centroids[c] = c_vec
            class_indices[c] = idxs
            print(f"   Classe '{c:20s}': {len(idxs):5d} docs | Centróide 128-D calculado")

    # 5. Para cada cluster, encontrar os top 250 candidatos não-mapeados mais próximos
    print("5. Minerando os Top 250 candidatos não-mapeados mais próximos de cada cluster...")
    unmapped_indices = [i for i in range(n_docs) if stems[i] not in annotations]
    unmapped_arr = np.array(unmapped_indices, dtype=np.int32)
    unmapped_embs = embs[unmapped_arr]

    top_candidates_per_cluster = {}
    candidate_indices_set = set()

    for c, c_vec in centroids.items():
        sims = np.dot(unmapped_embs, c_vec)
        top_k = np.argsort(-sims)[:250]
        selected_unmapped = [unmapped_arr[k] for k in top_k]
        top_candidates_per_cluster[c] = [(idx, float(sims[k])) for k, idx in zip(top_k, selected_unmapped)]
        candidate_indices_set.update(selected_unmapped)

    print(f"   Candidatos ativos selecionados para rotulação: {len(candidate_indices_set):,}")

    # 6. Montar o Dataset Final
    print("6. Compilando registros do dataset de rotulação...")
    final_records = []

    # A) Documentos Anotados
    for stem, lbl in annotations.items():
        if stem not in stem_to_idx:
            continue
        idx = stem_to_idx[stem]
        dname = docs[idx]["doc_name"]
        info = lookup.get(dname, {})
        forn = info.get("fornecedor", "N/A")
        sub = info.get("subcota", "N/A")
        ano = info.get("ano", 2021)
        url = urls.get(stem, f"https://www.camara.leg.br/cota-parlamentar/documentos/publ/{ano}/{dname}")

        # Calcular similaridade com o centróide da classe
        sim = float(np.dot(embs[idx], centroids[lbl])) if lbl in centroids else 1.0

        final_records.append({
            "id": str(stem),
            "name": str(dname),
            "class": str(lbl),
            "fornecedor": str(forn),
            "subcota": str(sub),
            "ano": int(ano),
            "url": str(url),
            "closest_cluster": str(lbl),
            "similarity": float(round(sim, 4)),
            "coords_idx": int(idx),
            "status": "rotulado"
        })

    # B) Documentos Candidatos Não Mapeados
    for idx in candidate_indices_set:
        stem = stems[idx]
        dname = docs[idx]["doc_name"]
        info = lookup.get(dname, {})
        forn = info.get("fornecedor", "N/A")
        sub = info.get("subcota", "N/A")
        ano = info.get("ano", 2021)
        url = urls.get(stem, f"https://www.camara.leg.br/cota-parlamentar/documentos/publ/{ano}/{dname}")

        # Encontrar o cluster mais próximo entre todos
        best_sim = -1.0
        best_cluster = "Não Mapeado"
        for c, c_vec in centroids.items():
            sim = float(np.dot(embs[idx], c_vec))
            if sim > best_sim:
                best_sim = sim
                best_cluster = c

        final_records.append({
            "id": str(stem),
            "name": str(dname),
            "class": "Não Mapeado",
            "fornecedor": str(forn),
            "subcota": str(sub),
            "ano": int(ano),
            "url": str(url),
            "closest_cluster": str(best_cluster),
            "similarity": float(round(best_sim, 4)),
            "coords_idx": int(idx),
            "status": "candidato"
        })

    print(f"   Total de registros no dataset de rotulação: {len(final_records):,}")

    # Salvar comprimido em gzip
    out_gz = WEB_DATA_DIR / "labeling_dataset.json.gz"
    out_json = WEB_DATA_DIR / "labeling_dataset.json"

    data_bytes = json.dumps(final_records, ensure_ascii=False).encode("utf-8")
    with gzip.open(out_gz, "wb", compresslevel=6) as f:
        f.write(data_bytes)

    gz_size_mb = out_gz.stat().st_size / (1024 * 1024)
    print(f"   Salvo: {out_gz} ({gz_size_mb:.2f} MB)")

    # 7. Atualizar metadata.json com as novas classes e contagens
    meta_json_path = WEB_DATA_DIR / "metadata.json"
    with open(meta_json_path, "r", encoding="utf-8") as f:
        meta_web = json.load(f)

    # Reconstruir paleta
    class_counts = {}
    for r in final_records:
        c = r["class"]
        class_counts[c] = class_counts.get(c, 0) + 1

    new_palette = {}
    class_order = ["Não Mapeado"] + sorted([c for c in EXTENDED_PALETTE.keys() if c != "Não Mapeado"])
    for cid, cname in enumerate(class_order):
        pal_entry = EXTENDED_PALETTE.get(cname, {"hex": "#a1a1aa", "rgb": [161, 161, 170]})
        rgb_norm = [round(x / 255.0, 4) for x in pal_entry["rgb"]]
        new_palette[str(cid)] = {
            "name": cname,
            "hex": pal_entry["hex"],
            "rgb": rgb_norm,
            "count": class_counts.get(cname, 0)
        }

    meta_web["palette"] = new_palette
    meta_web["total_annotated"] = sum(v for k, v in class_counts.items() if k != "Não Mapeado")
    meta_web["total_labeling_candidates"] = len(candidate_indices_set)

    with open(meta_json_path, "w", encoding="utf-8") as f:
        json.dump(meta_web, f, indent=2, ensure_ascii=False)

    print(f"   Atualizado: {meta_json_path}")
    print("\n==================================================================")
    print(" DATASET DE ROTULAÇÃO GERADO COM SUCESSO!")
    print(f" Total de Documentos: {len(final_records):,}")
    print(f"   - Anotados: {len(annotations):,}")
    print(f"   - Candidatos Ativos: {len(candidate_indices_set):,}")
    print(f" Tamanho do Arquivo: {gz_size_mb:.2f} MB")
    print(f" Tempo Total: {time.time()-t0:.1f}s")
    print("==================================================================")


if __name__ == "__main__":
    main()
