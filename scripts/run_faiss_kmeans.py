#!/usr/bin/env python3
"""
Execução de FAISS K-Means (K=10) no dataset global do CEAP (1.339.678 documentos).
Realiza clusterização esférica sobre os embeddings multimodais (ColPali VLM 128-d e DINOv2 384-d),
analisa semanticamente os clusters cruzando com as subcotas e fornecedores oficiais da CEAP,
e exporta as 20 amostras mais representativas (mais próximas ao centroide) para inspeção visual.

Estrutura de saída:
  data/latent_vectors_faiss_k10.npz
  data/pdf_metadata_lookup.json (cache de subcotas/fornecedores)
  data/cluster-faiss-vlm-k10/{0..9}/
    ├── 01_{doc_stem}.png
    ├── 01_{doc_stem}.pdf -> symlink para o PDF original
    └── ... (20 amostras)
    ├── summary.json
    └── index.html (galeria interativa)
  data/cluster-faiss-dino-k10/{0..9}/
    └── ...
  data/clusters_k10_comparison.html (dashboard comparativo VLM vs DINOv2)
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import faiss
import numpy as np

# Nota: pymupdf é importado sob demanda dentro de render_pdf_first_page
# para evitar colisões no registro global SWIG com o faiss.

try:
    import psycopg
except ImportError:
    psycopg = None


# Mapeamento semântico heurístico baseado em subcotas da Câmara dos Deputados
SUCOTA_TAXONOMY_MAP = {
    "COMBUSTÍVEIS E LUBRIFICANTES": "CUPOM_FISCAL_COMBUSTIVEL",
    "PASSAGEM AÉREA": "BILHETE_AEREO_EMBARQUE",
    "PASSAGENS AÉREAS": "BILHETE_AEREO_EMBARQUE",
    "TELEFONIA": "FATURA_TELECOMUNICACAO",
    "SERVIÇOS POSTAIS": "COMPROVANTE_POSTAL_CORREIOS",
    "LOCAÇÃO OU FRETAMENTO DE VEÍCULOS AUTOMOTORES": "CONTRATO_RECIBO_LOCACAO",
    "LOCAÇÃO DE VEÍCULOS": "CONTRATO_RECIBO_LOCACAO",
    "SERVIÇO DE TÁXI, PEDÁGIO E ESTACIONAMENTO": "RECIBO_TAXI_ESTACIONAMENTO",
    "HOSPEDAGEM ,EXCETO DO PARLAMENTAR NO DISTRITO FEDERAL": "NOTA_FISCAL_HOSPEDAGEM",
    "HOSPEDAGEM": "NOTA_FISCAL_HOSPEDAGEM",
    "FORNECIMENTO DE ALIMENTAÇÃO DO PARLAMENTAR": "CUPOM_REFEICAO_ALIMENTACAO",
    "DIVULGAÇÃO DA ATIVIDADE PARLAMENTAR": "NOTA_FISCAL_SERVICOS_NFSE",
    "CONSULTORIAS, PESQUISAS E TRABALHOS TÉCNICOS": "NOTA_FISCAL_CONSULTORIA",
    "MANUTENÇÃO DE ESCRITÓRIO DE APOIO À ATIVIDADE PARLAMENTAR": "DANFE_A4_OU_FATURA",
    "SERVIÇO DE SEGURANÇA PRESTADO POR EMPRESA ESPECIALIZADA": "NOTA_FISCAL_SEGURANCA",
    "PARTICIPAÇÃO EM CURSO, PALESTRA OU EVENTO SIMILAR": "COMPROVANTE_INSCRICAO_EVENTO",
}


def infer_cluster_taxonomy(dominant_subcota: str, top_fornecedores: list[tuple[str, int]]) -> str:
    """Infere o nome da classe documental a partir da subcota predominante e fornecedores."""
    sub_upper = dominant_subcota.upper().strip()
    for key, tax in SUCOTA_TAXONOMY_MAP.items():
        if key in sub_upper:
            return tax

    # Heurística por fornecedores
    if top_fornecedores:
        top_names = " ".join([f[0].upper() for f in top_fornecedores[:3]])
        if any(a in top_names for a in ["GOL", "LATAM", "AZUL", "AVIANCA", "TAM"]):
            return "BILHETE_AEREO_EMBARQUE"
        if any(c in top_names for c in ["CLARO", "VIVO", "TIM", "OI", "TELEFONICA", "EMBRATEL"]):
            return "FATURA_TELECOMUNICACAO"
        if any(p in top_names for p in ["POSTO", "COMBUSTIVEL", "IPIRANGA", "SHELL", "PETROBRAS"]):
            return "CUPOM_FISCAL_COMBUSTIVEL"
        if any(u in top_names for u in ["UBER", "99", "TAXI", "CABIFY"]):
            return "RECIBO_APP_TRANSPORTE"
        if any(l in top_names for l in ["LOCALIZA", "MOVIDA", "UNIDAS", "RENT A CAR"]):
            return "CONTRATO_RECIBO_LOCACAO"
        if any(h in top_names for h in ["HOTEL", "POUSADA", "FLAT", "INN"]):
            return "NOTA_FISCAL_HOSPEDAGEM"

    # Fallback
    clean_sub = "".join(c if c.isalnum() else "_" for c in dominant_subcota.upper())
    clean_sub = "_".join(filter(None, clean_sub.split("_")))
    return clean_sub[:30] if clean_sub else "DOCUMENTO_DIVERSO"


def render_pdf_first_page(pdf_path: Path, output_png: Path, dpi: int = 150) -> bool:
    """Renderiza a 1ª página de um PDF em alta resolução PNG."""
    try:
        import pymupdf
        doc = pymupdf.open(pdf_path)
        if len(doc) == 0:
            doc.close()
            return False
        page = doc[0]
        pix = page.get_pixmap(dpi=dpi)
        output_png.parent.mkdir(parents=True, exist_ok=True)
        pix.save(str(output_png))
        doc.close()
        return True
    except Exception as e:
        print(f"    [Aviso] Falha ao renderizar {pdf_path.name}: {e}")
        return False


def get_or_create_metadata_lookup(raw_dir: Path, cache_file: Path) -> dict[str, dict[str, Any]]:
    """Carrega ou constrói índice de metadados {pdf_filename: {subcota, fornecedor, ano}}."""
    if cache_file.exists():
        print(f"Carregando cache de metadados de {cache_file}...")
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[Aviso] Erro ao ler cache {cache_file}: {e}. Reconstruindo...")

    print(f"Indexando metadados dos arquivos brutos em {raw_dir} (Ano-*.json)...")
    t0 = time.time()
    lookup = {}
    ano_files = sorted(raw_dir.glob("Ano-*.json"))
    for p in ano_files:
        try:
            fallback_year = int(p.stem.split("-")[-1]) if p.stem.split("-")[-1].isdigit() else 0
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
            items = d.get("dados", []) if isinstance(d, dict) else d
            for it in items:
                url = it.get("urlDocumento") or ""
                if url and url.lower().endswith(".pdf"):
                    fname = url.split("/")[-1]
                    if fname not in lookup:
                        subcota = (it.get("descricao") or "").strip()
                        fornecedor = (it.get("fornecedor") or "").strip()
                        ano = int(it.get("ano", fallback_year))
                        lookup[fname] = {
                            "subcota": subcota,
                            "fornecedor": fornecedor,
                            "ano": ano,
                        }
        except Exception as e:
            print(f"[Aviso] Erro ao processar {p.name}: {e}")

    t1 = time.time()
    print(f"Indexados {len(lookup):,} documentos em {t1-t0:.2f}s.")

    if lookup:
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(lookup, f)
            print(f"Cache de metadados salvo em: {cache_file}")
        except Exception as e:
            print(f"[Aviso] Falha ao salvar cache: {e}")

    return lookup


def run_faiss_spherical_kmeans(
    embeddings: np.ndarray,
    k: int = 10,
    niter: int = 25,
    seed: int = 42,
    max_points_per_centroid: int = 150000,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Executa spherical FAISS K-Means sobre os embeddings normalizados L2.
    
    Retorna:
      labels: (N,) int32 - índice do cluster atribuído
      similarities: (N,) float32 - similaridade de cosseno com o centroide atribuído [0..1]
      centroids: (k, D) float32 - vetores centroides normalizados
    """
    n_docs, dim = embeddings.shape
    print(f"Iniciando FAISS Spherical K-Means: N={n_docs:,}, D={dim}, K={k}, niter={niter}...")
    t0 = time.time()

    # Cópia para não modificar o array original e garantir C-contiguous float32
    embs_norm = np.ascontiguousarray(embeddings, dtype=np.float32)
    faiss.normalize_L2(embs_norm)

    kmeans = faiss.Kmeans(
        d=dim,
        k=k,
        niter=niter,
        verbose=verbose,
        spherical=True,
        seed=seed,
        max_points_per_centroid=max_points_per_centroid,
    )
    kmeans.train(embs_norm)
    t_train = time.time()
    print(f"Treinamento FAISS concluído em {t_train - t0:.2f}s.")

    # Busca exata do centroide mais próximo usando inner product (cosseno)
    t_search_start = time.time()
    D, I = kmeans.index.search(embs_norm, 1)
    t_search_end = time.time()
    print(f"Atribuição dos {n_docs:,} vetores concluída em {t_search_end - t_search_start:.2f}s.")

    labels = I.flatten().astype(np.int32)
    similarities = D.flatten().astype(np.float32)
    centroids = kmeans.centroids.astype(np.float32)

    return labels, similarities, centroids


def analyze_clusters(
    labels: np.ndarray,
    similarities: np.ndarray,
    documents: list[dict],
    meta_lookup: dict[str, dict[str, Any]],
    k: int = 10,
) -> dict[int, dict[str, Any]]:
    """Gera perfil semântico e estatístico completo de cada cluster."""
    n_total = len(labels)
    analysis = {}

    for c_id in range(k):
        mask = np.where(labels == c_id)[0]
        count = len(mask)
        pct = (count / n_total) * 100 if n_total > 0 else 0.0

        c_sims = similarities[mask]
        mean_sim = float(np.mean(c_sims)) if count > 0 else 0.0
        min_sim = float(np.min(c_sims)) if count > 0 else 0.0
        max_sim = float(np.max(c_sims)) if count > 0 else 0.0

        subcotas = []
        fornecedores = []
        years = []

        for idx in mask:
            doc_info = documents[idx]
            fname = doc_info.get("doc_name", "")
            meta = meta_lookup.get(fname, {})
            sub = meta.get("subcota")
            if sub:
                subcotas.append(sub)
            forn = meta.get("fornecedor")
            if forn:
                fornecedores.append(forn)
            yr = meta.get("ano")
            if yr:
                years.append(yr)

        sub_counts = Counter(subcotas).most_common(5)
        forn_counts = Counter(fornecedores).most_common(5)
        yr_counts = Counter(years).most_common(5)

        dominant_subcota = sub_counts[0][0] if sub_counts else "N/A"
        dominant_subcota_pct = (sub_counts[0][1] / len(subcotas) * 100) if subcotas else 0.0

        inferred_tax = infer_cluster_taxonomy(dominant_subcota, forn_counts)

        analysis[c_id] = {
            "cluster_id": c_id,
            "total_docs": count,
            "percentage": pct,
            "similarity_mean": mean_sim,
            "similarity_min": min_sim,
            "similarity_max": max_sim,
            "distance_mean": 1.0 - mean_sim,
            "inferred_taxonomy": inferred_tax,
            "dominant_subcota": dominant_subcota,
            "dominant_subcota_pct": dominant_subcota_pct,
            "top_subcotas": [(s, c, round((c / len(subcotas)) * 100, 1)) for s, c in sub_counts] if subcotas else [],
            "top_fornecedores": [(f, c, round((c / len(fornecedores)) * 100, 1)) for f, c in forn_counts] if fornecedores else [],
            "top_years": [(y, c, round((c / len(years)) * 100, 1)) for y, c in yr_counts] if years else [],
        }

    return analysis


def generate_html_gallery_k10(
    tech_name: str,
    tech_display: str,
    k: int,
    cluster_analysis: dict[int, dict[str, Any]],
    exported_data: dict[int, list[dict[str, Any]]],
    output_html: Path,
) -> None:
    """Gera visualizador HTML interativo para os K clusters com metadados semânticos."""
    sections_html = []
    nav_links = []

    for c_id in range(k):
        c_meta = cluster_analysis[c_id]
        samples = exported_data.get(c_id, [])
        tax = c_meta["inferred_taxonomy"]
        pct = c_meta["percentage"]
        total_docs = c_meta["total_docs"]
        dist_mean = c_meta["distance_mean"]

        nav_links.append(f'<a href="#cluster-{c_id}" class="nav-btn">Cluster {c_id}: {tax}</a>')

        # Subcota pills
        top_subs_pills = " ".join([
            f'<span class="badge badge-sub" title="{s} ({p}%)">{s[:28]}... ({p:.0f}%)</span>'
            for s, _, p in c_meta["top_subcotas"][:3]
        ])
        # Fornecedor pills
        top_forns_pills = " ".join([
            f'<span class="badge badge-forn" title="{f} ({p}%)">{f[:24]}... ({p:.0f}%)</span>'
            for f, _, p in c_meta["top_fornecedores"][:3]
        ])

        cards_html = []
        for s in samples:
            rank = s["rank"]
            doc_name = s["doc_name"]
            img_rel = f"{c_id}/{s['image']}"
            pdf_rel = f"{c_id}/{s['pdf']}"
            dist = s["dist_to_centroid"]
            pages = s["num_pages"]
            year = s["year"]
            sub = s.get("subcota", "N/A")
            forn = s.get("fornecedor", "N/A")

            card = f"""
            <div class="card">
              <a href="{img_rel}" target="_blank" title="Clique para ampliar imagem">
                <img src="{img_rel}" alt="{doc_name}" loading="lazy">
              </a>
              <div class="card-info">
                <div class="doc-rank">#{rank:02d} &bull; {year}</div>
                <div class="doc-name" title="{doc_name}">{doc_name}</div>
                <div class="doc-subcota" title="{sub}">{sub}</div>
                <div class="doc-forn" title="{forn}">🏢 {forn}</div>
                <div class="meta-row">
                  <span>Dist: <b>{dist:.3f}</b></span>
                  <span>{pages} pág(s)</span>
                </div>
                <div class="actions">
                  <a href="{pdf_rel}" target="_blank" class="btn-pdf">Ver PDF</a>
                  <a href="{img_rel}" target="_blank" class="btn-img">Imagem</a>
                </div>
              </div>
            </div>
            """
            cards_html.append(card)

        grid_html = "\n".join(cards_html)
        section = f"""
        <div class="cluster-section" id="cluster-{c_id}">
          <div class="cluster-header">
            <div>
              <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                <span class="cluster-title">Cluster {c_id}: {tax}</span>
                <span class="cluster-badge">{total_docs:,} docs ({pct:.1f}%)</span>
                <span class="cluster-badge-dist">Dist. Média: {dist_mean:.3f}</span>
              </div>
              <div class="pills-container">
                <span style="font-size: 11px; color: #94a3b8; font-weight: 600;">Subcotas:</span> {top_subs_pills}
                <span style="font-size: 11px; color: #94a3b8; font-weight: 600; margin-left: 10px;">Fornecedores:</span> {top_forns_pills}
              </div>
            </div>
            <div class="cluster-nav-anchor">
              <a href="#top">&uarr; Voltar ao Topo</a>
            </div>
          </div>
          <div class="grid">
            {grid_html}
          </div>
        </div>
        """
        sections_html.append(section)

    all_sections = "\n".join(sections_html)
    all_nav = "\n".join(nav_links)

    html_content = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Galeria FAISS K={k} &bull; {tech_display}</title>
<style>
  :root {{
    --bg-main: #0b0f19;
    --bg-card: #151d30;
    --border-color: #263352;
    --primary: #38bdf8;
    --primary-hover: #0ea5e9;
    --accent: #a855f7;
    --text-main: #f1f5f9;
    --text-muted: #94a3b8;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    background: var(--bg-main);
    color: var(--text-main);
    margin: 0;
    padding: 24px;
  }}
  header {{
    max-width: 1440px;
    margin: 0 auto 28px;
    padding-bottom: 20px;
    border-bottom: 1px solid var(--border-color);
  }}
  h1 {{
    font-size: 26px;
    margin: 0 0 8px;
    color: var(--primary);
  }}
  p.desc {{
    color: var(--text-muted);
    font-size: 14px;
    margin: 0 0 16px;
    line-height: 1.5;
  }}
  .nav-bar {{
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    align-items: center;
    margin-top: 12px;
  }}
  .nav-btn {{
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    color: var(--text-main);
    padding: 6px 12px;
    border-radius: 6px;
    text-decoration: none;
    font-size: 12px;
    font-weight: 500;
    transition: all 0.2s;
  }}
  .nav-btn:hover {{
    background: var(--primary);
    color: #000;
    border-color: var(--primary);
  }}
  .container {{
    max-width: 1440px;
    margin: 0 auto;
  }}
  .cluster-section {{
    background: #111827;
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 36px;
  }}
  .cluster-header {{
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    border-bottom: 1px solid var(--border-color);
    padding-bottom: 16px;
    margin-bottom: 20px;
  }}
  .cluster-title {{
    font-size: 20px;
    font-weight: 700;
    color: #ffffff;
  }}
  .cluster-badge {{
    background: #1e3a8a;
    color: #93c5fd;
    padding: 4px 10px;
    border-radius: 9999px;
    font-size: 12px;
    font-weight: 600;
  }}
  .cluster-badge-dist {{
    background: #312e81;
    color: #c7d2fe;
    padding: 4px 10px;
    border-radius: 9999px;
    font-size: 12px;
    font-weight: 600;
  }}
  .pills-container {{
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    align-items: center;
    margin-top: 6px;
  }}
  .badge {{
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 500;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 250px;
  }}
  .badge-sub {{
    background: #064e3b;
    color: #6ee7b7;
    border: 1px solid #047857;
  }}
  .badge-forn {{
    background: #4c1d95;
    color: #ddd6fe;
    border: 1px solid #6d28d9;
  }}
  .cluster-nav-anchor a {{
    color: var(--text-muted);
    text-decoration: none;
    font-size: 12px;
  }}
  .cluster-nav-anchor a:hover {{
    color: var(--primary);
  }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(230px, 1fr));
    gap: 16px;
  }}
  .card {{
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    transition: transform 0.2s, border-color 0.2s, box-shadow 0.2s;
  }}
  .card:hover {{
    transform: translateY(-4px);
    border-color: var(--primary);
    box-shadow: 0 8px 20px rgba(0,0,0,0.4);
  }}
  .card img {{
    width: 100%;
    height: 290px;
    object-fit: contain;
    background: #ffffff;
    display: block;
    border-bottom: 1px solid var(--border-color);
  }}
  .card-info {{
    padding: 12px;
    display: flex;
    flex-direction: column;
    gap: 5px;
    font-size: 12px;
  }}
  .doc-rank {{
    font-size: 11px;
    font-weight: 700;
    color: #38bdf8;
    text-transform: uppercase;
  }}
  .doc-name {{
    font-weight: 600;
    color: #ffffff;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }}
  .doc-subcota {{
    color: #34d399;
    font-size: 11px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    font-weight: 500;
  }}
  .doc-forn {{
    color: #c084fc;
    font-size: 11px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }}
  .meta-row {{
    display: flex;
    justify-content: space-between;
    color: var(--text-muted);
    font-size: 11px;
    margin-top: 2px;
  }}
  .actions {{
    display: flex;
    gap: 8px;
    margin-top: 6px;
  }}
  .actions a {{
    flex: 1;
    text-align: center;
    padding: 5px 0;
    border-radius: 4px;
    text-decoration: none;
    font-size: 11px;
    font-weight: 600;
    transition: background 0.15s;
  }}
  .btn-pdf {{
    background: #334155;
    color: #f1f5f9;
  }}
  .btn-pdf:hover {{
    background: #475569;
  }}
  .btn-img {{
    background: #0284c7;
    color: #ffffff;
  }}
  .btn-img:hover {{
    background: var(--primary-hover);
  }}
</style>
</head>
<body id="top">
<header>
  <h1>Galeria FAISS Spherical K-Means (K={k}) &bull; {tech_display}</h1>
  <p class="desc">
    20 documentos mais representativos (mais próximos do centroide no espaço latente esférico) para cada uma das {k} classes identificadas no dataset global do CEAP (1.339.678 documentos).
  </p>
  <div class="nav-bar">
    <span style="font-size: 13px; color: var(--text-muted); margin-right: 4px;">Ir para:</span>
    {all_nav}
    <a href="../clusters_k10_comparison.html" class="nav-btn" style="margin-left: auto; background: #4f46e5; border-color: #6366f1;">&harr; Comparativo VLM vs DINOv2</a>
    <a href="../latent_vectors_full_3d.html" class="nav-btn" style="background: #2563eb; border-color: #3b82f6;">&larr; Espaço 3D UMAP</a>
  </div>
</header>

<div class="container">
  {all_sections}
</div>
</body>
</html>
"""
    output_html.parent.mkdir(parents=True, exist_ok=True)
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"  [HTML] Galeria gerada: {output_html}")


def generate_clusters_comparison_dashboard(
    k: int,
    vlm_analysis: dict[int, dict[str, Any]],
    dino_analysis: dict[int, dict[str, Any]],
    output_file: Path,
) -> None:
    """Gera página de comparação lado-a-lado das 10 classes identificadas por ColPali VLM vs DINOv2."""
    vlm_rows = []
    for c_id in range(k):
        m = vlm_analysis[c_id]
        vlm_rows.append(f"""
        <tr>
          <td><b>Cluster {c_id}</b></td>
          <td><span class="badge-class">{m['inferred_taxonomy']}</span></td>
          <td><b>{m['total_docs']:,}</b> ({m['percentage']:.1f}%)</td>
          <td title="{m['dominant_subcota']}">{m['dominant_subcota'][:32]}... ({m['dominant_subcota_pct']:.0f}%)</td>
          <td>{m['distance_mean']:.3f}</td>
          <td><a href="cluster-faiss-vlm-k10/index.html#cluster-{c_id}" class="tbl-link">Ver Amostras &rarr;</a></td>
        </tr>
        """)

    dino_rows = []
    for c_id in range(k):
        m = dino_analysis[c_id]
        dino_rows.append(f"""
        <tr>
          <td><b>Cluster {c_id}</b></td>
          <td><span class="badge-class">{m['inferred_taxonomy']}</span></td>
          <td><b>{m['total_docs']:,}</b> ({m['percentage']:.1f}%)</td>
          <td title="{m['dominant_subcota']}">{m['dominant_subcota'][:32]}... ({m['dominant_subcota_pct']:.0f}%)</td>
          <td>{m['distance_mean']:.3f}</td>
          <td><a href="cluster-faiss-dino-k10/index.html#cluster-{c_id}" class="tbl-link">Ver Amostras &rarr;</a></td>
        </tr>
        """)

    vlm_tbody = "\n".join(vlm_rows)
    dino_tbody = "\n".join(dino_rows)

    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Comparativo FAISS K={k} &bull; ColPali VLM vs DINOv2</title>
<style>
  :root {{
    --bg-main: #0b0f19;
    --bg-card: #151d30;
    --border-color: #263352;
    --primary: #38bdf8;
    --secondary: #818cf8;
    --text-main: #f1f5f9;
    --text-muted: #94a3b8;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    background: var(--bg-main);
    color: var(--text-main);
    margin: 0;
    padding: 24px;
  }}
  .container {{ max-width: 1440px; margin: 0 auto; }}
  header {{
    margin-bottom: 28px;
    padding-bottom: 20px;
    border-bottom: 1px solid var(--border-color);
  }}
  h1 {{ font-size: 26px; color: var(--primary); margin: 0 0 8px; }}
  p.desc {{ color: var(--text-muted); font-size: 14px; margin: 0 0 16px; line-height: 1.5; }}
  .grid-two {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
    margin-bottom: 32px;
  }}
  @media (max-width: 1024px) {{
    .grid-two {{ grid-template-columns: 1fr; }}
  }}
  .panel {{
    background: #111827;
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 20px;
  }}
  .panel-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
    border-bottom: 1px solid var(--border-color);
    padding-bottom: 12px;
  }}
  .panel-title {{
    font-size: 18px;
    font-weight: 700;
  }}
  .panel-title.vlm {{ color: #38bdf8; }}
  .panel-title.dino {{ color: #a78bfa; }}
  .btn-gallery {{
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    color: var(--text-main);
    padding: 5px 12px;
    border-radius: 6px;
    text-decoration: none;
    font-size: 12px;
    font-weight: 600;
  }}
  .btn-gallery:hover {{ background: var(--primary); color: #000; }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
  }}
  th, td {{
    padding: 10px 8px;
    text-align: left;
    border-bottom: 1px solid #1e293b;
  }}
  th {{ color: var(--text-muted); font-weight: 600; }}
  tr:hover {{ background: rgba(255,255,255,0.02); }}
  .badge-class {{
    background: #1e293b;
    color: #38bdf8;
    padding: 2px 6px;
    border-radius: 4px;
    font-weight: 600;
    font-family: monospace;
    font-size: 11px;
  }}
  .tbl-link {{
    color: #38bdf8;
    text-decoration: none;
    font-weight: 600;
  }}
  .tbl-link:hover {{ text-decoration: underline; }}
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>Dashboard Comparativo FAISS K={k} &bull; ColPali VLM vs DINOv2</h1>
    <p class="desc">
      Identificação e caracterização semântica das {k} classes dominantes no CEAP (1.339.678 documentos),
      contrastando a separabilidade multimodal (ColPali VLM) com a representação estrutural-visual (DINOv2).
    </p>
    <div>
      <a href="cluster-faiss-vlm-k10/index.html" class="btn-gallery" style="background: #0284c7; color: #fff;">Ver Galeria ColPali VLM (K={k}) &rarr;</a>
      <a href="cluster-faiss-dino-k10/index.html" class="btn-gallery" style="background: #7c3aed; color: #fff; margin-left: 8px;">Ver Galeria DINOv2 (K={k}) &rarr;</a>
      <a href="latent_vectors_full_3d.html" class="btn-gallery" style="margin-left: 8px;">Espaço 3D UMAP &rarr;</a>
    </div>
  </header>

  <div class="grid-two">
    <div class="panel">
      <div class="panel-header">
        <span class="panel-title vlm">ColPali VLM (Visão Multimodal - 128D)</span>
        <a href="cluster-faiss-vlm-k10/index.html" class="btn-gallery">Abrir Galeria &rarr;</a>
      </div>
      <table>
        <thead>
          <tr>
            <th>Cluster</th>
            <th>Classe Sugerida</th>
            <th>Documentos</th>
            <th>Subcota Dominante</th>
            <th>Dist. Média</th>
            <th>Ação</th>
          </tr>
        </thead>
        <tbody>
          {vlm_tbody}
        </tbody>
      </table>
    </div>

    <div class="panel">
      <div class="panel-header">
        <span class="panel-title dino">DINOv2 (Estrutura Visual / Layout - 384D)</span>
        <a href="cluster-faiss-dino-k10/index.html" class="btn-gallery">Abrir Galeria &rarr;</a>
      </div>
      <table>
        <thead>
          <tr>
            <th>Cluster</th>
            <th>Classe Sugerida</th>
            <th>Documentos</th>
            <th>Subcota Dominante</th>
            <th>Dist. Média</th>
            <th>Ação</th>
          </tr>
        </thead>
        <tbody>
          {dino_tbody}
        </tbody>
      </table>
    </div>
  </div>
</div>
</body>
</html>
"""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[HTML] Dashboard comparativo gerado em: {output_file}")


def export_samples_for_technique(
    tech_name: str,
    tech_display: str,
    labels: np.ndarray,
    similarities: np.ndarray,
    documents: list[dict],
    meta_lookup: dict[str, dict[str, Any]],
    pdf_dir: Path,
    output_dir: Path,
    k: int = 10,
    samples_per_cluster: int = 20,
) -> dict[int, list[dict[str, Any]]]:
    """Exporta amostras representativas (PNG renderizado + symlink PDF) e gera galeria."""
    output_dir.mkdir(parents=True, exist_ok=True)
    exported = {}

    print(f"\n=======================================================")
    print(f"Exportando {samples_per_cluster} amostras por cluster para: {tech_display}")
    print(f"Diretório de saída: {output_dir}")
    print(f"=======================================================")

    for c_id in range(k):
        mask = np.where(labels == c_id)[0]
        c_sims = similarities[mask]
        sorted_indices = np.argsort(-c_sims)  # maior similaridade primeiro

        cluster_dir = output_dir / str(c_id)
        cluster_dir.mkdir(parents=True, exist_ok=True)

        samples = []
        rank = 1

        for local_idx in sorted_indices:
            global_idx = mask[local_idx]
            doc_info = documents[global_idx]
            doc_name = doc_info.get("doc_name", "")
            pdf_path = pdf_dir / doc_name

            if not pdf_path.exists():
                continue

            doc_stem = pdf_path.stem
            png_name = f"{rank:02d}_{doc_stem}.png"
            pdf_symlink_name = f"{rank:02d}_{doc_stem}.pdf"

            out_png = cluster_dir / png_name
            out_pdf_symlink = cluster_dir / pdf_symlink_name

            # 1. Renderizar primeira página
            rendered = render_pdf_first_page(pdf_path, out_png, dpi=150)
            if not rendered:
                continue

            # 2. Criar symlink para PDF original
            try:
                if out_pdf_symlink.is_symlink() or out_pdf_symlink.exists():
                    out_pdf_symlink.unlink()
                os.symlink(os.path.abspath(pdf_path), out_pdf_symlink)
            except Exception as e:
                print(f"    [Aviso] Falha ao criar symlink para PDF: {e}")

            meta = meta_lookup.get(doc_name, {})
            dist_val = 1.0 - float(c_sims[local_idx])

            sample_item = {
                "rank": rank,
                "doc_id": int(doc_info.get("doc_id", global_idx)),
                "doc_name": doc_name,
                "year": meta.get("ano", doc_info.get("year", "N/A")),
                "subcota": meta.get("subcota", "N/A"),
                "fornecedor": meta.get("fornecedor", "N/A"),
                "num_pages": doc_info.get("num_pages", 1),
                "dist_to_centroid": dist_val,
                "image": png_name,
                "pdf": pdf_symlink_name,
            }
            samples.append(sample_item)
            print(f"  Cluster {c_id} #{rank:02d}: {doc_name} (Dist: {dist_val:.3f}, {sample_item['subcota'][:30]})")

            rank += 1
            if len(samples) >= samples_per_cluster:
                break

        exported[c_id] = samples

    return exported


def update_database_clusters(
    documents: list[dict],
    labels_vlm: np.ndarray,
    sim_vlm: np.ndarray,
    vlm_analysis: dict[int, dict[str, Any]],
    db_url: str = "postgresql://postgres:postgres@localhost:5432/ceap",
) -> None:
    """Atualiza a tabela 'documento' no PostgreSQL com os clusters K=10 e classes inferidas."""
    if psycopg is None:
        print("[Aviso] psycopg não instalado. Pulando atualização do PostgreSQL.")
        return

    print("\nAtualizando tabela 'documento' no PostgreSQL com as classificações K=10...")
    try:
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                batch = []
                for idx, doc_info in enumerate(documents):
                    fname = doc_info.get("doc_name", "")
                    doc_stem = fname.replace(".pdf", "")
                    if doc_stem.isdigit():
                        ide_doc = int(doc_stem)
                        c_id = int(labels_vlm[idx])
                        dist = float(1.0 - sim_vlm[idx])
                        classe = vlm_analysis[c_id]["inferred_taxonomy"]
                        conf = float(sim_vlm[idx])
                        batch.append((c_id, dist, classe, conf, ide_doc))

                    if len(batch) >= 10000:
                        cur.executemany(
                            """
                            UPDATE documento
                            SET cluster_id = %s, dist_centroide = %s, classe_predita = %s, confianca_classe = %s
                            WHERE ide_documento = %s;
                            """,
                            batch,
                        )
                        conn.commit()
                        batch.clear()

                if batch:
                    cur.executemany(
                        """
                        UPDATE documento
                        SET cluster_id = %s, dist_centroide = %s, classe_predita = %s, confianca_classe = %s
                        WHERE ide_documento = %s;
                        """,
                        batch,
                    )
                    conn.commit()

        print("[OK] PostgreSQL atualizado com sucesso!")
    except Exception as e:
        print(f"[Aviso] Falha ao atualizar PostgreSQL: {e}")


def main():
    parser = argparse.ArgumentParser(description="Executa FAISS K-Means com K=10 no CEAP completo")
    parser.add_argument("--npz", type=Path, default=Path("data/latent_vectors_full.npz"), help="Arquivo NPZ com embeddings")
    parser.add_argument("--metadata", type=Path, default=Path("data/latent_vectors_full_metadata.json"), help="Arquivo JSON de metadados")
    parser.add_argument("--pdf-dir", type=Path, default=Path("/media/ssd2/datasets/CEAP"), help="Diretório dos PDFs")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"), help="Diretório dos arquivos brutos Ano-*.json")
    parser.add_argument("--output-base", "-o", type=Path, default=Path("data"), help="Pasta base de saída")
    parser.add_argument("--k", type=int, default=10, help="Número de clusters (padrão: 10)")
    parser.add_argument("--niter", type=int, default=25, help="Iterações de K-Means (padrão: 25)")
    parser.add_argument("--seed", type=int, default=42, help="Seed para reproducibilidade")
    parser.add_argument("--samples", "-n", type=int, default=20, help="Amostras exportadas por cluster (padrão: 20)")
    parser.add_argument("--max-docs", type=int, default=None, help="Limite de documentos para processar (amostragem rápida)")
    parser.add_argument("--skip-export", action="store_true", help="Pular exportação de imagens PNG")
    parser.add_argument("--update-db", action="store_true", help="Atualizar banco PostgreSQL")
    args = parser.parse_args()

    print("======================================================================")
    print(f"FAISS K-Means Pipeline &bull; K={args.k} Classes no Dataset CEAP")
    print("======================================================================")

    # 1. Carregar embeddings
    print(f"Carregando embeddings de {args.npz}...")
    npz_data = np.load(args.npz)
    vlm_embs = npz_data["vlm_doc_embeddings"]
    dino_embs = npz_data["dino_doc_embeddings"]
    total_docs = len(vlm_embs)
    print(f"Embeddings carregados: {total_docs:,} documentos")
    print(f"  ColPali VLM: {vlm_embs.shape}")
    print(f"  DINOv2:      {dino_embs.shape}")

    # 2. Carregar lista de documentos
    print(f"Carregando metadados de {args.metadata}...")
    with open(args.metadata, "r", encoding="utf-8") as f:
        meta_json = json.load(f)
    documents = meta_json.get("documents", [])
    print(f"Documentos indexados: {len(documents):,}")

    # Aplicar corte de amostragem rápida se solicitado
    if args.max_docs and args.max_docs < total_docs:
        print(f"[Amostragem Rápida] Limitando execução aos primeiros {args.max_docs:,} documentos...")
        vlm_embs = vlm_embs[:args.max_docs]
        dino_embs = dino_embs[:args.max_docs]
        documents = documents[:args.max_docs]
        total_docs = len(vlm_embs)

    # 3. Carregar / construir lookup de metadados (subcota, fornecedor, ano)
    cache_lookup = args.output_base / "pdf_metadata_lookup.json"
    meta_lookup = get_or_create_metadata_lookup(args.raw_dir, cache_lookup)

    # 4. Executar FAISS K-Means para ColPali VLM
    print("\n-------------------------------------------------------------")
    print(f"[1/2] Executando FAISS K-Means (K={args.k}) em ColPali VLM (128-d)...")
    print("-------------------------------------------------------------")
    vlm_labels, vlm_sims, vlm_centroids = run_faiss_spherical_kmeans(
        vlm_embs, k=args.k, niter=args.niter, seed=args.seed
    )
    vlm_analysis = analyze_clusters(vlm_labels, vlm_sims, documents, meta_lookup, k=args.k)

    # 5. Executar FAISS K-Means para DINOv2
    print("\n-------------------------------------------------------------")
    print(f"[2/2] Executando FAISS K-Means (K={args.k}) em DINOv2 (384-d)...")
    print("-------------------------------------------------------------")
    dino_labels, dino_sims, dino_centroids = run_faiss_spherical_kmeans(
        dino_embs, k=args.k, niter=args.niter, seed=args.seed
    )
    dino_analysis = analyze_clusters(dino_labels, dino_sims, documents, meta_lookup, k=args.k)

    # 6. Salvar NPZ com os novos clusters K=10
    output_npz = args.output_base / f"latent_vectors_faiss_k{args.k}.npz"
    print(f"\nSalvando clusters e centroides em: {output_npz}...")
    np.savez_compressed(
        output_npz,
        clusters_doc_vlm=vlm_labels,
        similarities_doc_vlm=vlm_sims,
        centroids_doc_vlm=vlm_centroids,
        clusters_doc_dino=dino_labels,
        similarities_doc_dino=dino_sims,
        centroids_doc_dino=dino_centroids,
    )
    print(f"[OK] NPZ salvo ({output_npz.stat().st_size / (1024*1024):.1f} MB)")

    # 7. Imprimir resumo das classes identificadas no terminal
    print("\n======================================================================")
    print(f"RESUMO DAS {args.k} CLASSES IDENTIFICADAS &bull; COLPALI VLM")
    print("======================================================================")
    for c_id in range(args.k):
        a = vlm_analysis[c_id]
        print(f"Cluster {c_id:02d}: [{a['inferred_taxonomy']:<28}] {a['total_docs']:>8,} docs ({a['percentage']:>5.1f}%) | Subcota: {a['dominant_subcota'][:35]} ({a['dominant_subcota_pct']:.0f}%)")

    print("\n======================================================================")
    print(f"RESUMO DAS {args.k} CLASSES IDENTIFICADAS &bull; DINOv2")
    print("======================================================================")
    for c_id in range(args.k):
        a = dino_analysis[c_id]
        print(f"Cluster {c_id:02d}: [{a['inferred_taxonomy']:<28}] {a['total_docs']:>8,} docs ({a['percentage']:>5.1f}%) | Subcota: {a['dominant_subcota'][:35]} ({a['dominant_subcota_pct']:.0f}%)")

    # 8. Exportar amostras e gerar galerias
    vlm_exported = {}
    dino_exported = {}

    if not args.skip_export:
        # ColPali VLM
        vlm_dir = args.output_base / f"cluster-faiss-vlm-k{args.k}"
        vlm_exported = export_samples_for_technique(
            tech_name="vlm",
            tech_display="ColPali VLM",
            labels=vlm_labels,
            similarities=vlm_sims,
            documents=documents,
            meta_lookup=meta_lookup,
            pdf_dir=args.pdf_dir,
            output_dir=vlm_dir,
            k=args.k,
            samples_per_cluster=args.samples,
        )
        # Salvar summary e galeria HTML VLM
        with open(vlm_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump({
                "analysis": vlm_analysis,
                "samples": vlm_exported,
            }, f, indent=2, ensure_ascii=False)
        generate_html_gallery_k10(
            tech_name="vlm",
            tech_display="ColPali VLM (Visão Multimodal)",
            k=args.k,
            cluster_analysis=vlm_analysis,
            exported_data=vlm_exported,
            output_html=vlm_dir / "index.html",
        )
        # Alias/symlink
        symlink_vlm = args.output_base / f"cluster-vlm-k{args.k}"
        if not symlink_vlm.exists():
            try:
                os.symlink(vlm_dir.name, symlink_vlm)
            except Exception:
                pass

        # DINOv2
        dino_dir = args.output_base / f"cluster-faiss-dino-k{args.k}"
        dino_exported = export_samples_for_technique(
            tech_name="dino",
            tech_display="DINOv2",
            labels=dino_labels,
            similarities=dino_sims,
            documents=documents,
            meta_lookup=meta_lookup,
            pdf_dir=args.pdf_dir,
            output_dir=dino_dir,
            k=args.k,
            samples_per_cluster=args.samples,
        )
        # Salvar summary e galeria HTML DINO
        with open(dino_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump({
                "analysis": dino_analysis,
                "samples": dino_exported,
            }, f, indent=2, ensure_ascii=False)
        generate_html_gallery_k10(
            tech_name="dino",
            tech_display="DINOv2 (Estrutura Visual)",
            k=args.k,
            cluster_analysis=dino_analysis,
            exported_data=dino_exported,
            output_html=dino_dir / "index.html",
        )
        # Alias/symlink
        symlink_dino = args.output_base / f"cluster-dino-k{args.k}"
        if not symlink_dino.exists():
            try:
                os.symlink(dino_dir.name, symlink_dino)
            except Exception:
                pass

        # Gerar Dashboard Comparativo K=10
        comparison_html = args.output_base / f"clusters_k{args.k}_comparison.html"
        generate_clusters_comparison_dashboard(
            k=args.k,
            vlm_analysis=vlm_analysis,
            dino_analysis=dino_analysis,
            output_file=comparison_html,
        )

    # 9. Opcional: Atualizar PostgreSQL
    if args.update_db:
        update_database_clusters(documents, vlm_labels, vlm_sims, vlm_analysis)

    print("\n======================================================================")
    print(f"[CONCLUÍDO COM SUCESSO] FAISS K-Means K={args.k} finalizado!")
    print(f"Galerias geradas:")
    print(f"  ColPali VLM: {args.output_base / f'cluster-faiss-vlm-k{args.k}' / 'index.html'}")
    print(f"  DINOv2:      {args.output_base / f'cluster-faiss-dino-k{args.k}' / 'index.html'}")
    print(f"  Comparativo: {args.output_base / f'clusters_k{args.k}_comparison.html'}")
    print("======================================================================")


if __name__ == "__main__":
    main()
