#!/usr/bin/env python3
"""
Exporta amostras visuais dos clusters de embeddings (DINOv2 e ColPali VLM).
Calcula os centroides de cada cluster, seleciona os N documentos mais representativos
(mais próximos do centroide), renderiza a primeira página em PNG e cria atalhos para os PDFs.

Estrutura de saída:
  data/cluster-{technique}/{cluster_id}/
    ├── 01_{doc_stem}.png
    ├── 01_{doc_stem}.pdf -> symlink para o PDF original
    └── ... (20 amostras)
  data/cluster-{technique}/summary.json
  data/cluster-{technique}/index.html (galeria interativa)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pymupdf


def render_pdf_first_page(pdf_path: Path, output_png: Path, dpi: int = 150) -> bool:
    """Renderiza a 1ª página de um PDF em alta resolução PNG."""
    try:
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


def generate_html_gallery(technique_name: str, tech_display: str, summary: dict, output_html: Path) -> None:
    """Gera visualizador HTML interativo para navegar pelos clusters e documentos."""
    cluster_sections_html = []

    for c_id_str, data in summary.items():
        total_docs = data["total_docs_in_cluster"]
        pct = data["cluster_percentage"]
        samples = data["exported_samples"]

        cards_html = []
        for s in samples:
            rank = s["rank"]
            doc_name = s["doc_name"]
            img_rel = f"{c_id_str}/{s['image']}"
            pdf_rel = f"{c_id_str}/{s['pdf']}"
            dist = s["dist_to_centroid"]
            pages = s["num_pages"]
            year = s["year"]

            card = f"""
            <div class="card">
              <a href="{img_rel}" target="_blank" title="Clique para ampliar imagem">
                <img src="{img_rel}" alt="{doc_name}" loading="lazy">
              </a>
              <div class="card-info">
                <div class="doc-rank">#{rank:02d} &bull; {year}</div>
                <div class="doc-name" title="{doc_name}">{doc_name}</div>
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
        <div class="cluster-section" id="cluster-{c_id_str}">
          <div class="cluster-header">
            <div>
              <span class="cluster-title">Cluster {c_id_str}</span>
              <span class="cluster-badge">{total_docs:,} documentos ({pct:.1f}%)</span>
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
        cluster_sections_html.append(section)

    all_sections = "\n".join(cluster_sections_html)

    html_content = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Galeria dos Clusters - {tech_display}</title>
<style>
  :root {{
    --bg-main: #0b0f19;
    --bg-card: #151d30;
    --border-color: #263352;
    --primary: #38bdf8;
    --primary-hover: #0ea5e9;
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
    max-width: 1400px;
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
    gap: 12px;
    flex-wrap: wrap;
    align-items: center;
  }}
  .nav-btn {{
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    color: var(--text-main);
    padding: 6px 14px;
    border-radius: 6px;
    text-decoration: none;
    font-size: 13px;
    font-weight: 500;
    transition: all 0.2s;
  }}
  .nav-btn:hover {{
    background: var(--primary);
    color: #000;
    border-color: var(--primary);
  }}
  .container {{
    max-width: 1400px;
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
    align-items: center;
    border-bottom: 1px solid var(--border-color);
    padding-bottom: 16px;
    margin-bottom: 20px;
  }}
  .cluster-title {{
    font-size: 20px;
    font-weight: 700;
    color: #ffffff;
    margin-right: 12px;
  }}
  .cluster-badge {{
    background: #1e3a8a;
    color: #93c5fd;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 12px;
    font-weight: 600;
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
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
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
    height: 280px;
    object-fit: contain;
    background: #ffffff;
    display: block;
    border-bottom: 1px solid var(--border-color);
  }}
  .card-info {{
    padding: 12px;
    display: flex;
    flex-direction: column;
    gap: 6px;
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
  .meta-row {{
    display: flex;
    justify-content: space-between;
    color: var(--text-muted);
    font-size: 11px;
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
  <h1>Galeria Visual dos Clusters &bull; {tech_display}</h1>
  <p class="desc">
    20 documentos mais representativos (mais próximos do centroide no espaço latente) para cada um dos 5 clusters identificados no dataset global do CEAP (1.339.678 documentos).
  </p>
  <div class="nav-bar">
    <span style="font-size: 13px; color: var(--text-muted); margin-right: 4px;">Ir para:</span>
    <a href="#cluster-0" class="nav-btn">Cluster 0</a>
    <a href="#cluster-1" class="nav-btn">Cluster 1</a>
    <a href="#cluster-2" class="nav-btn">Cluster 2</a>
    <a href="#cluster-3" class="nav-btn">Cluster 3</a>
    <a href="#cluster-4" class="nav-btn">Cluster 4</a>
    <a href="../latent_vectors_full_3d.html" class="nav-btn" style="margin-left: auto; background: #4f46e5; border-color: #6366f1;">&larr; Ver Espaço 3D UMAP</a>
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
    print(f"  [HTML] Galeria interativa gerada: {output_html}")


def export_clusters(
    npz_path: Path,
    meta_path: Path,
    pdf_dir: Path,
    output_base: Path,
    samples_per_cluster: int = 20,
    techniques: list[str] = None,
) -> None:
    if techniques is None:
        techniques = ["dino", "colpali"]

    print(f"Carregando embeddings de {npz_path}...")
    npz_data = np.load(npz_path)

    print(f"Carregando metadados de {meta_path}...")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    documents = meta_data.get("documents", [])
    total_docs = len(documents)
    print(f"Total de documentos indexados: {total_docs:,}")

    # Carregar mapeamento de ano
    pdf_to_year = {}
    year_cache = output_base / "pdf_to_year.json"
    if year_cache.exists():
        with open(year_cache, "r", encoding="utf-8") as f:
            pdf_to_year = json.load(f)

    # Configuração das técnicas
    tech_configs = {
        "dino": {
            "name": "dino",
            "display": "DINOv2 (Estrutura Visual)",
            "embs_key": "dino_doc_embeddings",
            "cluster_key": "clusters_doc_dino",
            "dir_name": "cluster-dino",
        },
        "colpali": {
            "name": "colpali",
            "display": "ColPali VLM (Visão Multimodal)",
            "embs_key": "vlm_doc_embeddings",
            "cluster_key": "clusters_doc_vlm",
            "dir_name": "cluster-colpali",
        },
    }

    for tech in techniques:
        if tech not in tech_configs:
            print(f"[Aviso] Técnica desconhecida: {tech}")
            continue

        cfg = tech_configs[tech]
        embs = npz_data[cfg["embs_key"]]
        clusters = npz_data[cfg["cluster_key"]]
        unique_clusters = sorted(np.unique(clusters))
        k = len(unique_clusters)

        tech_dir = output_base / cfg["dir_name"]
        tech_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=======================================================")
        print(f"Exportando {samples_per_cluster} amostras por cluster para: {cfg['display']}")
        print(f"Destino: {tech_dir}")
        print(f"=======================================================")

        summary = {}

        for c_id in unique_clusters:
            mask = np.where(clusters == c_id)[0]
            cluster_count = len(mask)
            pct = (cluster_count / total_docs) * 100
            cluster_dir = tech_dir / str(c_id)
            cluster_dir.mkdir(parents=True, exist_ok=True)

            print(f"\nCluster {c_id}: {cluster_count:,} documentos ({pct:.1f}%)")

            # Cálculo do centroide do cluster
            cluster_embs = embs[mask]
            centroid = np.mean(cluster_embs, axis=0)
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm > 0:
                centroid /= centroid_norm

            # Distância cosseno para o centroide: 1 - dot(emb, centroid)
            dists = 1.0 - np.dot(cluster_embs, centroid)
            sorted_indices = np.argsort(dists)

            # Selecionar os N mais próximos que possuam PDF existente
            selected_samples = []
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

                # 2. Criar link simbólico para o PDF original
                try:
                    if out_pdf_symlink.is_symlink() or out_pdf_symlink.exists():
                        out_pdf_symlink.unlink()
                    os.symlink(os.path.abspath(pdf_path), out_pdf_symlink)
                except Exception as e:
                    print(f"    [Aviso] Falha ao criar symlink para PDF: {e}")

                year = doc_info.get("year", pdf_to_year.get(doc_name, "N/A"))
                dist_val = float(dists[local_idx])

                sample_meta = {
                    "rank": rank,
                    "doc_id": int(doc_info.get("doc_id", global_idx)),
                    "doc_name": doc_name,
                    "year": year,
                    "num_pages": doc_info.get("num_pages", 1),
                    "dist_to_centroid": dist_val,
                    "image": png_name,
                    "pdf": pdf_symlink_name,
                }
                selected_samples.append(sample_meta)
                print(f"  #{rank:02d}: {doc_name} (Ano: {year}, Dist: {dist_val:.3f}, {doc_info.get('num_pages', 1)} págs)")

                rank += 1
                if len(selected_samples) >= samples_per_cluster:
                    break

            summary[str(c_id)] = {
                "total_docs_in_cluster": cluster_count,
                "cluster_percentage": pct,
                "exported_samples": selected_samples,
            }

        # Salvar summary.json
        summary_path = tech_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n[OK] Resumo salvo em: {summary_path}")

        # Gerar galeria HTML
        gallery_path = tech_dir / "index.html"
        generate_html_gallery(cfg["name"], cfg["display"], summary, gallery_path)

        # Criar symlinks adicionais para conveniência
        if cfg["name"] == "dino":
            alt_dir = output_base / "cluster-dinov2"
            if not alt_dir.exists():
                try:
                    os.symlink(cfg["dir_name"], alt_dir)
                except Exception:
                    pass
        elif cfg["name"] == "colpali":
            alt_dir = output_base / "cluster-vlm"
            if not alt_dir.exists():
                try:
                    os.symlink(cfg["dir_name"], alt_dir)
                except Exception:
                    pass


def main():
    parser = argparse.ArgumentParser(description="Exporta 20 amostras de cada um dos 5 clusters para visualização")
    parser.add_argument("--npz", type=Path, default=Path("data/latent_vectors_full.npz"), help="Arquivo NPZ com embeddings e clusters")
    parser.add_argument("--metadata", type=Path, default=Path("data/latent_vectors_full_metadata.json"), help="Arquivo JSON de metadados")
    parser.add_argument("--pdf-dir", type=Path, default=Path("/media/ssd2/datasets/CEAP"), help="Diretório com os PDFs originais")
    parser.add_argument("--output-base", "-o", type=Path, default=Path("data"), help="Pasta base de saída (ex: data)")
    parser.add_argument("--samples", "-n", type=int, default=20, help="Quantidade de amostras por cluster (padrão: 20)")
    parser.add_argument("--techniques", nargs="+", default=["dino", "colpali"], help="Técnicas a exportar (dino, colpali)")
    args = parser.parse_args()

    export_clusters(
        npz_path=args.npz,
        meta_path=args.metadata,
        pdf_dir=args.pdf_dir,
        output_base=args.output_base,
        samples_per_cluster=args.samples,
        techniques=args.techniques,
    )


if __name__ == "__main__":
    main()
