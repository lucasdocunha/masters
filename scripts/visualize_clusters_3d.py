#!/usr/bin/env python3
"""
Geração de visualização interativa 3D dos clusters de embeddings (DINO e ColPali VLM).
Reduz os vetores latentes para 3 dimensões (via UMAP ou PCA) e gera um arquivo HTML
independente e navegável em 3D (rotação 360°, zoom, translação e tooltip informativo).
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA


def reduce_to_3d(
    embeddings: np.ndarray,
    method: str = "umap",
    random_state: int = 42,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> np.ndarray:
    """Reduz os embeddings de alta dimensão para 3 componentes (X, Y, Z)."""
    if embeddings.shape[0] < 4:
        # Fallback para poucos pontos
        pca = PCA(n_components=min(3, embeddings.shape[0]), random_state=random_state)
        res = pca.fit_transform(embeddings)
        if res.shape[1] < 3:
            pad = np.zeros((res.shape[0], 3 - res.shape[1]), dtype=np.float32)
            res = np.hstack([res, pad])
        return res

    if method.lower() == "umap":
        try:
            import umap

            reducer = umap.UMAP(
                n_components=3,
                n_neighbors=min(n_neighbors, embeddings.shape[0] - 1),
                min_dist=min_dist,
                metric="cosine",
                random_state=random_state,
                n_jobs=-1,
            )
            return reducer.fit_transform(embeddings)
        except Exception as e:
            print(f"[Aviso] Falha ao executar UMAP ({e}). Utilizando PCA 3D...")

    # PCA 3D
    pca = PCA(n_components=3, random_state=random_state)
    return pca.fit_transform(embeddings)


def extract_year_from_name(name: str) -> str:
    """Tenta inferir o ano a partir do nome ou retorna 'N/A'."""
    match = re.search(r"(201[4-9]|202[0-6])", name)
    return match.group(1) if match else "N/A"


def get_pdf_to_year_map(raw_dir: Optional[Path] = None) -> dict[str, int]:
    """Retorna dicionário {nome_arquivo_pdf: ano} usando cache se disponível."""
    if raw_dir is None:
        raw_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    cache_file = raw_dir.parent / "pdf_to_year.json"
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    mapping = {}
    if raw_dir.exists():
        for ano_file in sorted(raw_dir.glob("Ano-*.json")):
            try:
                with open(ano_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                items = data.get("dados", []) if isinstance(data, dict) else data
                fallback_year = int(ano_file.stem.split("-")[-1]) if ano_file.stem.split("-")[-1].isdigit() else 0
                for it in items:
                    url = it.get("urlDocumento", "")
                    if url and url.lower().endswith(".pdf"):
                        fname = url.split("/")[-1]
                        if fname not in mapping:
                            mapping[fname] = int(it.get("ano", fallback_year))
            except Exception:
                pass
        if mapping:
            try:
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(mapping, f)
            except Exception:
                pass
    return mapping


YEAR_PALETTE = {
    "2014": "#1f77b4",  # Azul
    "2015": "#00bcd4",  # Ciano
    "2016": "#009688",  # Teal
    "2017": "#4caf50",  # Verde
    "2018": "#8bc34a",  # Verde claro
    "2019": "#ffeb3b",  # Amarelo
    "2020": "#ff9800",  # Laranja
    "2021": "#ff5722",  # Laranja escuro / coral
    "2022": "#e91e63",  # Rosa
    "2023": "#9c27b0",  # Roxo
    "2024": "#673ab7",  # Roxo escuro / índigo
    "2025": "#3f51b5",  # Azul royal
    "2026": "#795548",  # Marrom
    "N/A":  "#9e9e9e",  # Cinza
}


def build_3d_figure(
    coords_dino: np.ndarray,
    labels_dino: np.ndarray,
    coords_vlm: Optional[np.ndarray],
    labels_vlm: Optional[np.ndarray],
    meta_items: list[dict[str, Any]],
    title_suffix: str = "Documentos",
    color_by: str = "year",
) -> go.Figure:
    """Constrói a figura 3D interativa do Plotly com suporte a alternância DINO vs VLM e coloração por ano ou cluster."""
    fig = go.Figure()

    doc_names = [m.get("doc_name", f"item_{i}") for i, m in enumerate(meta_items)]
    num_pages = [m.get("num_pages", 1) for m in meta_items]
    years = [str(m.get("year", extract_year_from_name(name))) for name, m in zip(doc_names, meta_items)]

    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5"
    ]

    if color_by == "year":
        unique_groups = sorted(set(years), key=lambda y: (y == "N/A", str(y)))
        group_colors = {
            g: YEAR_PALETTE.get(g, colors[i % len(colors)])
            for i, g in enumerate(unique_groups)
        }
        group_title = "Ano"
    else:
        unique_groups = sorted(set(labels_dino))
        group_colors = {
            g: colors[g % len(colors)] for g in unique_groups
        }
        group_title = "Cluster"

    # 1. Traces do DINO
    dino_trace_indices = []
    for g in unique_groups:
        if color_by == "year":
            mask = np.array([y == g for y in years])
            trace_name = f"{g} ({np.sum(mask):,})"
        else:
            mask = (labels_dino == g)
            trace_name = f"DINO: Cluster {g} ({np.sum(mask):,})"

        if not np.any(mask):
            continue

        pts = coords_dino[mask]
        c_names = [doc_names[i] for i, m in enumerate(mask) if m]
        c_pages = [num_pages[i] for i, m in enumerate(mask) if m]
        c_years = [years[i] for i, m in enumerate(mask) if m]
        c_clusters = [labels_dino[i] for i, m in enumerate(mask) if m]

        customdata = np.stack((c_names, c_pages, c_years, [f"Cluster {c}" for c in c_clusters]), axis=-1)
        color = group_colors[g]

        trace = go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            mode="markers",
            name=trace_name,
            customdata=customdata,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Ano: %{customdata[2]}<br>"
                "Cluster: %{customdata[3]}<br>"
                "Páginas: %{customdata[1]}<br>"
                "X: %{x:.2f} | Y: %{y:.2f} | Z: %{z:.2f}<extra></extra>"
            ),
            marker=dict(
                size=3,
                color=color,
                opacity=0.8,
            ),
            visible=True,
        )
        fig.add_trace(trace)
        dino_trace_indices.append(len(fig.data) - 1)

    # 2. Traces do ColPali VLM
    vlm_trace_indices = []
    if coords_vlm is not None:
        if color_by != "year" and labels_vlm is not None:
            vlm_groups = sorted(set(labels_vlm))
        else:
            vlm_groups = unique_groups

        for g in vlm_groups:
            if color_by == "year":
                mask = np.array([y == g for y in years])
                trace_name = f"{g} ({np.sum(mask):,})"
                color = group_colors[g]
            else:
                mask = (labels_vlm == g)
                trace_name = f"ColPali: Cluster {g} ({np.sum(mask):,})"
                color = colors[(g + 5) % len(colors)]

            if not np.any(mask):
                continue

            pts = coords_vlm[mask]
            c_names = [doc_names[i] for i, m in enumerate(mask) if m]
            c_pages = [num_pages[i] for i, m in enumerate(mask) if m]
            c_years = [years[i] for i, m in enumerate(mask) if m]
            c_clusters = [labels_vlm[i] if labels_vlm is not None else 0 for i, m in enumerate(mask) if m]

            customdata = np.stack((c_names, c_pages, c_years, [f"Cluster {c}" for c in c_clusters]), axis=-1)

            trace = go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="markers",
                name=trace_name,
                customdata=customdata,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Ano: %{customdata[2]}<br>"
                    "Cluster: %{customdata[3]}<br>"
                    "Páginas: %{customdata[1]}<br>"
                    "X: %{x:.2f} | Y: %{y:.2f} | Z: %{z:.2f}<extra></extra>"
                ),
                marker=dict(
                    size=3,
                    color=color,
                    opacity=0.8,
                ),
                visible=False,
            )
            fig.add_trace(trace)
            vlm_trace_indices.append(len(fig.data) - 1)

    # Botões interativos para alternar entre DINO e ColPali
    updatemenus = []
    if vlm_trace_indices:
        total_traces = len(fig.data)
        dino_visibility = [True if i in dino_trace_indices else False for i in range(total_traces)]
        vlm_visibility = [True if i in vlm_trace_indices else False for i in range(total_traces)]

        updatemenus = [
            dict(
                type="buttons",
                direction="left",
                x=0.0,
                y=1.05,
                xanchor="left",
                yanchor="bottom",
                bgcolor="#2a2e39",
                bordercolor="#555",
                font=dict(color="#ffffff", size=12),
                buttons=[
                    dict(
                        label="DINOv2 (Visual)",
                        method="update",
                        args=[
                            {"visible": dino_visibility},
                            {"title": f"Espaço Latente 3D - DINOv2 ({title_suffix})"},
                        ],
                    ),
                    dict(
                        label="ColPali VLM (Multimodal)",
                        method="update",
                        args=[
                            {"visible": vlm_visibility},
                            {"title": f"Espaço Latente 3D - ColPali VLM ({title_suffix})"},
                        ],
                    ),
                ],
            )
        ]

    fig.update_layout(
        title=dict(
            text=f"Navegação 3D dos Documentos CEAP ({title_suffix})",
            font=dict(size=18, color="#ffffff"),
            x=0.5,
            xanchor="center",
        ),
        template="plotly_dark",
        scene=dict(
            xaxis=dict(title="Dimensão 1", backgroundcolor="#111", gridcolor="#333", showbackground=True),
            yaxis=dict(title="Dimensão 2", backgroundcolor="#111", gridcolor="#333", showbackground=True),
            zaxis=dict(title="Dimensão 3", backgroundcolor="#111", gridcolor="#333", showbackground=True),
            aspectmode="cube",
        ),
        legend=dict(
            title=dict(text=group_title, font=dict(color="#ffffff")),
            itemsizing="constant",
            bgcolor="rgba(20,20,20,0.7)",
            bordercolor="#444",
            borderwidth=1,
        ),
        margin=dict(l=10, r=10, b=10, t=60),
        updatemenus=updatemenus,
    )

    return fig


def generate_3d_visualization(
    npz_path: Path,
    meta_path: Optional[Path] = None,
    output_html: Path = Path("data/clusters_3d.html"),
    method: str = "umap",
    level: str = "doc",
    subsample: int = 25000,
    seed: int = 42,
    color_by: str = "year",
) -> Path:
    """Carrega dados, reduz para 3D e salva arquivo HTML interativo."""
    if not npz_path.exists():
        raise FileNotFoundError(f"Arquivo NPZ não encontrado: {npz_path}")

    print(f"Carregando embeddings de {npz_path}...")
    data = np.load(npz_path)

    # Carregar metadados
    metadata = {}
    if meta_path and meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    # Selecionar nível de agregação: documento ou página
    if level == "doc":
        dino_embs = data.get("dino_doc_embeddings")
        vlm_embs = data.get("vlm_doc_embeddings")
        labels_dino = data.get("clusters_doc_dino")
        labels_vlm = data.get("clusters_doc_vlm")
        items_meta = metadata.get("documents", [])
        title_suffix = "Nível Documento"
    else:
        dino_embs = data.get("dino_page_embeddings")
        vlm_embs = data.get("vlm_page_embeddings")
        labels_dino = data.get("clusters_page_dino")
        labels_vlm = data.get("clusters_page_vlm")
        items_meta = metadata.get("pages", [])
        title_suffix = "Nível Página"

    if dino_embs is None:
        raise ValueError("dino_embeddings não encontrados no arquivo NPZ")

    n_items = len(dino_embs)
    print(f"Total de itens ({level}): {n_items}")

    # Subamostragem para fluidez no navegador caso n_items seja muito grande
    if subsample > 0 and n_items > subsample:
        print(f"Subamostrando {subsample}/{n_items} pontos para visualização 3D fluida a 60 FPS...")
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(n_items, size=subsample, replace=False))
        dino_embs = dino_embs[indices]
        labels_dino = labels_dino[indices] if labels_dino is not None else None
        if vlm_embs is not None:
            vlm_embs = vlm_embs[indices]
        if labels_vlm is not None:
            labels_vlm = labels_vlm[indices]
        if items_meta and len(items_meta) == n_items:
            items_meta = [items_meta[i] for i in indices]
    else:
        indices = np.arange(n_items)

    # Preencher metadados padrão se não existirem
    if not items_meta or len(items_meta) != len(dino_embs):
        items_meta = [{"doc_name": f"item_{i}", "num_pages": 1} for i in range(len(dino_embs))]

    # Garantir que o campo 'year' esteja preenchido
    pdf_to_year = None
    for m in items_meta:
        if "year" not in m:
            if pdf_to_year is None:
                pdf_to_year = get_pdf_to_year_map()
            name = m.get("doc_name", "")
            m["year"] = pdf_to_year.get(name, extract_year_from_name(name))

    if labels_dino is None:
        labels_dino = np.zeros(len(dino_embs), dtype=int)
    if labels_vlm is None and vlm_embs is not None:
        labels_vlm = np.zeros(len(vlm_embs), dtype=int)

    # Redução 3D DINO
    print(f"Calculando projeção 3D ({method.upper()}) para DINOv2 ({len(dino_embs)} itens)...")
    coords_dino = reduce_to_3d(dino_embs, method=method, random_state=seed)

    # Redução 3D VLM
    coords_vlm = None
    if vlm_embs is not None:
        print(f"Calculando projeção 3D ({method.upper()}) para ColPali VLM ({len(vlm_embs)} itens)...")
        coords_vlm = reduce_to_3d(vlm_embs, method=method, random_state=seed)

    # Construção do gráfico 3D
    print("Construindo cena 3D interativa no Plotly...")
    fig = build_3d_figure(
        coords_dino=coords_dino,
        labels_dino=labels_dino,
        coords_vlm=coords_vlm,
        labels_vlm=labels_vlm,
        meta_items=items_meta,
        title_suffix=title_suffix,
        color_by=color_by,
    )

    output_html.parent.mkdir(parents=True, exist_ok=True)
    print(f"Salvando visualizador 3D em {output_html}...")
    fig.write_html(
        str(output_html),
        include_plotlyjs=True,  # 100% autossuficiente (funciona offline sem CDN)
        full_html=True,
    )

    print(f"[Sucesso] Arquivo 3D gerado: {output_html} ({output_html.stat().st_size / 1024 / 1024:.2f} MB)")
    return output_html


def main():
    parser = argparse.ArgumentParser(description="Renderização interativa 3D dos clusters de embeddings (CEAP)")
    parser.add_argument("--npz", type=Path, default=Path("data/latent_vectors_sample_5pct.npz"), help="Caminho do arquivo .npz com embeddings")
    parser.add_argument("--metadata", type=Path, default=Path("data/latent_vectors_sample_5pct_metadata.json"), help="Caminho dos metadados .json")
    parser.add_argument("--output-html", "-o", type=Path, default=Path("data/clusters_3d.html"), help="Caminho do HTML gerado")
    parser.add_argument("--method", choices=["umap", "pca"], default="umap", help="Método de redução 3D (umap ou pca)")
    parser.add_argument("--level", choices=["doc", "page"], default="doc", help="Nível da visualização (doc ou page)")
    parser.add_argument("--color-by", choices=["year", "cluster"], default="year", help="Coloração dos pontos (por ano ou por cluster)")
    parser.add_argument("--subsample", type=int, default=25000, help="Máximo de pontos para navegação 3D fluida (0 para todos)")
    parser.add_argument("--seed", type=int, default=42, help="Seed para reprodutibilidade")
    args = parser.parse_args()

    generate_3d_visualization(
        npz_path=args.npz,
        meta_path=args.metadata,
        output_html=args.output_html,
        method=args.method,
        level=args.level,
        color_by=args.color_by,
        subsample=args.subsample,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
