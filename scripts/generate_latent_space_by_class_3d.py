#!/usr/bin/env python3
"""
Gera visualização 3D interativa do Espaço Latente Multimodal (512-D: DINOv2 + ColPali)
colorido pelas classes conhecidas do modelo, com documentos não mapeados em cinza (#888888).
Salva em data/latent_vectors_sample_5pct_3d.html e Desktop/espaco_latente_3d.html.
"""

import json
import re
import random
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
import umap

BASE_DIR = Path("/home/lucas.ocunha/masters")
DATA_DIR = BASE_DIR / "data"
NPZ_PATH = DATA_DIR / "latent_vectors_sample_5pct.npz"
META_PATH = DATA_DIR / "latent_vectors_sample_5pct_metadata.json"
HISTORY_PATH = BASE_DIR / "ml_backend" / "models" / "annotations_history.json"
OUTPUT_HTML = DATA_DIR / "latent_vectors_sample_5pct_3d.html"
DESKTOP_HTML = Path("/home/lucas.ocunha/Desktop/espaco_latente_3d.html")

CLASS_COLORS = {
    "nfe": "#2563eb",                  # Azul royal
    "nfse": "#16a34a",                 # Verde
    "nfc": "#ea580c",                  # Laranja
    "recibo": "#9333ea",               # Roxo vibrante
    "uber": "#eab308",                 # Dourado / Amarelo
    "energia": "#06b6d4",              # Ciano
    "agua": "#38bdf8",                 # Azul claro / Celeste
    "telefonia": "#dc2626",            # Vermelho
    "passagem_aerea": "#db2777",       # Rosa escuro / Magenta
    "passagem_rodoviaria": "#059669",  # Esmeralda
    "pedagio": "#b45309",              # Âmbar escuro / Cobre
    "iptu": "#6366f1",                 # Índigo
    "fatura": "#0d9488",               # Teal escuro
    "emails": "#65a30d",               # Lima
    "comprovante_termico_pedagio": "#f97316", # Laranja brilhante
    "Não Mapeado": "#52525b",          # Cinza neutro suave (zinc-600)
}


def main():
    print("1. Carregando embeddings do sample 5% (70.109 docs)...")
    npz = np.load(NPZ_PATH)
    dino = npz["dino_doc_embeddings"]
    vlm = npz["vlm_doc_embeddings"]

    # Concatenação: 512 dimensões
    combined = np.hstack([dino, vlm]).astype(np.float32)
    norms = np.linalg.norm(combined, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    combined /= norms
    print(f"   Embeddings combinados: {combined.shape} (DINOv2 {dino.shape[1]}D + ColPali {vlm.shape[1]}D)")

    print("2. Carregando metadados e histórico de classes...")
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    docs = meta.get("documents", [])

    with open(HISTORY_PATH, "r", encoding="utf-8") as f:
        history = json.load(f)
    print(f"   Total de anotações no histórico: {len(history):,}")

    # Mapear cada documento para sua classe
    labels = []
    mapped_count = 0
    mapped_indices = []
    unmapped_indices = []

    for idx, doc in enumerate(docs):
        fname = doc.get("doc_name", "")
        stem = Path(fname).stem
        cls = history.get(stem)
        if cls:
            labels.append(cls)
            mapped_count += 1
            mapped_indices.append(idx)
        else:
            labels.append("Não Mapeado")
            unmapped_indices.append(idx)

    print(f"   Documentos mapeados no sample: {mapped_count:,} ({mapped_count/len(docs)*100:.1f}%)")
    print(f"   Documentos não mapeados: {len(unmapped_indices):,} ({len(unmapped_indices)/len(docs)*100:.1f}%)")

    # Amostragem inteligente para projeção fluida em 3D:
    # 100% dos documentos mapeados + 20.000 não mapeados em cinza
    random.seed(42)
    k_unmapped = min(20000, len(unmapped_indices))
    sampled_unmapped = random.sample(unmapped_indices, k_unmapped)
    selected_indices = sorted(mapped_indices + sampled_unmapped)

    sub_embs = combined[selected_indices]
    sub_docs = [docs[i] for i in selected_indices]
    sub_labels = [labels[i] for i in selected_indices]

    print(f"\n3. Calculando projeção 3D (UMAP) para {len(selected_indices):,} documentos...")
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
        n_jobs=-1,
    )
    coords_3d = reducer.fit_transform(sub_embs)
    print("   Projeção 3D calculada com sucesso!")

    print("\n4. Construindo figura 3D no Plotly...")
    fig = go.Figure()

    # Organizar traces: primeiro o "Não Mapeado" (para ficar ao fundo), depois as classes ativas
    classes_order = sorted(list(set(sub_labels)))
    if "Não Mapeado" in classes_order:
        classes_order.remove("Não Mapeado")
        classes_order = ["Não Mapeado"] + classes_order

    for cls in classes_order:
        mask = [l == cls for l in sub_labels]
        count = sum(mask)
        if count == 0:
            continue

        c_indices = [i for i, m in enumerate(mask) if m]
        pts = coords_3d[c_indices]
        names = [sub_docs[i].get("doc_name", "") for i in c_indices]
        pages = [sub_docs[i].get("num_pages", 1) for i in c_indices]
        years = []
        for name in names:
            m = re.search(r"(201[4-9]|202[0-6])", name)
            years.append(m.group(1) if m else "N/A")

        color = CLASS_COLORS.get(cls, "#38bdf8")

        if cls == "Não Mapeado":
            trace_name = f"Não Mapeado ({count:,})"
            marker_dict = dict(
                size=2.0,
                color=color,
                opacity=0.22,
            )
        else:
            trace_name = f"{cls.upper()} ({count:,})"
            marker_dict = dict(
                size=4.5,
                color=color,
                opacity=0.90,
                line=dict(color="#ffffff", width=0.5),
            )

        customdata = np.stack((names, pages, years, [cls] * count), axis=-1)

        trace = go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            mode="markers",
            name=trace_name,
            customdata=customdata,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "<b>Classe:</b> %{customdata[3]}<br>"
                "<b>Ano:</b> %{customdata[2]}<br>"
                "<b>Páginas:</b> %{customdata[1]}<br>"
                "<extra></extra>"
            ),
            marker=marker_dict,
        )
        fig.add_trace(trace)

    # Layout escuro elegante com alta performance
    fig.update_layout(
        title=dict(
            text="<b>Espaço Latente 3D (DINOv2 + ColPali 512-D) &bull; Classes CEAP</b>",
            font=dict(size=20, color="#ffffff"),
            x=0.5,
            xanchor="center",
        ),
        paper_bgcolor="#090d16",
        plot_bgcolor="#090d16",
        scene=dict(
            xaxis=dict(
                title="UMAP Dim 1",
                backgroundcolor="#090d16",
                gridcolor="#1e293b",
                showbackground=True,
                zerolinecolor="#334155",
            ),
            yaxis=dict(
                title="UMAP Dim 2",
                backgroundcolor="#090d16",
                gridcolor="#1e293b",
                showbackground=True,
                zerolinecolor="#334155",
            ),
            zaxis=dict(
                title="UMAP Dim 3",
                backgroundcolor="#090d16",
                gridcolor="#1e293b",
                showbackground=True,
                zerolinecolor="#334155",
            ),
            aspectmode="cube",
        ),
        legend=dict(
            title=dict(text="<b>Classes / Status</b>", font=dict(color="#ffffff", size=13)),
            font=dict(color="#e2e8f0", size=11),
            bgcolor="rgba(15, 23, 42, 0.85)",
            bordercolor="#334155",
            borderwidth=1,
            itemsizing="constant",
        ),
        margin=dict(l=10, r=10, b=10, t=50),
    )

    print(f"\n5. Salvando HTML em:")
    print(f"   -> {OUTPUT_HTML}")
    fig.write_html(str(OUTPUT_HTML), include_plotlyjs=True, full_html=True)

    print(f"   -> {DESKTOP_HTML}")
    fig.write_html(str(DESKTOP_HTML), include_plotlyjs=True, full_html=True)

    print("\n[SUCESSO] Visualização 3D do Espaço Latente por Classes gerada com sucesso!")


if __name__ == "__main__":
    main()
