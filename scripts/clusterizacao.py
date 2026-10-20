#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


def render_pdf_pages(pdf_path: Path) -> list[dict[str, Any]]:
    try:
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(pdf_path)
        return [
            {"page_idx": i, "image": page.render(scale=1.5).to_pil().convert("RGB"), "text": page.get_textpage().get_text_range().strip()}
            for i, page in enumerate(pdf)
        ]
    except ImportError:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        return [
            {"page_idx": i, "image": Image.frombytes("RGB", [p.get_pixmap().width, p.get_pixmap().height], p.get_pixmap().samples), "text": p.get_text().strip()}
            for i, p in enumerate(doc)
        ]


def load_documents(input_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    doc_meta, page_meta = [], []
    files = sorted(input_dir.glob("*.pdf")) + sorted(input_dir.glob("*.png")) + sorted(input_dir.glob("*.jpg"))

    for doc_id, file_path in enumerate(files):
        pages = render_pdf_pages(file_path) if file_path.suffix.lower() == ".pdf" else [{"page_idx": 0, "image": Image.open(file_path).convert("RGB"), "text": ""}]
        doc_meta.append({"doc_id": doc_id, "doc_name": file_path.name, "num_pages": len(pages)})
        for p in pages:
            page_meta.append({"doc_id": doc_id, "doc_name": file_path.name, "page_idx": p["page_idx"], "image": p["image"], "text": p["text"]})

    return doc_meta, page_meta


@torch.no_grad()
def extract_dino_embeddings(images: list[Image.Image], model_name: str, device: str) -> np.ndarray:
    print(f"Extraindo embeddings com DINO ({model_name})...")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    embeddings = []
    for i in range(0, len(images), 16):
        inputs = processor(images=images[i : i + 16], return_tensors="pt").to(device)
        out = torch.nn.functional.normalize(model(**inputs).last_hidden_state[:, 0, :], dim=-1)
        embeddings.append(out.cpu().float().numpy())
    return np.vstack(embeddings)


@torch.no_grad()
def extract_vlm_embeddings(images: list[Image.Image], texts: list[str], model_name: str, device: str) -> np.ndarray:
    print(f"Extraindo embeddings com VLM ({model_name})...")

    # 1. ColPali / ColQwen (VLM dedicada para documentos)
    if "colpali" in model_name.lower() or "colqwen" in model_name.lower():
        try:
            from colpali_engine.models import ColPali, ColPaliProcessor, ColQwen2, ColQwen2Processor
            cls_model, cls_proc = (ColQwen2, ColQwen2Processor) if "colqwen" in model_name.lower() else (ColPali, ColPaliProcessor)
            processor = cls_proc.from_pretrained(model_name)
            model = cls_model.from_pretrained(
                model_name,
                torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16,
            ).to(device).eval()

            embeddings = []
            batch_size = 4
            for i in range(0, len(images), batch_size):
                inputs = processor.process_images(images[i : i + batch_size]).to(device)
                patch_embs = model(**inputs)  # [B, N_patches, 128]
                doc_vecs = torch.nn.functional.normalize(patch_embs.mean(dim=1), dim=-1)
                embeddings.append(doc_vecs.cpu().float().numpy())
            return np.vstack(embeddings)
        except ImportError:
            print("Aviso: 'colpali-engine' não instalado. Para ColPali: pip install colpali-engine")
            print("Usando fallback para CLIP...")

    # 2. Qwen2-VL (VLM generativa via hidden-states)
    if "qwen" in model_name.lower():
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        processor = AutoProcessor.from_pretrained(model_name)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16,
        ).to(device).eval()
        embeddings = []
        for img in images:
            inputs = processor(images=[img], text=["Describe document:"], return_tensors="pt").to(device)
            out = model(**inputs, output_hidden_states=True)
            vec = torch.nn.functional.normalize(out.hidden_states[-1].mean(dim=1), dim=-1)
            embeddings.append(vec.cpu().float().numpy())
        return np.vstack(embeddings)

    # 3. CLIP / SigLIP (Fallback padrão contrastivo)
    from transformers import AutoTokenizer, CLIPModel
    processor = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()

    embeddings = []
    for i in range(0, len(images), 16):
        b_imgs, b_txts = images[i : i + 16], texts[i : i + 16]
        img_feats = torch.nn.functional.normalize(model.get_image_features(**processor(images=b_imgs, return_tensors="pt").to(device)), dim=-1)
        valid_idx = [idx for idx, t in enumerate(b_txts) if t.strip()]
        if valid_idx:
            txt_in = tokenizer([b_txts[idx][:300] for idx in valid_idx], padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
            txt_feats = torch.nn.functional.normalize(model.get_text_features(**txt_in), dim=-1)
            joint = img_feats.clone()
            joint[valid_idx] = torch.nn.functional.normalize(joint[valid_idx] + txt_feats, dim=-1)
            embeddings.append(joint.cpu().float().numpy())
        else:
            embeddings.append(img_feats.cpu().float().numpy())
    return np.vstack(embeddings)


def pool_document_embeddings(page_embs: np.ndarray, page_meta: list[dict[str, Any]], num_docs: int) -> np.ndarray:
    doc_embs = np.zeros((num_docs, page_embs.shape[1]), dtype=np.float32)
    doc_counts = np.zeros(num_docs, dtype=np.int32)
    for emb, meta in zip(page_embs, page_meta):
        doc_embs[meta["doc_id"]] += emb
        doc_counts[meta["doc_id"]] += 1
    doc_counts[doc_counts == 0] = 1
    doc_embs /= doc_counts[:, None]
    norms = np.linalg.norm(doc_embs, axis=-1, keepdims=True)
    return doc_embs / np.where(norms == 0, 1.0, norms)


def cluster(embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
    k = min(max(2, n_clusters), len(embeddings))
    try:
        from sklearn.cluster import KMeans
        return KMeans(n_clusters=k, random_state=42, n_init="auto").fit_predict(embeddings)
    except ImportError:
        from scipy.cluster.vq import kmeans2
        _, labels = kmeans2(embeddings.astype(float), k, minit="points", seed=42)
        return labels.astype(int)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extração de embeddings e clusterização por Documento e Página")
    parser.add_argument("--input-dir", "-i", type=Path, default=Path("data/raw/pdfs"), help="Pasta com PDFs ou imagens")
    parser.add_argument("--output-prefix", "-o", type=Path, default=Path("data/latent_vectors"), help="Prefixo dos arquivos salvos (.npz e .json)")
    parser.add_argument("--dino-model", default="facebook/dinov2-small", help="Modelo DINO (imagem)")
    parser.add_argument("--vlm-model", default="vidore/colpali-v1.2", help="VLM dedicada a documentos (ex: vidore/colpali-v1.2)")
    parser.add_argument("--k-doc", type=int, default=5, help="Clusters de documentos")
    parser.add_argument("--k-page", type=int, default=10, help="Clusters de páginas")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    doc_meta, page_meta = load_documents(args.input_dir)
    if not page_meta:
        print("Nenhum documento/página encontrado.")
        return

    images, texts = [p["image"] for p in page_meta], [p["text"] for p in page_meta]
    dino_page_embs = extract_dino_embeddings(images, args.dino_model, device)
    vlm_page_embs = extract_vlm_embeddings(images, texts, args.vlm_model, device)

    dino_doc_embs = pool_document_embeddings(dino_page_embs, page_meta, len(doc_meta))
    vlm_doc_embs = pool_document_embeddings(vlm_page_embs, page_meta, len(doc_meta))

    npz_path = args.output_prefix.with_suffix(".npz")
    np.savez_compressed(
        npz_path,
        dino_page_embeddings=dino_page_embs,
        vlm_page_embeddings=vlm_page_embs,
        dino_doc_embeddings=dino_doc_embs,
        vlm_doc_embeddings=vlm_doc_embs,
        clusters_doc_dino=cluster(dino_doc_embs, args.k_doc),
        clusters_doc_vlm=cluster(vlm_doc_embs, args.k_doc),
        clusters_page_dino=cluster(dino_page_embs, args.k_page),
        clusters_page_vlm=cluster(vlm_page_embs, args.k_page),
    )

    clean_pages = [{k: v for k, v in p.items() if k != "image"} for p in page_meta]
    with open(args.output_prefix.with_name(f"{args.output_prefix.stem}_metadata.json"), "w", encoding="utf-8") as f:
        json.dump({"documents": doc_meta, "pages": clean_pages}, f, ensure_ascii=False, indent=2)

    print(f"\nVetores latentes salvos em: {npz_path}")
    print("Para carregar no FAISS:")
    print("  import faiss, numpy as np")
    print("  data = np.load('" + str(npz_path) + "')")
    print("  index = faiss.IndexFlatIP(data['vlm_page_embeddings'].shape[1])")
    print("  index.add(data['vlm_page_embeddings'])")


if __name__ == "__main__":
    main()
