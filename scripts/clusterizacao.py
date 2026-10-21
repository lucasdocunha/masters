#!/usr/bin/env python3
import argparse
import gc
import json
import random
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
        pages = []
        for i, page in enumerate(pdf):
            try:
                img = page.render(scale=1.5).to_pil().convert("RGB")
                text = page.get_textpage().get_text_range().strip()
                pages.append({"page_idx": i, "image": img, "text": text})
            except Exception:
                pass
        return pages
    except Exception:
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(pdf_path)
            pages = []
            for i, p in enumerate(doc):
                try:
                    pix = p.get_pixmap(dpi=150)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    text = p.get_text().strip()
                    pages.append({"page_idx": i, "image": img, "text": text})
                except Exception:
                    pass
            doc.close()
            return pages
        except Exception:
            return []


def get_sampled_files(raw_dir: Path, pdf_dir: Path, years: list[int], sample_rate: float, seed: int = 42) -> list[Path]:
    existing_set = set(p.name for p in pdf_dir.iterdir() if p.name.endswith(".pdf"))
    rng = random.Random(seed)
    sampled = []

    for y in sorted(years):
        json_path = raw_dir / f"Ano-{y}.json"
        if not json_path.exists():
            continue
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        items = data.get("dados", []) if isinstance(data, dict) else data
        urls = sorted({
            it["urlDocumento"]
            for it in items
            if it.get("urlDocumento", "").lower().endswith(".pdf")
        })
        valid_files = [pdf_dir / u.split("/")[-1] for u in urls if u.split("/")[-1] in existing_set]
        k = max(1, int(len(valid_files) * sample_rate))
        chosen = rng.sample(valid_files, k)
        sampled.extend(chosen)
        print(f"[{y}] Amostrados {len(chosen)}/{len(valid_files)} documentos ({sample_rate * 100:.1f}%)")

    return sampled


@torch.no_grad()
def init_dino(model_name: str, device: str):
    print(f"Carregando DINO ({model_name})...", flush=True)
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    return processor, model


@torch.no_grad()
def extract_dino_batch(images: list[Image.Image], processor, model, device: str) -> np.ndarray:
    if not images:
        return np.empty((0, 384), dtype=np.float32)
    embeddings = []
    batch_size = 16
    for i in range(0, len(images), batch_size):
        inputs = processor(images=images[i : i + batch_size], return_tensors="pt").to(device)
        out = torch.nn.functional.normalize(model(**inputs).last_hidden_state[:, 0, :], dim=-1)
        embeddings.append(out.cpu().float().numpy())
    return np.vstack(embeddings)


@torch.no_grad()
def init_vlm(model_name: str, device: str):
    print(f"Carregando VLM ({model_name})...", flush=True)
    if "colpali" in model_name.lower() or "colqwen" in model_name.lower():
        try:
            from colpali_engine.models import ColPali, ColPaliProcessor, ColQwen2, ColQwen2Processor
            cls_model, cls_proc = (ColQwen2, ColQwen2Processor) if "colqwen" in model_name.lower() else (ColPali, ColPaliProcessor)
            processor = cls_proc.from_pretrained(model_name)
            model = cls_model.from_pretrained(
                model_name,
                torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16,
            ).to(device).eval()
            return "colpali", processor, model
        except ImportError:
            print("Aviso: 'colpali-engine' não instalado. Usando fallback para CLIP...")

    from transformers import AutoTokenizer, CLIPModel
    processor = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    return "clip", (processor, tokenizer), model


@torch.no_grad()
def extract_vlm_batch(images: list[Image.Image], texts: list[str], vlm_type: str, processor, model, device: str) -> np.ndarray:
    if not images:
        dim = 128 if vlm_type == "colpali" else 512
        return np.empty((0, dim), dtype=np.float32)

    embeddings = []
    if vlm_type == "colpali":
        batch_size = 4
        for i in range(0, len(images), batch_size):
            inputs = processor.process_images(images[i : i + batch_size]).to(device)
            patch_embs = model(**inputs)  # [B, N_patches, 128]
            doc_vecs = torch.nn.functional.normalize(patch_embs.mean(dim=1), dim=-1)
            embeddings.append(doc_vecs.cpu().float().numpy())
    else:
        # Fallback CLIP
        img_proc, tokenizer = processor
        batch_size = 16
        for i in range(0, len(images), batch_size):
            b_imgs, b_txts = images[i : i + batch_size], texts[i : i + batch_size]
            img_feats = torch.nn.functional.normalize(model.get_image_features(**img_proc(images=b_imgs, return_tensors="pt").to(device)), dim=-1)
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
    parser.add_argument("--input-dir", "-i", type=Path, default=Path("/media/ssd2/datasets/CEAP"), help="Pasta com PDFs ou imagens")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"), help="Pasta com Ano-*.json")
    parser.add_argument("--years", nargs="*", type=int, help="Lista de anos específicos (ex: 2014 2015 ...)")
    parser.add_argument("--sample-rate", type=float, default=1.0, help="Fração amostral (ex: 0.05 para 5%%)")
    parser.add_argument("--sample-seed", type=int, default=42, help="Seed para reprodutibilidade da amostragem")
    parser.add_argument("--output-prefix", "-o", type=Path, default=Path("data/latent_vectors"), help="Prefixo dos arquivos salvos (.npz e .json)")
    parser.add_argument("--dino-model", default="facebook/dinov2-small", help="Modelo DINO (imagem)")
    parser.add_argument("--vlm-model", default="vidore/colpali-v1.2", help="VLM dedicada a documentos (ex: vidore/colpali-v1.2)")
    parser.add_argument("--k-doc", type=int, default=5, help="Clusters de documentos")
    parser.add_argument("--k-page", type=int, default=10, help="Clusters de páginas")
    parser.add_argument("--checkpoint-every", type=int, default=500, help="Salva checkpoint a cada N documentos processados")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    # 1. Definir arquivos a processar
    if args.years and args.sample_rate < 1.0:
        files = get_sampled_files(args.raw_dir, args.input_dir, args.years, args.sample_rate, args.sample_seed)
    elif args.sample_rate < 1.0 and (args.raw_dir / "Ano-2014.json").exists():
        all_years = list(range(2014, 2027))
        files = get_sampled_files(args.raw_dir, args.input_dir, all_years, args.sample_rate, args.sample_seed)
    else:
        files = sorted(args.input_dir.glob("*.pdf")) + sorted(args.input_dir.glob("*.png")) + sorted(args.input_dir.glob("*.jpg"))

    total_files = len(files)
    print(f"Total de documentos para processar: {total_files}")
    if total_files == 0:
        print("Nenhum arquivo encontrado.")
        return

    # 2. Inicializar modelos na GPU
    dino_proc, dino_model = init_dino(args.dino_model, device)
    vlm_type, vlm_proc, vlm_model = init_vlm(args.vlm_model, device)

    # 3. Processamento em Streaming/Batches
    doc_meta = []
    page_meta = []
    dino_page_embs = []
    vlm_page_embs = []

    chunk_size = 32
    print(f"\nIniciando inferência em streaming com lotes de {chunk_size} documentos...", flush=True)

    for i in range(0, total_files, chunk_size):
        chunk_files = files[i : i + chunk_size]
        batch_images = []
        batch_texts = []
        batch_page_meta = []

        for file_path in chunk_files:
            pages = render_pdf_pages(file_path) if file_path.suffix.lower() == ".pdf" else [{"page_idx": 0, "image": Image.open(file_path).convert("RGB"), "text": ""}]
            if not pages:
                continue

            doc_id = len(doc_meta)
            doc_meta.append({"doc_id": doc_id, "doc_name": file_path.name, "num_pages": len(pages)})
            for p in pages:
                batch_images.append(p["image"])
                batch_texts.append(p["text"])
                batch_page_meta.append({"doc_id": doc_id, "doc_name": file_path.name, "page_idx": p["page_idx"], "text": p["text"]})

        if batch_images:
            d_embs = extract_dino_batch(batch_images, dino_proc, dino_model, device)
            v_embs = extract_vlm_batch(batch_images, batch_texts, vlm_type, vlm_proc, vlm_model, device)
            dino_page_embs.append(d_embs)
            vlm_page_embs.append(v_embs)
            page_meta.extend(batch_page_meta)

        # Liberar memória das imagens imediatamente
        del batch_images
        del batch_texts
        gc.collect()

        processed_docs = min(i + chunk_size, total_files)
        if processed_docs % 100 == 0 or processed_docs == total_files:
            pct = (processed_docs * 100) // total_files
            print(f"\r[{processed_docs}/{total_files}] ({pct}%) documentos processados ({len(page_meta)} páginas)", end="", flush=True)

    print("\n\nInferência concluída. Consolidando embeddings...", flush=True)

    dino_page_embs = np.vstack(dino_page_embs) if dino_page_embs else np.empty((0, 384), dtype=np.float32)
    vlm_page_embs = np.vstack(vlm_page_embs) if vlm_page_embs else np.empty((0, 128), dtype=np.float32)

    print(f"Páginas totais: {len(page_meta)} | DINO: {dino_page_embs.shape} | VLM: {vlm_page_embs.shape}")

    dino_doc_embs = pool_document_embeddings(dino_page_embs, page_meta, len(doc_meta))
    vlm_doc_embs = pool_document_embeddings(vlm_page_embs, page_meta, len(doc_meta))

    print(f"Documentos totais: {len(doc_meta)} | DINO: {dino_doc_embs.shape} | VLM: {vlm_doc_embs.shape}")

    print("Executando clusterização KMeans...", flush=True)
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
    print(f"  data = np.load('{npz_path}')")
    print("  index = faiss.IndexFlatIP(data['vlm_page_embeddings'].shape[1])")
    print("  index.add(data['vlm_page_embeddings'])")


if __name__ == "__main__":
    main()
