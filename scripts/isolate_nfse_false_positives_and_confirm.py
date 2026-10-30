#!/home/lucas.ocunha/masters/.venv_ocr/bin/python3
"""
1. Confirma os 5.770 documentos validados pelo usuário nos Lotes 1 e 2 do Tier 1 (98-100%)
   e Lote 1 do Tier 2 (97-98%) em annotations_history.json como 'nfse'.
2. Move os confirmados para a pasta CONFIRMADOS/ organizada em sublotes.
3. Identifica todos os documentos semelhantes aos 38 removidos (falsos positivos)
   nos lotes restantes não verificados (Tier 2 lotes 02 a 08 e Tier 3 lotes 01 a 10).
4. Remove-os de validacao_nfse_38k e isola-os em:
   data/validacao_batches/validacao_nfse_removidos_falsos_positivos/
   categorizados por tipo de documento/fornecedor.
5. Atualiza o manifest_summary.json.
"""

import os
import sys
import json
import time
import shutil
from pathlib import Path
from collections import Counter
import numpy as np

def main():
    t0 = time.time()
    base_dir = Path("/home/lucas.ocunha/masters")
    data_dir = base_dir / "data"
    batch_dir = data_dir / "validacao_batches" / "validacao_nfse_38k"
    history_file = base_dir / "ml_backend" / "models" / "annotations_history.json"
    dest_removed_dir = data_dir / "validacao_batches" / "validacao_nfse_removidos_falsos_positivos"
    dest_removed_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================================")
    print(" 1. CONFIRMAÇÃO DOS 5.770 DOCUMENTOS VALIDADOS PELO USUÁRIO")
    print("==================================================================")

    t1_l1 = batch_dir / "1_similares_98_a_100pct" / "lote_01_2000"
    t1_l2 = batch_dir / "1_similares_98_a_100pct" / "lote_02_1805"
    t2_l1 = batch_dir / "2_similares_97_a_98pct" / "lote_01_2000"

    t1_l1_files = sorted(list(t1_l1.glob("*.png"))) if t1_l1.exists() else []
    t1_l2_files = sorted(list(t1_l2.glob("*.png"))) if t1_l2.exists() else []
    t2_l1_files = sorted(list(t2_l1.glob("*.png"))) if t2_l1.exists() else []

    all_confirmed_files = t1_l1_files + t1_l2_files + t2_l1_files
    print(f"Arquivos encontrados para confirmação:")
    print(f"  • Tier 1 Lote 01: {len(t1_l1_files):,} docs")
    print(f"  • Tier 1 Lote 02: {len(t1_l2_files):,} docs")
    print(f"  • Tier 2 Lote 01: {len(t2_l1_files):,} docs")
    print(f"  • Total a confirmar: {len(all_confirmed_files):,} docs")

    # Ingestão em annotations_history.json
    with open(history_file, "r", encoding="utf-8") as f:
        history = json.load(f)

    added = 0
    for f in all_confirmed_files:
        stem = f.stem
        if stem not in history or history[stem] != "nfse":
            history[stem] = "nfse"
            added += 1

    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    print(f"[OK] Ingeridos {added:,} novos documentos como 'nfse' em annotations_history.json.")
    print(f"     Total acumulado em annotations_history: {len(history):,} anotações.")

    # Mover os confirmados para CONFIRMADOS/
    confirmados_dir = batch_dir / "CONFIRMADOS"
    confirmados_dir.mkdir(parents=True, exist_ok=True)

    dest_t1_l1 = confirmados_dir / "tier1_lote_01_2000"
    dest_t1_l2 = confirmados_dir / f"tier1_lote_02_{len(t1_l2_files)}"
    dest_t2_l1 = confirmados_dir / f"tier2_lote_01_{len(t2_l1_files)}"

    for src_dir, dst_dir in [(t1_l1, dest_t1_l1), (t1_l2, dest_t1_l2), (t2_l1, dest_t2_l1)]:
        if src_dir.exists():
            dst_dir.mkdir(parents=True, exist_ok=True)
            for png in src_dir.glob("*.png"):
                shutil.move(str(png), str(dst_dir / png.name))
            shutil.rmtree(src_dir)

    # Remover pasta 1_similares_98_a_100pct se vazia (já totalmente confirmada)
    t1_parent = batch_dir / "1_similares_98_a_100pct"
    if t1_parent.exists() and not any(t1_parent.iterdir()):
        shutil.rmtree(t1_parent)
        print("  Pasta '1_similares_98_a_100pct' 100% validada e movida para CONFIRMADOS/.")

    print("\n==================================================================")
    print(" 2. IDENTIFICAÇÃO DOS FALSOS POSITIVOS NOS LOTES RESTANTES")
    print("==================================================================")

    removed_stems = [
        "7040575", "7048260", "7048893", "7056547", "7057203", "7065423", "7075101", "7093634", 
        "7130896", "7138988", "7140754", "7187664", "7201085", "7275522", "7403187", "7415199", 
        "7471431", "7471761", "7482933", "7483282", "7626420", "7634364", "7672663", "7692865", 
        "7728103", "7772118", "7794133", "7816523", "7829294", "7858735", "7943819", "7958054", 
        "7958064", "7962952", "7962973", "7981291", "8014454", "8028722"
    ]
    print(f"Sementes negativas (removidas pelo usuário): {len(removed_stems)}")

    # Carregar metadados e embeddings
    meta = json.load(open(data_dir / "latent_vectors_full_metadata.json"))["documents"]
    stem_to_idx = {Path(d["doc_name"]).stem: i for i, d in enumerate(meta)}
    lookup = json.load(open(data_dir / "pdf_metadata_lookup.json"))

    npz = np.load(data_dir / "latent_vectors_full.npz", mmap_mode="r")
    dino = npz["dino_doc_embeddings"]

    neg_idx = [stem_to_idx[s] for s in removed_stems if s in stem_to_idx]
    neg_vecs = dino[neg_idx].astype(np.float32)
    neg_vecs /= np.linalg.norm(neg_vecs, axis=1, keepdims=True)

    # Identificar todos os PNGs não verificados restantes
    t2 = batch_dir / "2_similares_97_a_98pct"
    t3 = batch_dir / "3_similares_96_a_97pct"

    unchecked_pngs = []
    if t2.exists():
        for lote_p in sorted(t2.iterdir()):
            if lote_p.is_dir() and "lote_01" not in lote_p.name:
                unchecked_pngs.extend(sorted(list(lote_p.glob("*.png"))))

    if t3.exists():
        for lote_p in sorted(t3.iterdir()):
            if lote_p.is_dir():
                unchecked_pngs.extend(sorted(list(lote_p.glob("*.png"))))

    print(f"Total de imagens pendentes de verificação: {len(unchecked_pngs):,}")

    unv_stems = [p.stem for p in unchecked_pngs]
    unv_indices = [stem_to_idx[s] for s in unv_stems]
    unv_vecs = dino[unv_indices].astype(np.float32)
    unv_vecs /= np.linalg.norm(unv_vecs, axis=1, keepdims=True)

    all_sims_neg = np.dot(unv_vecs, neg_vecs.T)
    sim_max_neg = np.max(all_sims_neg, axis=1)
    best_neg_idx = np.argmax(all_sims_neg, axis=1)

    neg_kws = [
        "LOCALIZA", "HERTZ", "ORSEGUPS", "KERAXWEB", "INTERVEL", "ZIBELL", 
        "GP SISTEMAS", "BELVEDERE", "MEGA ADM", "TECNOPRINTER", "WORKS INFORMÁTICA", 
        "A.M.SILVA", "A.M. SILVA", "PRONTO FIBRA", "E A DE SOUZA MANITORAMENTO", 
        "E A DE SOUZA MONITORAMENTO", "TERMINAL GARAGEM MENEZES", "INCFAN SISTEMAS", 
        "IMOVEIS NO CENTRO", "RODRIGUES ASSESSORIA ECEM", "CLICK PRODUTOS E SERVIÇOS", 
        "RPW MONITORAMENTO", "N DE SOUZA COSTA MONITORAMENTO", "BRASIL DIGITAL SERVIÇO DE INFORMÁTICA", 
        "WORLDNET / NBS"
    ]

    def get_category(forn, stem, sim_val):
        forn_u = forn.upper()
        if any(k in forn_u for k in ["LOCALIZA", "HERTZ", "PARIMA RENT"]):
            return "01_locacao_veiculos_localiza"
        if any(k in forn_u for k in ["ORSEGUPS", "E A DE SOUZA", "RPW MONITORAMENTO", "N DE SOUZA COSTA"]):
            return "02_monitoramento_e_seguranca"
        if any(k in forn_u for k in ["GP SISTEMAS", "WORKS INFORMÁTICA", "INCFAN", "TECNOPRINTER", "CLICK PRODUTOS", "H2L", "SATURNO"]):
            return "03_software_ti_e_boletos"
        if any(k in forn_u for k in ["KERAXWEB", "INTERVEL", "PRONTO FIBRA", "BRASIL DIGITAL", "WORLDNET", "NEX DATA"]):
            return "04_telecom_e_internet"
        if any(k in forn_u for k in ["A.M.SILVA", "A.M. SILVA", "RADIO", "RÁDIO", "GO EDITORAÇÃO", "DESIGN GRAFICO", "ALUGUE GRAFICA", "DIFUSORA"]):
            return "05_publicidade_e_radio"
        if any(k in forn_u for k in ["ZIBELL", "BELVEDERE II", "IMOVEIS NO CENTRO", "MEGA ADM", "NN EMPREENDIMENTOS", "CONDOMINIO"]):
            return "06_imobiliaria_condominio_aluguel"
        return "07_outros_falsos_positivos"

    to_isolate = []
    for i, s in enumerate(unv_stems):
        info = lookup.get(f"{s}.pdf", {})
        forn = info.get("fornecedor", "").strip()
        sim_n = float(sim_max_neg[i])
        matched_seed = removed_stems[best_neg_idx[i]]
        src_png = unchecked_pngs[i]

        is_neg_entity = any(k in forn.upper() for k in neg_kws)
        is_very_similar = (sim_n >= 0.985)

        if is_neg_entity or is_very_similar:
            category = get_category(forn, s, sim_n)
            to_isolate.append({
                "doc_id": s,
                "fornecedor": forn,
                "subcota": info.get("subcota", ""),
                "sim_neg": round(sim_n, 4),
                "matched_seed": matched_seed,
                "categoria": category,
                "src_png": str(src_png),
                "original_lote": src_png.parent.name,
                "original_tier": src_png.parent.parent.name
            })

    print(f"Total de falsos positivos detectados para isolamento: {len(to_isolate):,}")

    cat_counts = Counter(item["categoria"] for item in to_isolate)
    print("\nDivisão por Categoria de Falso Positivo:")
    for cat, count in sorted(cat_counts.items()):
        print(f"  • {cat}: {count} documentos")

    # Mover arquivos para as pastas categorizadas
    print(f"\n3. Movendo {len(to_isolate):,} imagens para {dest_removed_dir.name}...")
    moved_count = 0
    for item in to_isolate:
        cat_dir = dest_removed_dir / item["categoria"]
        cat_dir.mkdir(parents=True, exist_ok=True)
        src_p = Path(item["src_png"])
        dst_p = cat_dir / src_p.name
        if src_p.exists():
            shutil.move(str(src_p), str(dst_p))
            moved_count += 1

    print(f"[OK] {moved_count:,} arquivos transferidos com sucesso!")

    # Salvar relatório detalhado
    report_p = dest_removed_dir / "relatorio_falsos_positivos.json"
    with open(report_p, "w", encoding="utf-8") as f:
        json.dump({
            "total_isolados": len(to_isolate),
            "categorias": dict(cat_counts),
            "documentos": to_isolate
        }, f, indent=2, ensure_ascii=False)

    # 4. Atualizar contagens e manifest_summary.json
    print("\n==================================================================")
    print(" 4. ATUALIZAÇÃO DOS LOTES PENDENTES")
    print("==================================================================")

    # Contar documentos restantes nas pastas
    t2_remaining = list(t2.rglob("*.png")) if t2.exists() else []
    t3_remaining = list(t3.rglob("*.png")) if t3.exists() else []
    total_remaining = len(t2_remaining) + len(t3_remaining)

    print(f"Lotes restantes limpos para validação:")
    print(f"  • Tier 2 (97% a 98%): {len(t2_remaining):,} documentos restantes")
    print(f"  • Tier 3 (96% a 97%): {len(t3_remaining):,} documentos restantes")
    print(f"  • Total pendente:     {total_remaining:,} documentos limpos")

    summary = {
        "status": "em_validacao",
        "total_inicial_38k": 38826,
        "validados_confirmados": len(all_confirmed_files),
        "falsos_positivos_removidos_usuario": len(removed_stems),
        "falsos_positivos_isolados_automaticamente": len(to_isolate),
        "total_pendente_validacao": total_remaining,
        "distribuicao_pendente": {
            "tier2_97_a_98pct": len(t2_remaining),
            "tier3_96_a_97pct": len(t3_remaining)
        },
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(batch_dir / "manifest_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    print(f"\nOperação concluída com sucesso em {elapsed:.1f}s!")

if __name__ == "__main__":
    main()
