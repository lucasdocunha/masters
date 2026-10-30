#!/home/lucas.ocunha/masters/.venv_ocr/bin/python3
"""
Rodada 2 de Confirmação e Isolamento de Falsos Positivos:
1. Confirma os documentos validados nos Lotes 02, 03 e 04 do Tier 2 em annotations_history.json como 'nfse'.
2. Move os lotes confirmados para CONFIRMADOS/.
3. Identifica todos os documentos semelhantes aos excluídos pelo usuário nos lotes restantes
   (Tier 2 lotes 05 a 08 e Tier 3 lotes 01 a 10) usando:
   - Os 77 fornecedores rejeitados acumulados (Daleth Veículos, Parlanet, John Richard, Gran Coffee, EDP, Imobiliárias, Postos, etc.)
   - Detecção precisa de Boletos Bancários (Recibo do Pagador, Comprovante de Entrega)
   - Contas de energia elétrica e utilities
   - Faturas de telecom
4. Move os 2.253 falsos positivos detectados para validacao_nfse_removidos_falsos_positivos/
5. Atualiza manifest_summary.json e relatorio_falsos_positivos.json.
"""

import os
import sys
import json
import time
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import pymupdf

BASE_DIR = Path("/home/lucas.ocunha/masters")
DATA_DIR = BASE_DIR / "data"
BATCH_DIR = DATA_DIR / "validacao_batches" / "validacao_nfse_38k"
HISTORY_FILE = BASE_DIR / "ml_backend" / "models" / "annotations_history.json"
DEST_REMOVED_DIR = DATA_DIR / "validacao_batches" / "validacao_nfse_removidos_falsos_positivos"

REJECTED_SUPPLIERS = [
    'A.M.SILVA COMUNICAÇÃO', 'AFIRMA COMUNICACAO E SERVICOS STARTUP GC LTDA', 'ANGA EXPLORAÇÃO RURAL LTDA',
    'AS HOTELARIAE EVENTOS LTDA', 'ASK TECNOLOGIA E COMUNICACAO', 'ATA AEROTAXI ABAETE LTDA',
    'AUTO POSTO FERNANDO STECCA LTDA', 'AUTO POSTO SANPETRO LTDA', 'BLESSING COMUNICACAO LTDA',
    'CID IMOVEIS EIRELI', 'CID IMÓVEIS', 'CLAUDIO ALBERTO FAUSTINO DA SILVA',
    'COMPANHIA ENERGÉTICA DE PERNAMBUCO', 'COND DO EDF ENSEADA OFFICE', 'COND.ED.BELVEDERE II',
    'CONDE & DELL ARINGA - DES DE SOFTWARES', 'CONDOMINIO WALK BUENO BUSINESS E LIFESTYLE',
    'CORPVS SEGURANÇA ELETRÔNICA LTDA', 'DALETH VEICULOS LTDA EPP', 'DEIMISSON BRUNO DE OLIVEIRA 09528358942',
    'DIEGO DE M C ALVES COMERCIO DE VEICULOS NOVOS E US', 'EDIFICIO COMERCIAL E RESIDENCIAL MONTREAL',
    'EDP ESPIRITO SANTO DISTRIBUICAO DE ENERGIA S.A.', 'ENGEMAK EIRELI ME', 'ENNGE SOLUCOES EM INFORMATICA LTDA',
    'GP SISTEMAS E SERVIÇOS LTDA', 'GRAN COFFEE', 'H2L EQUIPAMENTOS', 'HOTEL POUSADA REIS LTDA',
    'IBAGY IMÓVEIS', 'IMOBILIARIA ATLANTICA LTDA', 'IMOBILIARIA ZIBELL', 'IMOBILIÁRIA PRATES',
    'IMOVEIS NO CENTRO', 'INTERVEL TELECOMUNICAÇÕES LTDA', 'JOHN RICHARD LOCAÇÃO DE MÓVEIS LTDA',
    'K-PLATZ HOTEL', 'KERAXWEB SERVIÇOS DE INTERLIGAÇÃO E INTERNET LTDA', 'KHRONOS SEGURANCA PRIVADA LTDA',
    'KHRONOS SEGURANÇA PRIVADA LTDA', 'LOCALIZA', 'HERTZ', 'LUARA KRASNIEVICZ DOMINSKI',
    'MEGA ADM. DE IMÓVEIS LTDA', 'MEGA LOCADORA LTDA', 'MP ASSESSORIA GESTAO E MARKETING LTDA',
    'ORGBRISTOL ORGANIZACOES BRISTOL LTDA', 'ORSEGUPS MONITORAMENTO ELETRÔNICO LTDA',
    'PARANAVEL COMERCIAL DE VEICULOS LTDA', 'PARLANET WEB E TECNOLOGIA EIRELI', 'PILAR CONSULTORIA EMPRESARIAL LTDA',
    'POSTO DA SAUDADE LTDA', 'RADIO CHOPINZINHO LTDA', 'RECANTO SUL COM ADM IMOV LTDA.',
    'RODRIGO BORGHI DA SILVA E CIA LTDA NICKNETWORK', 'RODRIGUES ASSESSORIA ECEM', 'SANEAGO',
    'SCM PARTICIPAÇÕES SA', 'SINDICATO DE HOTÉIS RESTAURANTES BARES E SIMILAR', 'SOLARIS IMÓVEIS LTDA.',
    'STATIC TONER DO BRASIL LTDA.', 'TAXI AEREO HERCULES LTDA.',
    'TECNOPRINTER EA SUPRIMENTOS PARA IMPRESSORA EIRELI', 'TECNOVETTI PRODUTOS E SERV. PARA ESCRITORIO LTDA',
    'TERRITORIAL EMPREENDIMENTOS IMOBILIARIOS LTDA.', 'TERRITORIO EMPREENDIMENTOS IMOBILIARIOS',
    'WORKS INFORMÁTICA COMERCIAL LTDA.', 'ÍRIO JOSÉ FELIPE DIAS LTDA', 'MOCELLIN & CONSOLLI',
    'CRV ASSESSORIA IMOBILIÁRIA', 'E A DE SOUZA', 'TERMINAL GARAGEM MENEZES', 'PRONTO FIBRA',
    'BRASIL DIGITAL SERVIÇO', 'WORLDNET / NBS', 'RPW MONITORAMENTO', 'N DE SOUZA COSTA'
]

# Carregado no escopo global para que multiprocessing aproveite COW do fork Linux sem serialização
with open(DATA_DIR / "pdf_metadata_lookup.json", "r", encoding="utf-8") as f:
    LOOKUP = json.load(f)

def check_doc_worker(path_str):
    p = Path(path_str)
    stem = p.stem
    info = LOOKUP.get(f"{stem}.pdf", {})
    forn = info.get("fornecedor", "").strip()
    forn_u = forn.upper()
    subc = info.get("subcota", "").strip()

    # 1. Supplier match
    for k in REJECTED_SUPPLIERS:
        if k in forn_u:
            cat = "09_outros_falsos_positivos"
            if any(w in forn_u for w in ["LOCALIZA", "HERTZ", "DALETH", "MEGA LOCADORA", "PARANAVEL"]):
                cat = "01_locacao_veiculos_e_contratos"
            elif any(w in forn_u for w in ["ORSEGUPS", "E A DE SOUZA", "CORPVS", "KHRONOS", "RPW", "N DE SOUZA"]):
                cat = "02_monitoramento_e_seguranca"
            elif any(w in forn_u for w in ["GP SISTEMAS", "WORKS INFORMÁTICA", "INCFAN", "TECNOPRINTER", "ENNGE", "STATIC TONER", "ASK TECNOLOGIA"]):
                cat = "03_software_ti_e_boletos"
            elif any(w in forn_u for w in ["PARLANET", "KERAXWEB", "INTERVEL", "PRONTO FIBRA", "NICKNETWORK", "BRASIL DIGITAL", "WORLDNET"]):
                cat = "04_telecom_e_internet"
            elif any(w in forn_u for w in ["A.M.SILVA", "RADIO", "RÁDIO", "GO EDITORAÇÃO", "AFIRMA COMUNICACAO"]):
                cat = "05_publicidade_e_radio"
            elif any(w in forn_u for w in ["ZIBELL", "BELVEDERE", "ATLANTICA", "SOLARIS", "CID IM", "TERRITORIO", "MOCELLIN", "CRV ASSESSORIA", "RECANTO SUL", "EDIFICIO", "CONDOMINIO"]):
                cat = "06_imobiliaria_condominio_aluguel"
            elif any(w in forn_u for w in ["EDP", "ENERGÉTICA DE PERNAMBUCO", "SANEAGO"]):
                cat = "07_energia_eletrica_e_utilities"
            elif any(w in forn_u for w in ["JOHN RICHARD", "GRAN COFFEE"]):
                cat = "08_locacao_bens_moveis"
            return {
                "stem": stem,
                "fornecedor": forn,
                "subcota": subc,
                "motivo": "fornecedor_rejeitado",
                "categoria": cat,
                "path": path_str
            }

    # 2. Text check (Boletos, Utilities, etc.)
    pdf_p = Path(f"/media/ssd2/datasets/CEAP/{stem}.pdf")
    if pdf_p.exists():
        try:
            doc = pymupdf.open(str(pdf_p))
            if len(doc) > 0:
                txt = doc[0].get_text().upper()
                doc.close()
                if "NOTA FISCAL DE SERVIÇOS ELETRÔNICA" in txt or "NOTA FISCAL ELETRÔNICA DE SERVIÇOS" in txt or "DANFSE" in txt:
                    return None
                if "RECIBO DO PAGADOR" in txt or "COMPROVANTE DE ENTREGA" in txt or "BOLETO BANCÁRIO" in txt or "AUTENTICAÇÃO MECÂNICA" in txt or "BOLETO DE COBRANÇA" in txt:
                    return {
                        "stem": stem,
                        "fornecedor": forn,
                        "subcota": subc,
                        "motivo": "boleto_bancario",
                        "categoria": "08_boletos_bancarios_gerais",
                        "path": path_str
                    }
                if "CUSTO DE DISPONIBILIDADE" in txt or "CONSUMO KWH" in txt or "ENERGIA ELÉTRICA" in txt:
                    return {
                        "stem": stem,
                        "fornecedor": forn,
                        "subcota": subc,
                        "motivo": "energia_eletrica",
                        "categoria": "07_energia_eletrica_e_utilities",
                        "path": path_str
                    }
                if "FATURA DETALHADA" in txt and ("TELECOM" in forn_u or "INTERNET" in forn_u):
                    return {
                        "stem": stem,
                        "fornecedor": forn,
                        "subcota": subc,
                        "motivo": "telecom_fatura",
                        "categoria": "04_telecom_e_internet",
                        "path": path_str
                    }
        except Exception:
            pass
    return None

def main():
    t0 = time.time()
    dest_removed_dir = DEST_REMOVED_DIR
    dest_removed_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================================")
    print(" 1. CONFIRMAÇÃO DOS LOTES 02, 03 E 04 VALIDADOS PELO USUÁRIO")
    print("==================================================================")

    t2 = BATCH_DIR / "2_similares_97_a_98pct"
    t2_l2 = t2 / "lote_02_2000"
    t2_l3 = t2 / "lote_03_2000"
    t2_l4 = t2 / "lote_04_2000"

    t2_l2_files = sorted(list(t2_l2.glob("*.png"))) if t2_l2.exists() else []
    t2_l3_files = sorted(list(t2_l3.glob("*.png"))) if t2_l3.exists() else []
    t2_l4_files = sorted(list(t2_l4.glob("*.png"))) if t2_l4.exists() else []

    all_confirmed = t2_l2_files + t2_l3_files + t2_l4_files
    confirmados_dir = BATCH_DIR / "CONFIRMADOS"

    if all_confirmed:
        print(f"Arquivos mantidos para confirmação:")
        print(f"  • Tier 2 Lote 02: {len(t2_l2_files):,} docs")
        print(f"  • Tier 2 Lote 03: {len(t2_l3_files):,} docs")
        print(f"  • Tier 2 Lote 04: {len(t2_l4_files):,} docs")
        print(f"  • Total a confirmar nesta rodada: {len(all_confirmed):,} docs")

        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)

        added = 0
        for f in all_confirmed:
            stem = f.stem
            if stem not in history or history[stem] != "nfse":
                history[stem] = "nfse"
                added += 1

        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        print(f"[OK] {added:,} novos documentos confirmados como 'nfse' em annotations_history.json.")

        dest_t2_l2 = confirmados_dir / f"tier2_lote_02_{len(t2_l2_files)}"
        dest_t2_l3 = confirmados_dir / f"tier2_lote_03_{len(t2_l3_files)}"
        dest_t2_l4 = confirmados_dir / f"tier2_lote_04_{len(t2_l4_files)}"

        for src_dir, dst_dir in [(t2_l2, dest_t2_l2), (t2_l3, dest_t2_l3), (t2_l4, dest_t2_l4)]:
            if src_dir.exists():
                dst_dir.mkdir(parents=True, exist_ok=True)
                for png in src_dir.glob("*.png"):
                    shutil.move(str(png), str(dst_dir / png.name))
                shutil.rmtree(src_dir)
        print("  Pastas de lotes 02, 03 e 04 consolidadas em CONFIRMADOS/.")
    else:
        print("  Lotes 02, 03 e 04 já haviam sido integrados em CONFIRMADOS/.")

    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        history = json.load(f)
    print(f"  Total acumulado no histórico: {len(history):,} anotações.")

    print("\n==================================================================")
    print(" 2. IDENTIFICAÇÃO E ISOLAMENTO DOS FALSOS POSITIVOS NOS RESTANTES")
    print("==================================================================")

    t3 = BATCH_DIR / "3_similares_96_a_97pct"
    unchecked_files = []
    if t2.exists():
        for lote_p in sorted(t2.iterdir()):
            if lote_p.is_dir():
                unchecked_files.extend(sorted(list(lote_p.glob("*.png"))))

    if t3.exists():
        for lote_p in sorted(t3.iterdir()):
            if lote_p.is_dir():
                unchecked_files.extend(sorted(list(lote_p.glob("*.png"))))

    print(f"Total de imagens pendentes de verificação: {len(unchecked_files):,}")

    t_scan = time.time()
    n_workers = min(24, os.cpu_count() or 8)
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        scan_results = list(pool.map(check_doc_worker, [str(p) for p in unchecked_files], chunksize=100))

    to_isolate = [r for r in scan_results if r is not None]
    print(f"Varredura concluída em {time.time() - t_scan:.1f}s!")
    print(f"Total de falsos positivos detectados para isolamento: {len(to_isolate):,}")

    cat_counts = Counter(r["categoria"] for r in to_isolate)
    print("\nDivisão por Categoria de Falso Positivo:")
    for cat, count in sorted(cat_counts.items()):
        print(f"  • {cat}: {count:,} documentos")

    # Mover arquivos para validacao_nfse_removidos_falsos_positivos/
    print(f"\n3. Movendo {len(to_isolate):,} imagens para a pasta de isolamento...")
    moved_count = 0
    for item in to_isolate:
        cat_dir = dest_removed_dir / item["categoria"]
        cat_dir.mkdir(parents=True, exist_ok=True)
        src_p = Path(item["path"])
        dst_p = cat_dir / src_p.name
        if src_p.exists():
            shutil.move(str(src_p), str(dst_p))
            moved_count += 1

    print(f"[OK] {moved_count:,} arquivos transferidos com sucesso!")

    # Atualizar relatório acumulado de falsos positivos
    report_p = dest_removed_dir / "relatorio_falsos_positivos.json"
    old_data = []
    if report_p.exists():
        try:
            with open(report_p, "r", encoding="utf-8") as f:
                old_data = json.load(f).get("documentos", [])
        except Exception:
            pass

    combined_removed = old_data + to_isolate
    combined_cats = Counter(d.get("categoria", "outros") for d in combined_removed)
    with open(report_p, "w", encoding="utf-8") as f:
        json.dump({
            "total_isolados_acumulado": len(combined_removed),
            "categorias_acumuladas": dict(combined_cats),
            "novos_desta_rodada": len(to_isolate),
            "documentos": combined_removed
        }, f, indent=2, ensure_ascii=False)

    print("\n==================================================================")
    print(" 4. STATUS ATUALIZADO DOS LOTES RESTANTES")
    print("==================================================================")

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
        "validados_confirmados_total": sum(1 for v in history.values() if v == "nfse"),
        "validados_nesta_rodada": 5745,
        "falsos_positivos_isolados_nesta_rodada": len(to_isolate),
        "total_falsos_positivos_isolados": len(combined_removed),
        "total_pendente_validacao": total_remaining,
        "distribuicao_pendente": {
            "tier2_97_a_98pct": len(t2_remaining),
            "tier3_96_a_97pct": len(t3_remaining)
        },
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(BATCH_DIR / "manifest_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    print(f"\nOperação concluída com sucesso em {elapsed:.1f}s!")

if __name__ == "__main__":
    main()
