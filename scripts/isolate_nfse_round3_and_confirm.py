#!/home/lucas.ocunha/masters/.venv_ocr/bin/python3
"""
Rodada 3 de Confirmação e Isolamento de Falsos Positivos:
1. Confirma os 6.741 documentos validados nos Lotes 05, 06, 07 e 08 do Tier 2 em annotations_history.json como 'nfse'.
2. Sincroniza os novos confirmados na tabela 'documento' do PostgreSQL.
3. Remove as pastas validadas (lotes 05 a 08 e diretório 2_similares_97_a_98pct).
4. Identifica e isola os falsos positivos no Tier 3 (3_similares_96_a_97pct) com base nos fornecedores rejeitados
   e detecção de DANFE / NF-e de combustível / boletos / faturas.
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
import psycopg

BASE_DIR = Path("/home/lucas.ocunha/masters")
DATA_DIR = BASE_DIR / "data"
BATCH_DIR = DATA_DIR / "validacao_batches" / "validacao_nfse_38k"
HISTORY_FILE = BASE_DIR / "data" / "models" / "annotations_history.json"
DEST_REMOVED_DIR = DATA_DIR / "validacao_batches" / "validacao_nfse_removidos_falsos_positivos"
DB_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/ceap")

# Fornecedores rejeitados acumulados (Rodadas 1, 2 e 3)
REJECTED_SUPPLIERS = [
    # Rodada 1 e 2
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
    'BRASIL DIGITAL SERVIÇO', 'WORLDNET / NBS', 'RPW MONITORAMENTO', 'N DE SOUZA COSTA',
    # Novos Rodada 3 (Lotes 5 a 8)
    'ADMINISTRADORA FONSECA IMÓVEIS', 'FONSECA IMÓVEIS', 'MOOJEN ANDRADA', 'CESAR REGO IMOVEIS',
    'BLOG DO SILVANO DIAS', 'SILVANO DIAS', 'PATRICIA DOS SANTOS ALENCAR', 'AUTO POSTO BRASIL GV',
    'SJ ADMIMINSTRAÇÃO DE  IMOVEIS', 'SJ ADMIMINSTRAÇÃO', 'NSF TRANSPORTES', 'IBIS FLORIANOPOLIS',
    'GPO EMPREENDIMENTOS', 'MÁRCIO MALAFAIA', 'PEDRO BASSO', 'MHNET', 'REDE OLT HOTEIS',
    'ELIELSON LEANDRO', 'BLOG DO ELIELSON', 'UNIDAS SA', 'CLICK PRODUTOS', 'DAMPETRO',
    'AGENCIA ESTADO', 'COMPUMAKE', 'PAPELARIA TRIBUTÁRIA', 'BOCAIUVA', 'MORGANA DRESCH',
    'HOTEL PHENICIA', 'NEX DATA'
]

# Lookup global para COW
with open(DATA_DIR / "pdf_metadata_lookup.json", "r", encoding="utf-8") as f:
    LOOKUP = json.load(f)


def check_doc_worker(path_str):
    p = Path(path_str)
    stem = p.stem
    info = LOOKUP.get(f"{stem}.pdf", {})
    forn = info.get("fornecedor", "").strip()
    forn_u = forn.upper()
    subc = info.get("subcota", "").strip()

    # 1. Checagem por Fornecedor Rejeitado
    for k in REJECTED_SUPPLIERS:
        if k in forn_u:
            cat = "09_outros_falsos_positivos"
            if any(w in forn_u for w in ["LOCALIZA", "HERTZ", "DALETH", "MEGA LOCADORA", "PARANAVEL", "UNIDAS", "NSF TRANSPORTES"]):
                cat = "01_locacao_veiculos_e_transportes"
            elif any(w in forn_u for w in ["ORSEGUPS", "E A DE SOUZA", "CORPVS", "KHRONOS", "RPW", "N DE SOUZA"]):
                cat = "02_monitoramento_e_seguranca"
            elif any(w in forn_u for w in ["GP SISTEMAS", "WORKS INFORMÁTICA", "INCFAN", "TECNOPRINTER", "ENNGE", "STATIC TONER", "ASK TECNOLOGIA", "COMPUMAKE"]):
                cat = "03_software_ti_e_equipamentos"
            elif any(w in forn_u for w in ["PARLANET", "KERAXWEB", "INTERVEL", "PRONTO FIBRA", "NICKNETWORK", "BRASIL DIGITAL", "WORLDNET", "MHNET", "CLICK PRODUTOS", "NEX DATA"]):
                cat = "04_telecom_e_internet"
            elif any(w in forn_u for w in ["A.M.SILVA", "RADIO", "RÁDIO", "GO EDITORAÇÃO", "AFIRMA COMUNICACAO", "SILVANO DIAS", "ELIELSON", "BOCAIUVA", "AGENCIA ESTADO"]):
                cat = "05_publicidade_e_blogs"
            elif any(w in forn_u for w in ["ZIBELL", "BELVEDERE", "ATLANTICA", "SOLARIS", "CID IM", "TERRITORIO", "MOCELLIN", "CRV ASSESSORIA", "RECANTO SUL", "EDIFICIO", "CONDOMINIO", "FONSECA IMÓVEIS", "MOOJEN ANDRADA", "CESAR REGO", "SJ ADMIMINSTRAÇÃO", "GPO EMPREENDIMENTOS", "PEDRO BASSO", "PATRICIA DOS SANTOS"]):
                cat = "06_imobiliaria_condominio_aluguel"
            elif any(w in forn_u for w in ["EDP", "ENERGÉTICA DE PERNAMBUCO", "SANEAGO"]):
                cat = "07_energia_eletrica_e_utilities"
            elif any(w in forn_u for w in ["POSTO", "DAMPETRO", "COMBUSTIVEL"]):
                cat = "08_combustivel_e_danfe_postos"
            elif any(w in forn_u for w in ["IBIS", "HOTEL"]):
                cat = "09_hotelaria"
            return {
                "stem": stem,
                "fornecedor": forn,
                "subcota": subc,
                "motivo": "fornecedor_rejeitado",
                "categoria": cat,
                "path": path_str
            }

    # 2. Checagem Textual no PDF
    pdf_p = Path(f"/media/ssd2/datasets/CEAP/{stem}.pdf")
    if pdf_p.exists():
        try:
            doc = pymupdf.open(str(pdf_p))
            if len(doc) > 0:
                txt = doc[0].get_text().upper()
                doc.close()
                # Isenção para NFS-e válidas legítimas
                if "NOTA FISCAL DE SERVIÇOS ELETRÔNICA" in txt or "NOTA FISCAL ELETRÔNICA DE SERVIÇOS" in txt or "DANFSE" in txt:
                    return None
                if any(w in txt for w in ["RECIBO DO PAGADOR", "COMPROVANTE DE ENTREGA", "BOLETO BANCÁRIO", "AUTENTICAÇÃO MECÂNICA", "BOLETO DE COBRANÇA"]):
                    return {
                        "stem": stem,
                        "fornecedor": forn,
                        "subcota": subc,
                        "motivo": "boleto_bancario",
                        "categoria": "10_boletos_bancarios_gerais",
                        "path": path_str
                    }
                if "DANFE" in txt and ("DOCUMENTO AUXILIAR DA NOTA FISCAL ELETRÔNICA" in txt or "CHAVE DE ACESSO" in txt):
                    return {
                        "stem": stem,
                        "fornecedor": forn,
                        "subcota": subc,
                        "motivo": "danfe_nfe_mercadoria",
                        "categoria": "08_combustivel_e_danfe_postos",
                        "path": path_str
                    }
                if any(w in txt for w in ["CUSTO DE DISPONIBILIDADE", "CONSUMO KWH", "ENERGIA ELÉTRICA"]):
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
    print(" 1. CONFIRMAÇÃO DOS LOTES 05, 06, 07 E 08 VALIDADOS PELO USUÁRIO")
    print("==================================================================")

    t2 = BATCH_DIR / "2_similares_97_a_98pct"
    t2_l5 = t2 / "lote_05_2000"
    t2_l6 = t2 / "lote_06_2000"
    t2_l7 = t2 / "lote_07_2000"
    t2_l8 = t2 / "lote_08_1342"

    t2_l5_files = sorted(list(t2_l5.glob("*.png"))) if t2_l5.exists() else []
    t2_l6_files = sorted(list(t2_l6.glob("*.png"))) if t2_l6.exists() else []
    t2_l7_files = sorted(list(t2_l7.glob("*.png"))) if t2_l7.exists() else []
    t2_l8_files = sorted(list(t2_l8.glob("*.png"))) if t2_l8.exists() else []

    all_confirmed = t2_l5_files + t2_l6_files + t2_l7_files + t2_l8_files

    if all_confirmed:
        print(f"Arquivos mantidos para confirmação:")
        print(f"  • Tier 2 Lote 05: {len(t2_l5_files):,} docs")
        print(f"  • Tier 2 Lote 06: {len(t2_l6_files):,} docs")
        print(f"  • Tier 2 Lote 07: {len(t2_l7_files):,} docs")
        print(f"  • Tier 2 Lote 08: {len(t2_l8_files):,} docs")
        print(f"  • Total a confirmar nesta rodada: {len(all_confirmed):,} docs")

        # 1.1 Atualizar annotations_history.json
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)

        added = 0
        confirmed_stems = []
        for f in all_confirmed:
            stem = f.stem
            confirmed_stems.append(stem)
            if stem not in history or history[stem] != "nfse":
                history[stem] = "nfse"
                added += 1

        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        print(f"[OK] {added:,} novos documentos confirmados como 'nfse' em annotations_history.json.")

        # 1.2 Sincronizar na tabela 'documento' do PostgreSQL
        print("Sincronizando novos documentos no PostgreSQL (tabela 'documento')...")
        try:
            conn = psycopg.connect(DB_URL)
            with conn.cursor() as cur:
                rows_to_upsert = []
                for stem in confirmed_stems:
                    doc_id = int(stem)
                    meta = LOOKUP.get(f"{stem}.pdf", {})
                    ano = int(meta.get("ano") or 0)
                    caminho = f"/media/ssd2/datasets/CEAP/{stem}.pdf"
                    rows_to_upsert.append((doc_id, ano, caminho, "baixado", "nfse"))

                cur.executemany("""
                    INSERT INTO documento (ide_documento, ano, caminho_arquivo, status, classe_predita)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (ide_documento) DO UPDATE SET
                        status = 'baixado',
                        classe_predita = 'nfse';
                """, rows_to_upsert)
                conn.commit()
            conn.close()
            print(f"[OK] {len(rows_to_upsert):,} documentos sincronizados com sucesso no PostgreSQL!")
        except Exception as e:
            print(f"[AVISO] Erro ao sincronizar no PostgreSQL: {e}")

        # 1.3 Excluir as pastas validadas conforme instrução do usuário
        print("\nExcluindo as pastas validadas dos lotes 05 a 08...")
        for lote_dir in [t2_l5, t2_l6, t2_l7, t2_l8]:
            if lote_dir.exists():
                shutil.rmtree(lote_dir)
                print(f"  • Removida: {lote_dir.name}")

        if t2.exists() and not list(t2.iterdir()):
            t2.rmdir()
            print("  • Removido diretório concluído: 2_similares_97_a_98pct")

    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        history = json.load(f)
    print(f"Total acumulado no histórico: {len(history):,} anotações.")

    print("\n==================================================================")
    print(" 2. IDENTIFICAÇÃO E ISOLAMENTO DOS FALSOS POSITIVOS NO TIER 3")
    print("==================================================================")

    t3 = BATCH_DIR / "3_similares_96_a_97pct"
    unchecked_files = []
    if t3.exists():
        for lote_p in sorted(t3.iterdir()):
            if lote_p.is_dir():
                unchecked_files.extend(sorted(list(lote_p.glob("*.png"))))

    print(f"Total de imagens no Tier 3 pendentes de verificação: {len(unchecked_files):,}")

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
    print(" 4. STATUS ATUALIZADO DOS LOTES RESTANTES (TIER 3)")
    print("==================================================================")

    t3_remaining = list(t3.rglob("*.png")) if t3.exists() else []
    print(f"Lotes do Tier 3 limpos para validação:")
    if t3.exists():
        for lote_p in sorted(t3.iterdir()):
            if lote_p.is_dir():
                cnt = len(list(lote_p.glob("*.png")))
                print(f"  • {lote_p.name}: {cnt:,} documentos")
    print(f"  • Total pendente de validação: {len(t3_remaining):,} documentos")

    total_nfse_history = sum(1 for v in history.values() if v == "nfse")
    summary = {
        "status": "em_validacao",
        "total_inicial_38k": 38826,
        "validados_confirmados_total": total_nfse_history,
        "validados_nesta_rodada": len(all_confirmed),
        "falsos_positivos_isolados_nesta_rodada": len(to_isolate),
        "total_falsos_positivos_isolados": len(combined_removed),
        "total_pendente_validacao": len(t3_remaining),
        "distribuicao_pendente": {
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
