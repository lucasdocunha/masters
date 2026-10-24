from pathlib import Path
from scripts.db_import_ceap import (
    clean_digits,
    parse_date,
    build_deputado_record,
    build_fornecedor_record,
    build_subcota_record,
    build_documento_record,
    build_despesa_record,
)


def test_clean_digits() -> None:
    assert clean_digits("085.324.290/0013-1") == "08532429000131"
    assert clean_digits("123.456.789-00") == "12345678900"
    assert clean_digits(None) == ""
    assert clean_digits("") == ""


def test_parse_date() -> None:
    assert parse_date("2024-02-07T00:00:00") == "2024-02-07"
    assert parse_date("07/02/2024") == "2024-02-07"
    assert parse_date(None) is None


def test_record_builders() -> None:
    sample_item = {
        "nomeParlamentar": "DEPUTADO TESTE",
        "cpf": "12345678900",
        "siglaUF": "PR",
        "siglaPartido": "PARTIDO",
        "legislatura": 57,
        "numeroSubCota": 1,
        "descricao": "MANUTENÇÃO DE ESCRITÓRIO",
        "numeroEspecificacaoSubCota": 0,
        "descricaoEspecificacao": "",
        "fornecedor": "FORNECEDOR LTDA",
        "cnpjCPF": "085.324.290/0013-1",
        "numero": "1021",
        "tipoDocumento": 0,
        "dataEmissao": "2024-02-07T00:00:00",
        "valorDocumento": 1148.7,
        "valorGlosa": 0,
        "valorLiquido": 1148.7,
        "mes": 2,
        "ano": 2024,
        "numeroDeputadoID": 2812,
        "idDocumento": 7696122,
        "urlDocumento": "https://www.camara.leg.br/7696122.pdf",
    }

    dep = build_deputado_record(sample_item)
    assert dep is not None
    assert dep["id_deputado"] == 2812
    assert dep["sg_uf"] == "PR"

    forn = build_fornecedor_record(sample_item)
    assert forn is not None
    assert forn["cnpj_cpf"] == "08532429000131"
    assert forn["tipo_pessoa"] == "J"

    sub = build_subcota_record(sample_item)
    assert sub is not None
    assert sub["num_subcota"] == 1

    pdf_dir = Path("/tmp")
    doc = build_documento_record(sample_item, pdf_dir)
    assert doc is not None
    assert doc["ide_documento"] == 7696122
    assert doc["ano"] == 2024

    desp = build_despesa_record(sample_item)
    assert desp is not None
    assert desp["valor_liquido"] == 1148.7
    assert desp["data_emissao"] == "2024-02-07"


def test_schema_file_exists_and_contains_core_elements() -> None:
    schema_path = Path("sql/init_schema.sql")
    if not schema_path.exists():
        schema_path = Path("masters/sql/init_schema.sql")
    assert schema_path.exists()

    content = schema_path.read_text(encoding="utf-8")
    # Verifica extensões
    assert "pg_trgm" in content
    assert "vector" in content

    # Verifica tabelas principais
    assert "CREATE TABLE IF NOT EXISTS deputado" in content
    assert "CREATE TABLE IF NOT EXISTS fornecedor" in content
    assert "CREATE TABLE IF NOT EXISTS documento" in content
    assert "CREATE TABLE IF NOT EXISTS despesa" in content
    assert "CREATE TABLE IF NOT EXISTS embedding_documento" in content
    assert "CREATE TABLE IF NOT EXISTS extracao_documento" in content
    assert "hnsw (vetor vector_cosine_ops)" in content
