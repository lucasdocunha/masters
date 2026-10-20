import pytest
from scripts.evaluate_extraction import (
    clean_digits,
    parse_date_to_iso,
    clean_doc_number,
    compute_text_similarity,
    evaluate_single_doc,
)


def test_clean_digits():
    assert clean_digits("085.324.290/0013-1") == "08532429000131"
    assert clean_digits("123-45") == "12345"
    assert clean_digits(None) == ""


def test_parse_date_to_iso():
    assert parse_date_to_iso("2024-02-07T00:00:00") == "2024-02-07"
    assert parse_date_to_iso("2024-02-07") == "2024-02-07"
    assert parse_date_to_iso("07/02/2024") == "2024-02-07"
    assert parse_date_to_iso("invalido") is None
    assert parse_date_to_iso(None) is None


def test_clean_doc_number():
    assert clean_doc_number("0001021") == "1021"
    assert clean_doc_number("1021") == "1021"
    assert clean_doc_number("NF-0045A") == "NF0045A"
    assert clean_doc_number(None) == ""


def test_compute_text_similarity():
    sim = compute_text_similarity("AMORETTO CAFES EXPRESSO LTDA", "Amoretto Cafés Expresso Ltda - ME")
    assert sim >= 0.75

    sim_diff = compute_text_similarity("POSTO SHELL", "HOTEL COPACABANA")
    assert sim_diff < 0.20


def test_evaluate_single_doc_match():
    expected = {
        "idDocumento": 1001,
        "ano": 2024,
        "fornecedor": "POSTO IPIRANGA LTDA",
        "cnpjCPF": "12.345.678/0001-90",
        "numero": "001234",
        "dataEmissao": "2024-05-10T00:00:00",
        "valorDocumento": 150.00,
        "valorLiquido": 150.00,
    }

    extracted = {
        "fornecedor_nome": "Posto Ipiranga",
        "fornecedor_cnpj_cpf": "12345678000190",
        "numero_documento": "1234",
        "data_emissao": "2024-05-10",
        "valor_total": 150.00,
    }

    res = evaluate_single_doc(expected, extracted)
    assert res["cnpj"]["match"] is True
    assert res["valor"]["match"] is True
    assert res["data"]["match"] is True
    assert res["numero"]["match"] is True
    assert res["fornecedor"]["match"] is True
    assert res["match_perfeito"] is True


def test_evaluate_single_doc_partial():
    expected = {
        "idDocumento": 1002,
        "ano": 2024,
        "fornecedor": "RESTAURANTE BOM SABOR",
        "cnpjCPF": "99.888.777/0001-11",
        "numero": "555",
        "dataEmissao": "2024-06-01",
        "valorDocumento": 200.00,
        "valorLiquido": 180.00,  # Glosa de 20 reais
    }

    # Modelo extraiu o valor bruto da nota (200.00), bate com valorDocumento
    extracted = {
        "fornecedor_nome": "Restaurante Bom Sabor",
        "fornecedor_cnpj_cpf": "99888777000111",
        "numero_documento": "555",
        "data_emissao": "2024-06-01",
        "valor_total": 200.00,
    }

    res = evaluate_single_doc(expected, extracted)
    assert res["valor"]["match"] is True
    assert res["match_perfeito"] is True
