from scripts.select_eval_sample import format_expected_record


def test_format_expected_record():
    raw_item = {
        "idDocumento": 7696122,
        "ano": "2024",
        "mes": "2",
        "fornecedor": " AMORETTO CAFES EXPRESSO LTDA ",
        "cnpjCPF": "085.324.290/0013-1 ",
        "numero": "1021",
        "dataEmissao": "2024-02-07T00:00:00",
        "valorDocumento": "1148.7",
        "valorLiquido": "1148.7",
        "valorGlosa": "0",
        "nomeParlamentar": "LID.GOV-CD",
        "descricao": "MANUTENÇÃO DE ESCRITÓRIO DE APOIO À ATIVIDADE PARLAMENTAR",
        "urlDocumento": "https://www.camara.leg.br/7696122.pdf"
    }

    formatted = format_expected_record(raw_item)
    assert formatted["idDocumento"] == 7696122
    assert formatted["ano"] == 2024
    assert formatted["mes"] == 2
    assert formatted["fornecedor"] == "AMORETTO CAFES EXPRESSO LTDA"
    assert formatted["cnpjCPF"] == "085.324.290/0013-1"
    assert formatted["numero"] == "1021"
    assert formatted["valorDocumento"] == 1148.7
    assert formatted["valorLiquido"] == 1148.7
    assert formatted["valorGlosa"] == 0.0
