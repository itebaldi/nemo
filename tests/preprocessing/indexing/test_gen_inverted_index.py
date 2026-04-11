from nemo.preprocessing.indexing import Document
from nemo.preprocessing.indexing import gen_inverted_index


def test_gen_inverted_index_simple():
    documents = [
        Document(document_id=1, text="The quick brown fox"),
        Document(document_id=2, text="jumps over the lazy dog"),
    ]

    expected_index = {
        "BROWN": [1],
        "DOG": [2],
        "FOX": [1],
        "JUMPS": [2],
        "LAZY": [2],
        "QUICK": [1],
    }

    inverted_index = gen_inverted_index(documents)
    assert inverted_index == expected_index


def test_gen_inverted_index_with_repeated_words():
    documents = [
        Document(document_id=1, text="apple banana apple"),
        Document(document_id=2, text="banana orange"),
    ]

    expected_index = {
        "APPLE": [1, 1],
        "BANANA": [1, 2],
        "ORANGE": [2],
    }

    inverted_index = gen_inverted_index(documents)
    assert inverted_index == expected_index


def test_gen_inverted_index_with_processing():
    documents = [
        Document(document_id=1, text="Olá, este é um teste! Teste 1."),
        Document(document_id=2, text="Outro teste com acentuação."),
    ]

    expected_index = {
        "ACENTUACAO": [2],
        "OLA": [1],
        "OUTRO": [2],
        "TESTE": [1, 1, 2],
    }

    inverted_index = gen_inverted_index(documents)
    assert inverted_index == expected_index
