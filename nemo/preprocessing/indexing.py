from collections import defaultdict

from pydantic import BaseModel
from pydantic import ConfigDict
from toolz.functoolz import pipe

from inputs.stopwords import get_stop_words_for_text
from nemo.preprocessing.text import generate_ngrams
from nemo.preprocessing.text import normalize_text_whitespace
from nemo.preprocessing.text import remove_text_accents
from nemo.preprocessing.text import remove_text_punctuation
from nemo.preprocessing.text import remove_text_stopwords
from nemo.preprocessing.text import uppercase_text


class Document(BaseModel):
    """
    Generic document representation for indexing.

    Attributes
    ----------
    document_id : int
        Unique document identifier.
    text : str
        Raw document text.
    """

    model_config = ConfigDict(frozen=True)

    document_id: int
    text: str


def gen_inverted_index(
    documents: list[Document],
) -> dict[str, list[int]]:
    """
    Build an inverted index from a list of documents.

    Parameters
    ----------
    documents : list[Document]
        Documents to index.

    Returns
    -------
    dict[str, list[int]]
        Mapping from term to list of document identifiers.
        Document identifiers are repeated when the same term appears
        multiple times in the same document.
    """
    inverted_index: dict[str, list[int]] = defaultdict(list)

    for document in documents:
        terms = _tokenize_text(document.text)

        for term in terms:
            if len(term) < 2:
                continue

            if not term.isalpha():
                continue

            inverted_index[term].append(document.document_id)

    return dict(sorted(inverted_index.items()))


def _tokenize_text(text: str) -> list[str]:
    stop_words = get_stop_words_for_text(text)

    return pipe(
        remove_text_accents(text),
        remove_text_punctuation,
        uppercase_text,
        normalize_text_whitespace,
        remove_text_stopwords(stop_words=stop_words),
        generate_ngrams(n=1),
    )  # type: ignore
