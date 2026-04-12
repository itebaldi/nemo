from ast import literal_eval
from collections import defaultdict
from pathlib import Path
from typing import ClassVar

import pandas as pd
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import RootModel
from toolz.functoolz import pipe

from inputs.stopwords import get_stop_words_for_text
from nemo.importing import read_csv
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


class InvertedIndexMatrix(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    root: pd.DataFrame


class InvertedIndex(RootModel[dict[str, list[int]]]):
    """
    Inverted index representation.

    The root object maps each term to a list of document identifiers.
    Document identifiers are repeated when the same term appears
    multiple times in the same document.

    Examples
    --------
    For a collection of two documents:

    - Document 1 (ID 1): "apple banana apple"
    - Document 2 (ID 2): "banana orange"

    The inverted index would be represented as:

    .. code-block:: python

        {
            "APPLE": [1, 1],
            "BANANA": [1, 2],
            "ORANGE": [2],
        }

    """

    def vocabulary(self) -> list[str]:
        """
        Return the sorted vocabulary. Sorted list of indexed terms.
        """
        return sorted(self.root.keys())

    word_column: ClassVar[str] = "Word"
    documents_column: ClassVar[str] = "Documents"

    @classmethod
    def required_columns(cls) -> set[str]:
        """
        Return the required DataFrame columns.

        Returns
        -------
        set[str]
            Required DataFrame columns.
        """
        return {cls.word_column, cls.documents_column}

    @classmethod
    def validate_dataframe(cls, df: pd.DataFrame) -> None:
        """
        Validate whether a DataFrame has the expected schema.
        """
        missing_columns = cls.required_columns() - set(df.columns)
        if missing_columns:
            raise ValueError(
                f"DataFrame must contain columns {sorted(cls.required_columns())}. "
                f"Missing columns: {sorted(missing_columns)}."
            )

    @classmethod
    def _parse_document_ids(cls, value: object) -> list[int]:
        """
        Parse the document identifiers from a DataFrame cell.
        """
        if isinstance(value, list):
            return value

        parsed_value = literal_eval(str(value))
        if not isinstance(parsed_value, list):
            raise ValueError(
                f"Expected a list of document ids in column '{cls.documents_column}', "
                f"got {type(parsed_value).__name__}."
            )

        return [int(document_id) for document_id in parsed_value]

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "InvertedIndex":
        """
        Create an inverted index from a DataFrame.
        """
        cls.validate_dataframe(df)

        term_document_ids: dict[str, list[int]] = {
            str(row[cls.word_column]): cls._parse_document_ids(
                row[cls.documents_column]
            )
            for _, row in df.iterrows()
        }

        return cls(term_document_ids)

    def to_dataframe(self) -> InvertedIndexMatrix:
        """
        Convert the inverted index into a DataFrame.
        """
        rows = [
            {
                self.word_column: term,
                self.documents_column: document_ids,
            }
            for term, document_ids in self.root.items()
        ]
        return InvertedIndexMatrix(root=pd.DataFrame(rows))

    @classmethod
    def dataframe_from_csv(
        cls,
        file_path: str | Path,
        separator: str = ";",
    ) -> InvertedIndexMatrix:
        """
        Read and validate an inverted-index DataFrame from a CSV file.
        """
        df = read_csv(file_path=file_path, separator=separator)
        cls.validate_dataframe(df)
        return InvertedIndexMatrix(root=df)


def gen_inverted_index(
    documents: list[Document],
) -> InvertedIndex:
    """
    Build an inverted index from a list of documents.

    Parameters
    ----------
    documents : list[Document]
        Documents to index.

    Returns
    -------
    InvertedIndex
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

    return InvertedIndex(dict(sorted(inverted_index.items())))


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
