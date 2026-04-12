import math
from collections import Counter
from pathlib import Path
from typing import Callable
from typing import ClassVar

import pandas as pd
from pydantic import BaseModel
from pydantic import ConfigDict
from toolz.functoolz import pipe

from nemo.importing import read_csv
from nemo.preprocessing.indexing import InvertedIndex
from nemo.preprocessing.indexing import InvertedIndexMatrix
from nemo.tools import curry


class VectorModel(BaseModel):
    """
    Vector space model representation.

    Term-document matrix with TF-IDF weights, where rows represent terms and
    columns represent document identifiers.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    root: pd.DataFrame

    word_column: ClassVar[str] = InvertedIndex.word_column

    @classmethod
    def validate_dataframe(cls, df: pd.DataFrame) -> None:
        """
        Validate whether a DataFrame has the expected schema for a vector model.

        Expected format
        ---------------
        - One column called "Word"
        - At least one document column
        - Document column names must be valid integers
        - Document values must be numeric
        """
        if cls.word_column not in df.columns:
            raise ValueError(
                f"DataFrame must contain the '{cls.word_column}' column."
            )

        document_columns = [col for col in df.columns if col != cls.word_column]

        if not document_columns:
            raise ValueError(
                "DataFrame must contain at least one document column besides 'Word'."
            )

        duplicated_columns = df.columns[df.columns.duplicated()].tolist()
        if duplicated_columns:
            raise ValueError(
                f"DataFrame contains duplicated columns: {duplicated_columns}."
            )

        invalid_document_columns: list[object] = []
        for column in document_columns:
            try:
                int(column)
            except (TypeError, ValueError):
                invalid_document_columns.append(column)

        if invalid_document_columns:
            raise ValueError(
                "All document columns must be integer document ids. "
                f"Invalid columns: {invalid_document_columns}."
            )

        numeric_values = df[document_columns].apply(pd.to_numeric, errors="coerce")
        invalid_mask = numeric_values.isna() & df[document_columns].notna()

        if invalid_mask.any().any():
            invalid_positions = [
                (row_index, column_name)
                for row_index, row in invalid_mask.iterrows()
                for column_name, is_invalid in row.items()
                if is_invalid
            ]
            raise ValueError(
                "All TF-IDF values must be numeric. "
                f"Invalid cells found at: {invalid_positions[:10]}."
            )

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "VectorModel":
        """
        Create a validated VectorModel from a DataFrame.
        """
        cls.validate_dataframe(df)

        parsed_df = df.copy()
        parsed_df[cls.word_column] = parsed_df[cls.word_column].map(str)

        document_columns = [
            col for col in parsed_df.columns if col != cls.word_column
        ]

        renamed_columns = {column: int(column) for column in document_columns}
        parsed_df = parsed_df.rename(columns=renamed_columns)

        parsed_df[list(renamed_columns.values())] = parsed_df[
            list(renamed_columns.values())
        ].apply(pd.to_numeric, errors="raise")

        parsed_df = parsed_df.set_index(cls.word_column)
        parsed_df.index.name = cls.word_column

        return cls(root=parsed_df)

    @classmethod
    def dataframe_from_csv(
        cls,
        file_path: str | Path,
        separator: str = ";",
    ) -> "VectorModel":
        """
        Read and validate a vector-model DataFrame from a CSV file.
        """
        df = read_csv(file_path, separator=separator)
        return cls.from_dataframe(df)

    @staticmethod
    def normalized_tf_method(term_frequency_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize term frequencies within each document.

        Each value is divided by the highest term frequency in the same document,
        so the weights stay between 0 and 1.

        Good when you want to compare term importance across documents without
        giving too much advantage to documents with larger raw counts.
        """
        max_per_document = term_frequency_matrix.max(axis=0)
        return term_frequency_matrix.div(max_per_document, axis=1).fillna(0.0)

    @staticmethod
    def fractional_tf_method(
        term_frequency_matrix: pd.DataFrame,
        k: float = 1.0,
    ) -> pd.DataFrame:
        """
        Reduce the impact of repeated term occurrences.

        Transforms each frequency using tf / (tf + k), making the growth slower
        as the term appears more times.

        Good when repeated occurrences should still matter, but not too much.
        """
        return (term_frequency_matrix / (term_frequency_matrix + k)).astype(float)

    @staticmethod
    def log_tf_method(term_frequency_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Apply logarithmic scaling to term frequencies.

        A term that appears more times still gets a higher weight, but the increase
        becomes smaller and smaller.

        Good when you want to reduce the effect of very frequent terms while still
        rewarding repetition.
        """
        return term_frequency_matrix.map(
            lambda value: 1 + math.log(value) if value > 0 else 0.0
        ).astype(float)

    @staticmethod
    def pivoted_tf_method(term_frequency_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Adjust term frequencies based on document length.

        Terms in longer documents are scaled down, while terms in shorter documents
        are scaled up relative to the average document length.

        Good when document size varies a lot and you want to avoid favoring longer
        documents just because they contain more words.
        """
        document_lengths = term_frequency_matrix.sum(axis=0).astype(float)
        average_document_length = document_lengths.mean()

        k_d = document_lengths / average_document_length
        return term_frequency_matrix.div(k_d, axis=1).fillna(0.0).astype(float)

    @staticmethod
    def lifted_tf_method(
        term_frequency_matrix: pd.DataFrame,
        a: float = 0.4,
    ) -> pd.DataFrame:
        """
        Increase normalized term frequencies by a fixed minimum amount.

        First normalizes each document to values between 0 and 1, then lifts the
        result using the parameter a.

        Good when you do not want low frequencies to become too small after
        normalization.
        """
        if not 0 <= a <= 1:
            raise ValueError("a must be between 0 and 1.")

        normalized_tf = term_frequency_matrix.div(
            term_frequency_matrix.max(axis=0), axis=1
        ).fillna(0.0)

        return (a + (1 - a) * normalized_tf).astype(float)

    @staticmethod
    def standard_idf_method(
        term_frequency_matrix: pd.DataFrame,
        total_documents: int,
    ) -> pd.Series:
        """
        Measure how rare each term is across the document collection.

        Terms that appear in many documents receive lower values, while terms
        that appear in fewer documents receive higher values.

        This method uses the formula log(N / n), where:
        - N is the total number of documents
        - n is the number of documents that contain the term

        Use this when you want common terms across the collection to have less
        importance than rarer terms.
        """
        document_frequencies = (term_frequency_matrix > 0).sum(axis=1)
        document_frequencies = document_frequencies.astype(float)
        return (total_documents / document_frequencies).map(math.log)


def gen_vector_space_model(
    inverted_index: InvertedIndex | InvertedIndexMatrix,
    tf_method: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    idf_method: Callable[[pd.DataFrame, int], pd.Series] | None = None,
) -> VectorModel:
    """
    Generate a TF-IDF vector space model from an inverted index.

    Parameters
    ----------
    inverted_index : InvertedIndex | InvertedIndexMatrix
        Inverted index representation, either as a dictionary-based model
        or as a DataFrame-based matrix.

    Returns
    -------
    VectorModel
        TF-IDF matrix with terms as rows and document identifiers as columns.
    """
    if isinstance(inverted_index, InvertedIndex):
        inverted_index = inverted_index.to_dataframe()

    return VectorModel(
        root=pipe(
            inverted_index,
            _gen_term_frequency_matrix,
            _gen_tf_idf_matrix(tf_method=tf_method, idf_method=idf_method),
        )  # type: ignore
    )


def _gen_term_frequency_matrix(
    inverted_index: InvertedIndexMatrix,
) -> pd.DataFrame:
    df = inverted_index.root

    term_frequencies = {
        str(row[InvertedIndex.word_column]): dict(
            Counter(row[InvertedIndex.documents_column])
        )
        for _, row in df.iterrows()
    }

    term_frequency_matrix = (
        pd.DataFrame.from_dict(term_frequencies, orient="index")
        .sort_index()
        .fillna(0)
        .astype(int)
    )
    term_frequency_matrix.index.name = InvertedIndex.word_column

    return term_frequency_matrix


@curry
def _gen_tf_idf_matrix(
    term_frequency_matrix: pd.DataFrame,
    tf_method: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    idf_method: Callable[[pd.DataFrame, int], pd.Series] | None = None,
) -> pd.DataFrame:
    """
    Generate a TF-IDF matrix from a term-frequency matrix.

    Parameters
    ----------
    term_frequency_matrix : pd.DataFrame
        Term-frequency matrix with terms as rows and document identifiers
        as columns.
    tf_method : Callable[[pd.DataFrame], pd.DataFrame] | None, default=None
        Callable used to transform the term-frequency matrix before
        applying inverse document frequency weighting. If ``None``,
        normalized term frequency is used.
    idf_method : Callable[[pd.Series, int], pd.Series] | None, default=None
        Callable used to generate inverse document frequency values.
        If ``None``, standard IDF is used.

    Returns
    -------
    pd.DataFrame
        TF-IDF matrix with terms as rows and document identifiers
        as columns.
    """

    tf_matrix = (
        VectorModel.normalized_tf_method(term_frequency_matrix)
        if tf_method is None
        else tf_method(term_frequency_matrix)
    )

    if tf_matrix.shape != term_frequency_matrix.shape:
        raise ValueError(
            "tf_method must return a DataFrame with the same shape as term_frequency_matrix."
        )

    total_documents = len(tf_matrix.columns)

    # inverse document frequencies for each term.
    idf = (
        VectorModel.standard_idf_method(term_frequency_matrix, total_documents)
        if idf_method is None
        else idf_method(term_frequency_matrix, total_documents)
    )

    if not idf.index.equals(term_frequency_matrix.index):
        raise ValueError(
            "idf_method must return a Series indexed by the term_frequency_matrix rows."
        )

    return tf_matrix.mul(idf, axis=0)
