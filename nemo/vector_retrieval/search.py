from ast import literal_eval
from typing import ClassVar

import pandas as pd
from pydantic import RootModel

from nemo.vector_retrieval.query import Query
from nemo.vector_retrieval.query import RankedDocument
from nemo.vector_retrieval.tf_idf import VectorModel


class SearchResults(RootModel[dict[str, list[RankedDocument]]]):
    query_number_column: ClassVar[str] = "QueryNumber"
    results_column: ClassVar[str] = "Results"

    @classmethod
    def required_columns(cls) -> set[str]:
        """
        Return the required DataFrame columns.

        Returns
        -------
        set[str]
            Required DataFrame columns.
        """
        return {cls.query_number_column, cls.results_column}

    @classmethod
    def validate_dataframe(cls, df: pd.DataFrame) -> None:
        """
        Validate whether a DataFrame has the expected schema.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate.

        Raises
        ------
        ValueError
            If the DataFrame does not contain the required columns.
        """
        missing_columns = cls.required_columns() - set(df.columns)
        if missing_columns:
            raise ValueError(
                f"DataFrame must contain columns {sorted(cls.required_columns())}. "
                f"Missing columns: {sorted(missing_columns)}."
            )

    @classmethod
    def _parse_ranked_documents(cls, value: object) -> list[RankedDocument]:
        """
        Parse ranked documents from a DataFrame cell.

        Parameters
        ----------
        value : object
            DataFrame cell containing ranked-document tuples.

        Returns
        -------
        list[RankedDocument]
            Parsed ranked documents.

        Raises
        ------
        ValueError
            If the value cannot be parsed as a list of
            ``(rank, document_id, score)`` tuples.
        """
        parsed_value = value if isinstance(value, list) else literal_eval(str(value))

        if not isinstance(parsed_value, list):
            raise ValueError(
                f"Expected a list of ranked documents, got "
                f"{type(parsed_value).__name__}."
            )

        ranked_documents: list[RankedDocument] = []

        for item in parsed_value:
            if not isinstance(item, (list, tuple)) or len(item) != 3:
                raise ValueError(
                    "Each ranked document must be a tuple or list with "
                    "(rank, document_id, score)."
                )

            rank, document_id, score = item

            ranked_documents.append(
                RankedDocument(
                    rank=int(rank),
                    document_id=int(document_id),
                    score=float(score),
                )
            )

        return ranked_documents

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "SearchResults":
        """
        Create search results from a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing ``QueryNumber`` and ``Results``.

        Returns
        -------
        SearchResults
            Parsed search results.
        """
        cls.validate_dataframe(df)

        results: dict[str, list[RankedDocument]] = {
            str(row[cls.query_number_column]): cls._parse_ranked_documents(
                row[cls.results_column]
            )
            for _, row in df.iterrows()
        }

        return cls(results)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the search results into a DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame containing ``QueryNumber`` and ``Results``.
        """
        rows = []

        for query_number, ranked_documents in self.root.items():
            results_tuple = [
                (
                    ranked_document.rank,
                    ranked_document.document_id,
                    ranked_document.score,
                )
                for ranked_document in ranked_documents
            ]

            rows.append(
                {
                    self.query_number_column: query_number,
                    self.results_column: results_tuple,
                }
            )

        return pd.DataFrame(rows)


def search(
    queries: list[Query],
    vector_model: VectorModel,
) -> SearchResults:
    results: dict[str, list[RankedDocument]] = {}

    for query in queries:
        ranked_documents = query.search(vector_model)
        results[query.query_id] = ranked_documents

    return SearchResults(results)
