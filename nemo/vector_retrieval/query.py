import pandas as pd
from pydantic import BaseModel
from pydantic import ConfigDict

from nemo.vector_retrieval.indexing import tokenize_text
from nemo.vector_retrieval.tf_idf import VectorModel


class RankedDocument(BaseModel):
    """
    Ranked document for a query result.

    Attributes
    ----------
    rank : int
        Rank position in the result list.
    document_id : int
        Document identifier.
    score : float
        Similarity score.
    """

    model_config = ConfigDict(frozen=True)

    rank: int
    document_id: int
    score: float


class Query(BaseModel):
    """
    Generic query representation for vector-space retrieval.

    Attributes
    ----------
    query_id : str
        Unique query identifier.
    text : str
        Query text.
    """

    model_config = ConfigDict(frozen=True)

    query_id: str
    text: str

    def search(self, vector_model: VectorModel) -> list[RankedDocument]:
        """
        Search a vector model using a query.

        Parameters
        ----------
        query : Query
            Query to search.
        vector_model : pd.DataFrame
            Term-document matrix.

        Returns
        -------
        list[RankedDocument]
            Ranked retrieval results.
        """
        query_vector = _gen_query_vector(
            query=self,
            vocabulary=vector_model.root.index,
        )

        return _rank_documents(
            query_vector=query_vector,
            vector_model=vector_model,
        )


def _gen_query_vector(
    query: Query,
    vocabulary: pd.Index,
) -> pd.Series:
    """
    Generate a query vector with unit term weights.

    Uses the same term tokenization as document indexing.

    Parameters
    ----------
    query : Query
        Query to vectorize.
    vocabulary : pd.Index
        Vocabulary of the vector model.

    Returns
    -------
    pd.Series
        Query vector indexed by vocabulary terms.
    """
    query_vector = pd.Series(0.0, index=vocabulary)
    query_terms = tokenize_text(query.text)

    for term in query_terms:
        if term in query_vector.index:
            query_vector.loc[term] = 1.0

    return query_vector


def _cosine_similarity(
    left_vector: pd.Series,
    right_vector: pd.Series,
) -> float:
    """
    Compute cosine similarity between two vectors.

    Parameters
    ----------
    left_vector : pd.Series
        Left vector.
    right_vector : pd.Series
        Right vector.

    Returns
    -------
    float
        Cosine similarity score.
    """
    numerator = float((left_vector * right_vector).sum())

    left_norm = float((left_vector**2).sum() ** 0.5)
    right_norm = float((right_vector**2).sum() ** 0.5)

    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0

    return numerator / (left_norm * right_norm)


def _rank_documents(
    query_vector: pd.Series,
    vector_model: VectorModel,
) -> list[RankedDocument]:
    """
    Rank documents against a query vector.

    Parameters
    ----------
    query_vector : pd.Series
        Query vector indexed by model vocabulary.
    vector_model : VectorModel
        Term-document matrix.

    Returns
    -------
    list[RankedDocument]
        Ranked documents sorted by descending similarity.
    """
    scored_documents: list[tuple[int, float]] = []

    for document_id in vector_model.root.columns:
        document_vector = vector_model.root[document_id]
        score = _cosine_similarity(query_vector, document_vector)
        scored_documents.append((int(document_id), score))

    scored_documents.sort(key=lambda item: item[1], reverse=True)

    return [
        RankedDocument(rank=position, document_id=document_id, score=score)
        for position, (document_id, score) in enumerate(scored_documents, start=1)
    ]
