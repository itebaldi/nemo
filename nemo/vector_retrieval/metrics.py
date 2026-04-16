from pathlib import Path

from pydantic import BaseModel
from pydantic import ConfigDict

from nemo.vector_retrieval.query import RankedDocument
from nemo.vector_retrieval.search import SearchResults


class Relevance(BaseModel):
    model_config = ConfigDict(frozen=True)

    query_per_documents: dict[str, set[int]]


class QueryMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    query_id: str
    relevant_docs: int
    avg_precision: float
    coverage: float
    recall: dict[str, float]
    precision: dict[str, float]


class MetricsSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    queries_evaluated: int
    mean_precision: dict[str, float]
    mean_recall: dict[str, float]
    mean_avg_precision: float
    mean_coverage: float
    min_coverage: float
    max_coverage: float
    full_coverage_count: int

    def to_json(
        self, file_path: str | Path | None = None, indent: int = 2
    ) -> str | Path:
        """
        Convert the metrics summary to JSON or write it to a JSON file.

        Parameters
        ----------
        file_path : str | Path | None, default=None
            Output file path. If provided, the JSON content is written to this path.
            If ``None``, the JSON string is returned.
        indent : int, default=2
            Indentation level for JSON formatting.

        Returns
        -------
        str | Path
            JSON string if ``file_path`` is ``None``, otherwise the written file path.
        """
        json_content = self.model_dump_json(indent=indent)

        if file_path is None:
            return json_content

        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json_content, encoding="utf-8")
        return path


class Metrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    queries_metrics: list[QueryMetrics]

    def summary(self) -> MetricsSummary:
        """
        Generate a summary from per-query metrics.

        Returns
        -------
        MetricsSummary
            Summary computed from the stored query metrics.
        """
        n = len(self.queries_metrics)

        if n == 0:
            return MetricsSummary(
                queries_evaluated=0,
                mean_precision={},
                mean_recall={},
                mean_avg_precision=0.0,
                mean_coverage=0.0,
                min_coverage=0.0,
                max_coverage=0.0,
                full_coverage_count=0,
            )

        precision_keys = sorted(
            {
                key
                for query_metrics in self.queries_metrics
                for key in query_metrics.precision
            },
            key=int,
        )
        recall_keys = sorted(
            {
                key
                for query_metrics in self.queries_metrics
                for key in query_metrics.recall
            },
            key=int,
        )

        mean_precision = {
            key: sum(
                query_metrics.precision.get(key, 0.0)
                for query_metrics in self.queries_metrics
            )
            / n
            for key in precision_keys
        }

        mean_recall = {
            key: sum(
                query_metrics.recall.get(key, 0.0)
                for query_metrics in self.queries_metrics
            )
            / n
            for key in recall_keys
        }

        avg_precision_values = [
            query_metrics.avg_precision for query_metrics in self.queries_metrics
        ]
        coverage_values = [
            query_metrics.coverage for query_metrics in self.queries_metrics
        ]

        return MetricsSummary(
            queries_evaluated=n,
            mean_precision=mean_precision,
            mean_recall=mean_recall,
            mean_avg_precision=sum(avg_precision_values) / n,
            mean_coverage=sum(coverage_values) / n,
            min_coverage=min(coverage_values),
            max_coverage=max(coverage_values),
            full_coverage_count=sum(value == 1.0 for value in coverage_values),
        )


def compute_metrics(
    relevance: Relevance,
    search_results: SearchResults,
) -> Metrics:

    metrics: list[QueryMetrics] = []

    for query_id, relevant_documents in relevance.query_per_documents.items():
        ranked_result = search_results.root.get(query_id)

        if ranked_result is None:
            continue

        metrics.append(
            _compute_query_metrics(
                query_id=query_id,
                relevant_documents=relevant_documents,
                ranked_result=ranked_result,
            )
        )

    return Metrics(queries_metrics=metrics)


def _compute_query_metrics(
    query_id: str,
    relevant_documents: set[int],
    ranked_result: list[RankedDocument],
) -> QueryMetrics:

    ranked_documents = [ranked.document_id for ranked in ranked_result]

    return QueryMetrics(
        query_id=query_id,
        coverage=coverage(ranked_documents, relevant_documents),
        avg_precision=average_precision(ranked_documents, relevant_documents),
        relevant_docs=len(relevant_documents),
        precision={
            "10": precision_at_k(ranked_documents, relevant_documents, k=10),
            "20": precision_at_k(ranked_documents, relevant_documents, k=20),
        },
        recall={"10": recall_at_k(ranked_documents, relevant_documents, k=10)},
    )


def precision_at_k(
    ranked_documents: list[int],
    relevant_documents: set[int],
    k: int,
) -> float:
    """
    Compute precision at rank k.

    Parameters
    ----------
    ranked_documents : list[int]
        Ranked list of retrieved document identifiers.
    relevant_documents : set[int]
        Set of relevant document identifiers.
    k : int
        Rank cutoff.

    Returns
    -------
    float
        Precision at rank k.
    """
    if k <= 0:
        raise ValueError("k must be greater than 0.")

    top_k = ranked_documents[:k]
    relevant_hits = sum(document_id in relevant_documents for document_id in top_k)

    return relevant_hits / k


def recall_at_k(
    ranked_documents: list[int],
    relevant_documents: set[int],
    k: int,
) -> float:
    """
    Compute recall at rank k.

    Parameters
    ----------
    ranked_documents : list[int]
        Ranked list of retrieved document identifiers.
    relevant_documents : set[int]
        Set of relevant document identifiers.
    k : int
        Rank cutoff.

    Returns
    -------
    float
        Recall at rank k.
    """
    if not relevant_documents:
        return 0.0

    top_k = ranked_documents[:k]
    relevant_hits = sum(document_id in relevant_documents for document_id in top_k)

    return relevant_hits / len(relevant_documents)


def average_precision(
    ranked_documents: list[int],
    relevant_documents: set[int],
) -> float:
    """
    Compute average precision for a ranked result list.

    Parameters
    ----------
    ranked_documents : list[int]
        Ranked list of retrieved document identifiers.
    relevant_documents : set[int]
        Set of relevant document identifiers.

    Returns
    -------
    float
        Average precision.
    """
    if not relevant_documents:
        return 0.0

    relevant_hits = 0
    precisions: list[float] = []

    for rank, document_id in enumerate(ranked_documents, start=1):
        if document_id in relevant_documents:
            relevant_hits += 1
            precisions.append(relevant_hits / rank)

    return sum(precisions) / len(relevant_documents)


def coverage(
    ranked_documents: list[int],
    relevant_documents: set[int],
) -> float:
    """
    Compute coverage of relevant documents in the retrieved ranking.

    Parameters
    ----------
    ranked_documents : list[int]
        Ranked list of retrieved document identifiers.
    relevant_documents : set[int]
        Set of relevant document identifiers.

    Returns
    -------
    float
        Fraction of relevant documents that appear anywhere in the ranking.
    """
    if not relevant_documents:
        return 0.0

    retrieved_relevant_documents = set(ranked_documents).intersection(
        relevant_documents
    )
    return len(retrieved_relevant_documents) / len(relevant_documents)
