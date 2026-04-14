import pytest

from nemo.files.csv import read_csv
from nemo.retrieval_assignment.search_engine import SearcherConfig
from nemo.retrieval_assignment.search_engine import gen_results
from nemo.vector_retrieval.tf_idf import VectorModel


@pytest.mark.slow
def test_gen_results():
    config = SearcherConfig.create()

    vector_model = VectorModel.dataframe_from_csv(config.model_path)

    queries = read_csv(
        config.queries_path,
        separator=";",
        dtypes={"QueryNumber": str, "QueryText": str},
    )

    df = gen_results(
        vector_model=vector_model,
        queries=queries,
        output_path=config.results_path,
    )

    assert df is not None
