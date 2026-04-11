from nemo.vector_retrieval.query_processor import QueryProcessorConfig
from nemo.vector_retrieval.query_processor import gen_processed_queries


def test_gen_processed_queries():
    config = QueryProcessorConfig.create()

    df = gen_processed_queries(
        file_path="inputs/vector_retrieval/CysticFibrosis2/cfquery.xml",
        # output_path="outputs/vector_retrieval/RESULT/consultas.csv",
    )

    assert df.shape == (2, 2)
    assert all(df.columns == ["QueryNumber", "QueryText"])
