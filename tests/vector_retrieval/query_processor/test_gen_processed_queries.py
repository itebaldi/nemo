from nemo.importing import read_xml
from nemo.vector_retrieval.query_processor import QueryProcessorConfig
from nemo.vector_retrieval.query_processor import gen_expected_docs
from nemo.vector_retrieval.query_processor import gen_processed_queries


def test_gen_processed_queries():
    config = QueryProcessorConfig.create()

    root = read_xml(config.read_path)

    df = gen_processed_queries(
        xml_root=root,
        # output_path=config.queries_output_path,
    )

    assert all(df.columns == ["QueryNumber", "QueryText"])


def test_gen_expected_docs():
    config = QueryProcessorConfig.create()
    root = read_xml(config.read_path)

    df = gen_expected_docs(
        xml_root=root,
        # output_path=config.expected_output_path,
    )

    assert all(df.columns == ["QueryNumber", "DocNumber", "DocVotes"])
