import pytest

from nemo.files.xml import read_xml
from nemo.retrieval_assignment.query_processor import QueryProcessorConfig
from nemo.retrieval_assignment.query_processor import gen_expected_docs
from nemo.retrieval_assignment.query_processor import gen_processed_queries


@pytest.mark.slow
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
