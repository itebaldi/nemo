from nemo.vector_retrieval.search_engine import gen_results


def test_gen_results():
    df = gen_results(
        write_output=True,
    )

    assert df is not None
