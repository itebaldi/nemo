from nemo.vector_retrieval.inverted_list import InvertedListGeneratorConfig
from nemo.vector_retrieval.inverted_list import gen_inverted_list


def test_gen_inverted_list():
    config = InvertedListGeneratorConfig.create()

    df = gen_inverted_list(
        file_paths=config.read_paths,
        # output_path=config.write_path,
    )

    assert df is not None
