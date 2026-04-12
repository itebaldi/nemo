from nemo.vector_retrieval.vector_model import VectorModelConfig
from nemo.vector_retrieval.vector_model import gen_vector_model


def test_gen_vector_model():
    config = VectorModelConfig.create()

    df = gen_vector_model(
        read_path=config.read_path,
        # output_path=config.write_path,
    )

    assert df is not None
