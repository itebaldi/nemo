import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from nemo.preprocessing.indexing import InvertedIndex
from nemo.preprocessing.tf_idf import VectorModel
from nemo.preprocessing.tf_idf import gen_vector_space_model


@pytest.fixture
def sample_inverted_index() -> InvertedIndex:
    """Inverted index for testing."""
    return InvertedIndex(
        {
            "APPLE": [1, 1],
            "BANANA": [1, 2],
            "ORANGE": [2],
        }
    )


def test_gen_vector_space_model__default_methods(
    sample_inverted_index: InvertedIndex,
):
    vector_model = gen_vector_space_model(inverted_index=sample_inverted_index)

    # TF (raw):
    #        1  2
    # APPLE  2  0
    # BANANA 1  1
    # ORANGE 0  1

    # TF (normalized): max_per_document for doc 1 is 2, for doc 2 is 1.
    #        1    2
    # APPLE  1.0  0.0
    # BANANA 0.5  1.0
    # ORANGE 0.0  1.0

    # IDF (standard): N=2
    # APPLE:  log(2/1) = log(2)
    # BANANA: log(2/2) = log(1) = 0
    # ORANGE: log(2/1) = log(2)

    # TF-IDF = TF * IDF
    idf_apple = idf_orange = 0.6931471805599453  # log(2)

    expected_data = {
        1: {"APPLE": idf_apple, "BANANA": 0.0, "ORANGE": 0.0},
        2: {"APPLE": 0.0, "BANANA": 0.0, "ORANGE": idf_orange},
    }
    expected_df = pd.DataFrame(expected_data)
    expected_df.index.name = "Word"

    assert isinstance(vector_model, VectorModel)
    assert_frame_equal(vector_model.root, expected_df, check_dtype=False)


def test_gen_vector_space_model__with_log_tf(sample_inverted_index: InvertedIndex):
    """Test VSM generation with log_tf_method."""
    vector_model = gen_vector_space_model(
        inverted_index=sample_inverted_index, tf_method=VectorModel.log_tf_method
    )

    # TF (log): 1 + log(tf)
    #        1              2
    # APPLE  1 + log(2)     0
    # BANANA 1 + log(1) = 1  1
    # ORANGE 0              1

    # IDF (standard):
    # APPLE:  log(2)
    # BANANA: 0
    # ORANGE: log(2)

    # TF-IDF = TF * IDF
    log_2 = idf_apple = idf_orange = 0.6931471805599453
    tf_apple_log = 1 + log_2  # 1 + log(2)

    expected_data = {
        1: {"APPLE": tf_apple_log * idf_apple, "BANANA": 0.0, "ORANGE": 0.0},
        2: {"APPLE": 0.0, "BANANA": 0.0, "ORANGE": 1 * idf_orange},
    }
    expected_df = pd.DataFrame(expected_data)
    expected_df.index.name = "Word"

    assert_frame_equal(vector_model.root, expected_df, check_dtype=False)


def test_gen_vector_space_model__with_invalid_tf_method_shape(
    sample_inverted_index: InvertedIndex,
):

    def invalid_tf_method(df: pd.DataFrame) -> pd.DataFrame:
        return df.head(1)  # Returns a DataFrame with fewer rows

    with pytest.raises(ValueError, match="same shape"):
        gen_vector_space_model(
            inverted_index=sample_inverted_index, tf_method=invalid_tf_method
        )


def test_gen_vector_space_model__with_invalid_idf_method_index(
    sample_inverted_index: InvertedIndex,
):

    def invalid_idf_method(df: pd.DataFrame, total_docs: int) -> pd.Series:
        return pd.Series([1.0], index=["WRONG_TERM"])

    with pytest.raises(
        ValueError, match="indexed by the term_frequency_matrix rows"
    ):
        gen_vector_space_model(
            inverted_index=sample_inverted_index, idf_method=invalid_idf_method
        )
