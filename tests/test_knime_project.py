from pathlib import Path

from toolz.functoolz import pipe

from nemo.files.csv import read_csv
from nemo.preprocessing.dataframe import apply_stemming
from nemo.preprocessing.dataframe import create_bag_of_words_matrix
from nemo.preprocessing.dataframe import map_column_values
from nemo.preprocessing.dataframe import remove_punctuation
from nemo.preprocessing.dataframe import remove_stopwords
from nemo.preprocessing.dataframe import to_lowercase
from nemo.preprocessing.text import lowercase_text
from nemo.preprocessing.text import remove_text_punctuation
from nemo.preprocessing.text import remove_text_stopwords
from nemo.preprocessing.text import stem_text
from nemo.preprocessing.utils import transform_text_column
from nemo.stopwords import get_english_stopwords


def test_knime_project():

    table = pipe(
        read_csv(
            file_path=Path("inputs/yelp_labelled.txt"),
            separator="\t",
            header=None,
            column_names=["sentence", "sentiment"],
            dtypes={"sentence": "string", "sentiment": "int64"},
        ),
        map_column_values(
            column="sentiment",
            mapping={0: "negative", 1: "positive"},
        ),
        to_lowercase(column="sentence"),
        remove_punctuation(column="sentence"),
        remove_stopwords(column="sentence", stop_words=get_english_stopwords()),
        apply_stemming(column="sentence", language="english"),
        create_bag_of_words_matrix(
            column="sentence",
            ngram_range=(1, 1),
            preserve_columns=["sentence", "sentiment"],
        ),
    )

    expected_sentence = "wow love place"
    assert table["sentence"].iloc[0] == expected_sentence
    assert set(table["sentiment"].unique()) == {"negative", "positive"}


def test_knime_project__transform_text_column():
    table = pipe(
        read_csv(
            file_path=Path("inputs/yelp_labelled.txt"),
            separator="\t",
            header=None,
            column_names=["sentence", "sentiment"],
            dtypes={"sentence": "string", "sentiment": "int64"},
        ),
        transform_text_column(
            column="sentence",
            transforms=[
                lowercase_text,
                remove_text_punctuation,
                remove_text_stopwords(stop_words=get_english_stopwords()),
                stem_text(language="english"),
            ],
        ),
        create_bag_of_words_matrix(
            column="sentence",
            ngram_range=(2, 2),
            preserve_columns=["sentence", "sentiment"],
        ),
    )

    expected_sentence = "wow love place"
    assert table["sentence"].iloc[0] == expected_sentence

    assert set(table["sentiment"].unique()) == {0, 1}
