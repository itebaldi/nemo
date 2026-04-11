import pandas as pd
from toolz.functoolz import pipe

from inputs.stopwords import get_english_stopwords
from nemo.preprocessing.text import lowercase_text
from nemo.preprocessing.text import remove_text_punctuation
from nemo.preprocessing.text import remove_text_stopwords
from nemo.preprocessing.text import stem_text
from nemo.preprocessing.utils import transform_text_column


def test_transform_text_column():
    table = pipe(
        pd.DataFrame(
            {
                "sentence": [
                    "Wow, I loved this place!",
                    "This is another sentence.",
                ],
                "sentiment": [1, 0],
            }
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
    )

    expected_sentences = ["wow love place", "anoth sentenc"]
    assert table["sentence"].tolist() == expected_sentences

    assert set(table["sentiment"].unique()) == {0, 1}
