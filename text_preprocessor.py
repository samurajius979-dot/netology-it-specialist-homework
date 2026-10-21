"""Кастомный sklearn-трансформер предобработки текстовых обращений.

Используется и в ноутбуке ``03_ML_pipeline_solution.ipynb``, и в DAG Airflow
(``airflow_dag.py``) — один и тот же код обработки в обоих местах.
"""

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

HTML_TAG_RE = re.compile(r"<[^>]+>")
PUNCT_RE = re.compile(r"[^\w\s]")
WS_RE = re.compile(r"\s+")


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """Трансформер предобработки текстовых обращений.

    Параметры
    ----------
    lowercase : bool
        Привести текст к нижнему регистру.
    remove_stopwords : bool
        Удалить английские стоп-слова.
    lemmatize : bool
        Лемматизировать слова (WordNet).
    min_token_length : int
        Отбрасывать токены короче указанного количества символов.
    """

    def __init__(self, lowercase=True, remove_stopwords=True, lemmatize=False,
                 min_token_length=2, random_state=42):
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.min_token_length = min_token_length
        self.random_state = random_state

    def _stopwords(self):
        from nltk.corpus import stopwords

        try:
            return set(stopwords.words("english"))
        except LookupError:  # pragma: no cover - офлайн-окружение без данных nltk
            import nltk

            nltk.download("stopwords", quiet=True)
            return set(stopwords.words("english"))

    def _lemmatizer(self):
        try:
            from nltk.stem import WordNetLemmatizer

            return WordNetLemmatizer().lemmatize
        except LookupError:  # pragma: no cover
            import nltk

            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
            from nltk.stem import WordNetLemmatizer

            return WordNetLemmatizer().lemmatize

    def fit(self, X, y=None):
        self._stopwords_cache = self._stopwords()
        self._lemmatize_func = self._lemmatizer() if self.lemmatize else None
        self.feature_names_in_ = np.array(["Document"])
        return self

    def transform(self, X):
        if isinstance(X, pd.Series):
            texts = X.astype(str).tolist()
        elif isinstance(X, pd.DataFrame):
            texts = X["Document"].astype(str).tolist()
        else:
            texts = [str(t) for t in X]

        stop = self._stopwords_cache
        lem = self._lemmatize_func

        def clean(text):
            text = HTML_TAG_RE.sub(" ", text)
            text = PUNCT_RE.sub(" ", text)
            if self.lowercase:
                text = text.lower()
            tokens = [t for t in WS_RE.split(text) if len(t) >= self.min_token_length]
            if self.remove_stopwords:
                tokens = [t for t in tokens if t not in stop]
            if lem is not None:
                tokens = [lem(t) for t in tokens]
            return " ".join(tokens)

        return pd.Series([clean(t) for t in texts], index=getattr(X, "index", None))

    def get_feature_names_out(self, input_features=None):
        return np.array(["Document_clean"])
