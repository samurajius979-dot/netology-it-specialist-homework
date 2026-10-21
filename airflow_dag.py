"""DAG оркестрации ML-pipeline классификации обращений в поддержку.

Файл предназначен для каталога ``dags/`` окружения Apache Airflow.
Весь обучающий код лежит в одном объекте ``sklearn.pipeline.Pipeline``
(``TextPreprocessor -> TfidfVectorizer -> LogisticRegression``), поэтому оркестрация
сводится к последовательности шагов: extract -> validate -> preprocess -> train ->
evaluate -> log_to_mlflow -> register_model.
"""

from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "ml-engineer",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

DATA_RAW = Path("/opt/airflow/data/raw/all_tickets_processed_improved_v3.csv")
DATA_PROCESSED = Path("/opt/airflow/data/processed")
ARTIFACTS = Path("/opt/airflow/data/artifacts")
MLFLOW_TRACKING_URI = "http://mlflow:5000"


def extract(**_):
    """Скачать/скопировать сырой датасет обращений."""
    DATA_RAW.parent.mkdir(parents=True, exist_ok=True)
    if not DATA_RAW.exists():
        raise FileNotFoundError(DATA_RAW)
    return str(DATA_RAW)


def validate(**_):
    """Проверить схему, наличие пропусков и распределение классов."""
    import pandas as pd

    df = pd.read_csv(DATA_RAW)
    assert {"Document", "Topic_group"}.issubset(df.columns), "Нет обязательных колонок"
    assert df.notna().all().all(), "В датасете есть пропуски"
    return df["Topic_group"].value_counts().to_dict()


def preprocess(**_):
    """Сохранить очищенные тексты (шаг внутри обучающего Pipeline)."""
    import pandas as pd
    import sys

    sys.path.append("/opt/airflow/dags")
    from text_preprocessor import TextPreprocessor

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_RAW)
    prep = TextPreprocessor().fit(df[["Document"]])
    out = df[["Document"]].copy()
    out["Document_clean"] = prep.transform(df[["Document"]])
    out["Topic_group"] = df["Topic_group"]
    path = DATA_PROCESSED / "train_ready.csv"
    out.to_csv(path, index=False)
    return str(path)


def train(**_):
    """Обучить пайплайн и сохранить его на диск."""
    import joblib
    import pandas as pd
    import sys
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline

    sys.path.append("/opt/airflow/dags")
    from text_preprocessor import TextPreprocessor

    df = pd.read_csv(DATA_PROCESSED / "train_ready.csv")
    X_train, _, y_train, _ = train_test_split(
        df[["Document_clean"]],
        df["Topic_group"],
        test_size=0.2,
        stratify=df["Topic_group"],
        random_state=42,
    )
    pipe = Pipeline([
        ("prep", TextPreprocessor()),
        ("tfidf", TfidfVectorizer(ngram_range=(1, 1), min_df=2, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, C=4.0, random_state=42)),
    ])
    pipe.fit(X_train, y_train)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, ARTIFACTS / "pipeline.joblib")
    return str(ARTIFACTS / "pipeline.joblib")


def evaluate(**_):
    """Посчитать метрики на отложенной выборке."""
    import json
    import joblib
    import pandas as pd
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(DATA_PROCESSED / "train_ready.csv")
    _, X_test, _, y_test = train_test_split(
        df[["Document_clean"]],
        df["Topic_group"],
        test_size=0.2,
        stratify=df["Topic_group"],
        random_state=42,
    )
    pipe = joblib.load(ARTIFACTS / "pipeline.joblib")
    pred = pipe.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "f1_macro": f1_score(y_test, pred, average="macro"),
    }
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False))
    return metrics


def log_to_mlflow(**_):
    """Зафиксировать метрики и модель в MLflow."""
    import json
    import joblib
    import mlflow
    import mlflow.sklearn

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("ticket-classification-pipeline")
    metrics = json.loads((ARTIFACTS / "metrics.json").read_text())
    with mlflow.start_run(run_name="airflow_daily"):
        mlflow.log_param("source", "airflow")
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        mlflow.sklearn.log_model(joblib.load(ARTIFACTS / "pipeline.joblib"), artifact_path="model")
    return metrics


def register_model(**_):
    """Зарегистрировать лучшую модель в MLflow Model Registry."""
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    experiment = mlflow.get_experiment_by_name("ticket-classification-pipeline")
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.f1_macro DESC"],
        max_results=1,
    )
    if not runs:
        return "no runs"
    model_uri = f"runs:/{runs[0].info.run_id}/model"
    client.create_registered_model(name="ticket-classifier")
    client.create_model_version(name="ticket-classifier", source=model_uri, run_id=runs[0].info.run_id)
    return model_uri


with DAG(
    "ticket_classification_pipeline",
    default_args=default_args,
    description="ML-pipeline: классификация обращений в поддержку",
    schedule_interval="0 6 * * 1",
    start_date=datetime(2026, 10, 26),
    catchup=False,
    tags=["ml", "pipeline", "mlflow", "production"],
) as dag:
    extract_task = PythonOperator(task_id="extract", python_callable=extract)
    validate_task = PythonOperator(task_id="validate", python_callable=validate)
    preprocess_task = PythonOperator(task_id="preprocess", python_callable=preprocess)
    train_task = PythonOperator(task_id="train", python_callable=train)
    evaluate_task = PythonOperator(task_id="evaluate", python_callable=evaluate)
    log_task = PythonOperator(task_id="log_to_mlflow", python_callable=log_to_mlflow)
    register_task = PythonOperator(task_id="register_model", python_callable=register_model)

    (
        extract_task
        >> validate_task
        >> preprocess_task
        >> train_task
        >> evaluate_task
        >> log_task
        >> register_task
    )
