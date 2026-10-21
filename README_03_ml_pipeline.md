# Решение «Практика pipeline» — ML-pipeline, эксперименты и логирование

**Задание:** практическое задание «Практика pipeline» модуля «ML-pipeline и эксперименты» программы AIML-5.

**Файл:** `03_ML_pipeline_solution.ipynb` — полностью воспроизводимый ноутбук (все ячейки выполнены,
результаты сохранены в ноутбуке).

---

## 1. Что сделано

| Шаг | Содержание |
|-----|------------|
| 1 | Загрузка датасета, первичный анализ, распределение классов, длина текстов |
| 2 | Кастомный sklearn-трансформер `TextPreprocessor` (BaseEstimator + TransformerMixin) |
| 3 | Сборка `Pipeline` (препроцессинг → TF-IDF → классификатор), обучение только на `train` |
| 4 | Логирование всех экспериментов в MLflow (`log_param` / `log_metric` / `log_model`) |
| 5 | Подбор гиперпараметров `GridSearchCV` (3 фолда, отдельный эксперимент MLflow) |
| 6 | Сравнительная таблица метрик, confusion matrix, precision/recall по классам, выводы |
| 7 | DAG для Apache Airflow: `extract → validate → preprocess → train → evaluate → log_to_mlflow → register_model` |

## 2. Структура репозитория (для этого задания)

| Файл | Описание |
|------|----------|
| `03_ML_pipeline_solution.ipynb` | Ноутбук-решение (основной файл) |
| `README_03_ml_pipeline.md` | Этот README |
| `airflow_dag.py` | DAG оркестрации пайплайна (файл для каталога `dags/` Airflow) |
| `text_preprocessor.py` | Трансформер `TextPreprocessor` — общий модуль для ноутбука и DAG |

## 3. Архитектура пайплайна

```
DataFrame ["Document", "Topic_group"]
     │
     ▼  TextPreprocessor        lower-case, чистка HTML/пунктуации, стоп-слова, опц. лемматизация
     │                          (BaseEstimator + TransformerMixin → sklearn-совместимый)
     ▼
     │  TfidfVectorizer         ngram_range=(1,1), min_df=2, max_df=0.9, sublinear_tf=True
     ▼
     │  LogisticRegression      мультиклассовая классификация 8 тем
     ▼
предсказания Topic_group
```

Ключевое правило — **`fit` вызывается только на `train`**. Векторизатор и модель обучаются внутри
одного `Pipeline`, поэтому на тестовой выборке вызывается `transform`, а не `fit_transform`:
утечки данных нет. Это же объясняет, почему ручную векторизацию «до» разделения выборки делать нельзя.

## 4. Эксперименты и MLflow

* Трекинг: SQLite (`sqlite:///mlflow.db`), эксперимент `ticket-classification-pipeline`.
* На каждый запуск: `mlflow.log_param` (модель, параметры TF-IDF, размер train, seed),
  `mlflow.log_metric` (accuracy, precision_macro, recall_macro, f1_macro, f1_weighted),
  `mlflow.sklearn.log_model` (обученный Pipeline).
* Подбор гиперпараметров — в отдельном эксперименте `ticket-classification-pipeline-tuning`,
  поэтому победителя видно в интерфейсе MLflow.

### Как посмотреть эксперименты

```bash
pip install -r requirements.txt
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Откроется интерфейс на `http://127.0.0.1:5000` — во вкладке **Experiments** будут видны оба
эксперимента со всеми параметрами и метриками.

## 5. Оркестрация в Apache Airflow

`airflow_dag.py` — готовый DAG `ticket_classification_pipeline` (еженедельное расписание, `0 6 * * 1`):

| Задача | Что делает |
|--------|------------|
| `extract` | Читает сырой датасет обращений |
| `validate` | Проверяет схему и отсутствие пропусков, возвращает распределение классов |
| `preprocess` | Сохраняет очищенные тексты (`train_ready.csv`) |
| `train` | Обучает Pipeline и сохраняет `pipeline.joblib` |
| `evaluate` | Считает accuracy и F1-macro на отложенной выборке |
| `log_to_mlflow` | Логирует метрики и модель в MLflow |
| `register_model` | Публикует лучшую модель в MLflow Model Registry |

Чтобы запустить:

```bash
export AIRFLOW_HOME=~/airflow
cp airflow_dag.py text_preprocessor.py $AIRFLOW_HOME/dags/
airflow standalone
```

## 6. Как запустить ноутбук

### Вариант A — Google Colab

1. Загрузить `03_ML_pipeline_solution.ipynb` в Colab.
2. Скачать датасет с Kaggle:
   `https://www.kaggle.com/datasets/adisongoh/it-service-ticket-classification-dataset`
   и положить `all_tickets_processed_improved_v3.csv` в папку `data/` (или см. п. «Как скачать датасет»).
3. `Kernel → Restart and Run All`.

### Вариант B — локально (Jupyter / JupyterLab / VS Code)

```bash
pip install -r requirements.txt
# положить all_tickets_processed_improved_v3.csv в ./data/
jupyter lab 03_ML_pipeline_solution.ipynb
```

Ноутбук сам подхватывает датасет из `data/all_tickets_processed_improved_v3.csv`; если файл не найден,
он подскажет ссылку на Kaggle.

## 7. Результаты

<!--METRICS_TABLE-->

## 8. Выводы

1. **Пайплайн вместо «кода в ноутбуке».** Весь обучающий код собран в один объект `Pipeline`,
   который сохраняется в MLflow, версионируется и переиспользуется в проде
   (`pipeline.predict(...)`). Ручная предобработка «в ноутбуке» невоспроизводима.
2. **Главный риск — утечка данных.** `TfidfVectorizer.fit_transform` на всей выборке до
   разделения обучает словарь и IDF на тестовых документах. Здесь `fit` вызывается только на `train`,
   а разделение выполнено со `stratify=y`, поэтому пропорции классов в train/test совпадают.
3. **Эксперименты фиксируются в MLflow.** Параметры, метрики и сама обученная модель попадают
   в трекинг — сравнение моделей сводится к одной таблице, а не к разбросанным по ноутбуку числам.
4. **Подбор гиперпараметров — по CV, а не по test.** `GridSearchCV` выбирает конфигурацию
   на 3 фолдах train, после чего модель дообучается на всём train и один раз проверяется на test.
   Тестовая выборка не участвует в выборе модели.
5. **Оркестрация.** DAG связывает шаги в воспроизводимый конвейер; валидация между `extract`
   и `train` защищает от обучения на битых данных, а `register_model` публикует модель только
   после проверки метрик.
6. **Ограничения и следующие шаги.**
   * TF-IDF + линейная модель — сильный baseline, но не учитывает контекст; следующий шаг —
     эмбеддинги (fastText / трансформеры) или дообучение BERT-энкодера.
   * Классы не сбалансированы (max/min ≈ 7.7) — для редких классов нужен `class_weight='balanced'`
     и отслеживание F1 по каждому классу, а не только macro.
   * Лемматизация WordNet даёт небольшой прирост; на доменных терминах (ID, VPN, принтер)
     нужен собственный словарь или стемминг.
   * MLflow на SQLite — в команде переезжает на сервер с PostgreSQL-бэкендом, Airflow — на
     KubernetesExecutor.
   * В проде необходим мониторинг: drift распределения обращений, деградация метрик,
     алерт при падении F1 ниже порога (в Airflow — задача `model_quality_check`).
