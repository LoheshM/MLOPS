# MLOPS: end-to-end ML pipelines, orchestration and cloud deployment

**Six small projects that take models from notebook to pipeline to endpoint.** They include a modular student-score regression pipeline with a Flask app, an MLflow-tracked wine-quality pipeline, a Pegasus dialogue summariser behind FastAPI, a scikit-learn model trained and deployed on AWS SageMaker, an Airflow ETL into Postgres, and a Bedrock-powered blog generator on AWS Lambda.

> Learning portfolio built while following an MLOps course, 2024 · Python · scikit-learn, MLflow / DagsHub, Hugging Face Transformers, FastAPI, Flask, Apache Airflow (Astro), Docker, AWS SageMaker / Bedrock / Lambda / S3

## Projects at a glance

| # | Project | Folder | Task | Headline result (source) |
|---|---|---|---|---|
| 1 | Student math-score prediction | [mlproject-main/](mlproject-main/) | Regression, 9 models compared, modular training pipeline, Flask UI | Linear Regression test R² **0.880** ([notebook](mlproject-main/notebook/2.%20MODEL%20TRAINING.ipynb)) |
| 2 | Wine-quality pipeline with MLflow | [datascienceproject/](datascienceproject/) | 5-stage config-driven pipeline, ElasticNet, MLflow/DagsHub tracking, Flask UI | Test R² 0.251, RMSE 0.711 ([metrics.json](datascienceproject/artifacts/model_evaluation/metrics.json)) |
| 3 | Dialogue summariser | [Textsummarizer_HuggingFace/](Textsummarizer_HuggingFace/) | Fine-tune `google/pegasus-cnn_dailymail` on SAMSum, FastAPI train/predict API | Training loss about 3.0 after 1 epoch. Logged ROUGE is not meaningful (see below). |
| 4 | Mobile price classification on SageMaker | [awssagemaker/](awssagemaker/) | Random forest trained as a SageMaker job, deployed to a real-time endpoint | Test accuracy **0.883** on 300 rows ([notebook](awssagemaker/research.ipynb)) |
| 5 | Airflow ETL | [ETL/](ETL/) | Daily DAG: NASA APOD API → transform → Postgres | No metrics (data pipeline) |
| 6 | Generative AI on AWS | [Generative-AI-With-Cloud-main AWS Sagemaker/](Generative-AI-With-Cloud-main%20AWS%20Sagemaker/) | Lambda + Bedrock (Llama 2 13B chat) blog generator writing to S3. Hugging Face models on SageMaker endpoints. | No metrics. Third-party course material, see [Licence](#licence). |

---

## 1. Student math-score prediction (`mlproject-main/`)

### The problem

The task is to predict a student's math score from gender, race/ethnicity, parental level of education, lunch type, test-preparation course, and reading and writing scores. The data is the [Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams?datasetId=74977) dataset (1,000 rows, 8 columns), included as [notebook/data/stud.csv](mlproject-main/notebook/data/stud.csv).

### Approach

| Stage | What happens |
|---|---|
| EDA ([notebook 1](mlproject-main/notebook/1%20.%20EDA%20STUDENT%20PERFORMANCE%20.ipynb)) | Missing value and duplicate checks (none found), distributions, and score breakdowns by gender, lunch, parental education and group |
| Model comparison ([notebook 2](mlproject-main/notebook/2.%20MODEL%20TRAINING.ipynb)) | One-hot + standard scaling (19 features), 80/20 split, 9 regressors |
| Data ingestion | [`src/components/data_ingestion.py`](mlproject-main/src/components/data_ingestion.py) writes `artifacts/data.csv`, `train.csv` and `test.csv` (80/20, `random_state=42`) |
| Transformation | [`data_transformation.py`](mlproject-main/src/components/data_transformation.py): median imputer + scaler for numeric columns; mode imputer + one-hot + scaler for categorical ones |
| Training | [`model_trainer.py`](mlproject-main/src/components/model_trainer.py): `GridSearchCV` (cv=3) over 7 regressors. Keeps the best test R², rejects anything below 0.6, saves `artifacts/model.pkl`. The committed model is a `LinearRegression`. |
| Serving | [`app.py`](mlproject-main/app.py): Flask form at `/predictdata` → [`predict_pipeline.py`](mlproject-main/src/pipeline/predict_pipeline.py) loads the preprocessor and model |

```
stud.csv ─► DataIngestion ─► train.csv / test.csv
                               │
                               ▼
                     DataTransformation ─► preprocessor.pkl
                               │
                               ▼
                 ModelTrainer (GridSearchCV × 7) ─► model.pkl ─► Flask /predictdata
```

EDA findings recorded in the notebook:
- Female students have a higher average overall score, while male students score higher in maths.
- Standard lunch goes with better scores for both genders.
- Groups A and B tend to score lower.
- Parental education shows little overall effect.

### Results

Test-set scores from [2. MODEL TRAINING.ipynb](mlproject-main/notebook/2.%20MODEL%20TRAINING.ipynb) (200 held-out students):

| Model | Test R² | Test RMSE | Test MAE |
|---|---|---|---|
| Ridge | 0.8806 | 5.390 | 4.211 |
| **Linear Regression** | **0.8803** | **5.396** | **4.216** |
| CatBoost | 0.8516 | 6.009 | 4.613 |
| AdaBoost | 0.8498 | 6.045 | 4.681 |
| Random Forest | 0.8473 | 6.096 | 4.719 |
| Lasso | 0.8253 | 6.520 | 5.158 |
| XGBoost | 0.8216 | 6.589 | 5.084 |
| K-Neighbors | 0.7838 | 7.253 | 5.621 |
| Decision Tree | 0.7603 | 7.637 | 6.025 |

The notebook selects Linear Regression ("Accuracy of the model is 88.03"). Ridge is ahead by 0.0003, which is effectively a tie. Reading and writing scores are input features, and they correlate strongly with the math score, which explains most of this R².

### Run it

```bash
cd mlproject-main
pip install -r requirements.txt dill
python -m src.components.data_ingestion   # runs ingestion → transformation → training, prints test R²
python app.py                             # Flask on :5000, open /predictdata
```

---

## 2. Wine-quality pipeline with MLflow (`datascienceproject/`)

### Approach

The project uses a config-driven layout: `config/config.yaml`, `params.yaml`, `schema.yaml`, entity dataclasses, a configuration manager, and one component and one pipeline per stage. [`main.py`](datascienceproject/main.py) runs the stages in order:

| Stage | Detail |
|---|---|
| Data ingestion | Downloads `winequality-data.zip` (URL in [config.yaml](datascienceproject/config/config.yaml)) and unzips `winequality-red.csv` |
| Data validation | Checks every column against [schema.yaml](datascienceproject/schema.yaml) (11 physico-chemical features + `quality`) and writes `status.txt` |
| Data transformation | `train_test_split` with defaults (75/25 → 1,199 / 400 rows) |
| Model trainer | `ElasticNet(alpha=0.2, l1_ratio=0.1)` from [params.yaml](datascienceproject/params.yaml), saved with joblib |
| Model evaluation | RMSE / MAE / R² saved to `metrics.json`; params, metrics and model logged to MLflow, registered as `ElasticnetModel` when the tracking store is remote (DagsHub) |
| Serving | [app.py](datascienceproject/app.py) Flask on :8080: `/` form, `/predict`, `/train` (re-runs `main.py`) |

```
config.yaml / params.yaml / schema.yaml
        │
        ▼
ingest (zip URL) ─► validate (schema) ─► split 75/25 ─► ElasticNet ─► evaluate ─► metrics.json
                                                                            └──► MLflow (DagsHub) + model registry
```

### Results

From [artifacts/model_evaluation/metrics.json](datascienceproject/artifacts/model_evaluation/metrics.json):

| RMSE | MAE | R² |
|---|---|---|
| 0.711 | 0.564 | 0.251 |

A plain ElasticNet on the raw features explains only a quarter of the variance in quality. The project shows the pipeline and the tracking; it is not meant as a tuned model.

### Run it

```bash
cd datascienceproject
pip install -r requirements.txt
python main.py        # all five stages
python app.py         # http://localhost:8080
```

`mlflow_uri` is hard-coded in [configuration.py](datascienceproject/src/datascience/config/configuration.py). Point it, and the `MLFLOW_TRACKING_*` variables, at your own MLflow or DagsHub server.

---

## 3. Dialogue summariser with Pegasus (`Textsummarizer_HuggingFace/`)

### Approach

| Stage | Detail |
|---|---|
| Data ingestion | Downloads `summarizer-data.zip` (the SAMSum dataset: 14,732 train / 819 test / 818 validation dialogues) |
| Transformation | `google/pegasus-cnn_dailymail` tokenizer, dialogue max length 1,024, summary max length 128 |
| Training | HF `Trainer`, 1 epoch, batch size 1 × 16 gradient accumulation, 500 warmup steps, weight decay 0.01 ([params.yaml](Textsummarizer_HuggingFace/params.yaml)) |
| Evaluation | ROUGE-1/2/L/Lsum with `evaluate`, beam search (8 beams, length penalty 0.8, max 128) → `metrics.csv` |
| Serving | [app.py](Textsummarizer_HuggingFace/app.py) FastAPI on :8080: `GET /train` runs `main.py`, `POST /predict?text=…` returns a summary |

### Results

From the saved notebook outputs:

| Run | Steps | Train loss | Runtime |
|---|---|---|---|
| Colab NVIDIA L4 ([textsummarizer.ipynb](Textsummarizer_HuggingFace/research/textsummarizer.ipynb)) | 51 | 3.052 | 142 s |
| Local machine ([3_model_trainer.ipynb](Textsummarizer_HuggingFace/research/3_model_trainer.ipynb)) | 51 | 3.023 | 2,117 s |

The Colab notebook also logs ROUGE-1 0.022, ROUGE-2 0.0 and ROUGE-L 0.022 on 10 test dialogues. Treat these numbers as broken, not as a measure of the model:
- The evaluation code calls `d.replace("", " ")`, which puts a space between every character of the generated summary before scoring.
- The trainer is fed the **test** split (819 dialogues) rather than the train split, so the "test" ROUGE is not held out.

The sample generation in the notebook gives a mostly extractive summary.

### Run it

```bash
cd Textsummarizer_HuggingFace
pip install -r requirements.txt
python main.py        # ingest → transform → train → evaluate (GPU strongly recommended)
python app.py         # http://localhost:8080/docs
```

SAMSum was created by Samsung R&D Institute Poland and is licensed CC BY-NC-ND 4.0 (non-commercial).

---

## 4. Mobile price classification on SageMaker (`awssagemaker/`)

### Approach

[research.ipynb](awssagemaker/research.ipynb) walks through the full SageMaker workflow on [mob_price_classification_train.csv](awssagemaker/mob_price_classification_train.csv). The data has 2,000 phones, 20 features and 4 balanced `price_range` classes.

| Step | Detail |
|---|---|
| Split | 85/15 (`random_state=0`) → `train-V-1.csv` (1,700) / `test-V-1.csv` (300), uploaded to S3 with `sess.upload_data` |
| Training script | [script.py](awssagemaker/script.py): `RandomForestClassifier(n_estimators=100, random_state=0)`; reads the `SM_CHANNEL_*` and `SM_MODEL_DIR` environment variables, saves `model.joblib`, prints test metrics |
| Training job | `SKLearn` estimator, framework 0.23-1, `ml.m5.large` |
| Deployment | `SKLearnModel` from the S3 artifact → real-time endpoint on `ml.m4.xlarge` → `predictor.predict(...)` → `delete_endpoint` to stop charges |

### Results

From the training-job log in the notebook (`ml.m5.large`, 99 billable seconds):

| Class | Precision | Recall | F1 |
|---|---|---|---|
| 0 | 0.95 | 1.00 | 0.97 |
| 1 | 0.85 | 0.80 | 0.83 |
| 2 | 0.80 | 0.77 | 0.79 |
| 3 | 0.91 | 0.95 | 0.93 |
| **Accuracy** | | | **0.883** (300 rows) |

### Run it

You need an AWS account, an S3 bucket and a SageMaker execution role. Replace the bucket name and role ARN in the notebook, then run the cells in order. Delete the endpoint when you are done.

---

## 5. Airflow ETL: NASA APOD → Postgres (`ETL/`)

| Task | Detail |
|---|---|
| `create_table` | `PostgresHook` creates `apod_data(id, title, explanation, url, date, media_type)` if it doesn't exist |
| `extract_apod` | `SimpleHttpOperator` GET `planetary/apod` on the `nasa_api` connection, API key taken from the connection's extras |
| `transform_apod_data` | TaskFlow `@task`: keeps title, explanation, url, date and media_type |
| `load_data_to_postgres` | Parameterised `INSERT` through the `my_postgres_connection` connection |

```
create_table ─► extract_apod (NASA APOD API) ─► transform_apod_data ─► load_data_to_postgres
                             DAG nasa_apod_postgres · @daily · catchup=False
```

### Run it

This is an Astronomer project: the [Dockerfile](ETL/Dockerfile) is `astro-runtime:12.1.1`, and [docker-compose.yml](ETL/docker-compose.yml) adds a `postgres:13` service.

```bash
cd ETL
astro dev start
```

Then create two Airflow connections:
- `my_postgres_connection`, pointing at the Postgres service.
- `nasa_api` (HTTP, host `https://api.nasa.gov/`, extra `{"api_key": "<your key>"}`).

Finally, trigger `nasa_apod_postgres`. [ETL/README.md](ETL/README.md) has the full write-up.

---

## 6. Generative AI on AWS (`Generative-AI-With-Cloud-main AWS Sagemaker/`)

| Item | What it does |
|---|---|
| [Blog generation in aws/app.py](Generative-AI-With-Cloud-main%20AWS%20Sagemaker/Generative-AI-With-Cloud-main/Blog%20generation%20in%20aws/app.py) | AWS Lambda handler. It reads `blog_topic` from the request body, asks Bedrock `meta.llama2-13b-chat-v1` for a 200-word blog (temperature 0.5, top-p 0.9, 512 tokens), and writes it to S3 under `blog-output/<HHMMSS>.txt`. |
| [AWS sagemaker/Untitled (2).ipynb](Generative-AI-With-Cloud-main%20AWS%20Sagemaker/Generative-AI-With-Cloud-main/AWS%20sagemaker/Untitled%20%282%29.ipynb) | Deploys `distilbert-base-uncased-distilled-squad` as a SageMaker Hugging Face endpoint and runs question-answering requests |
| [AWS sagemaker/falcon40B-instruct-notebook-full.ipynb](Generative-AI-With-Cloud-main%20AWS%20Sagemaker/Generative-AI-With-Cloud-main/AWS%20sagemaker/falcon40B-instruct-notebook-full.ipynb) | AWS workshop lab: hosts `tiiuae/falcon-40b-instruct` with the Hugging Face LLM (TGI) container on `ml.g5.12xlarge` |

```
API Gateway / test event ─► Lambda (app.py) ─► Bedrock llama2-13b-chat ─► S3 blog-output/<time>.txt
```

To deploy the Lambda: package it with a recent `boto3` (the [requirements.txt](Generative-AI-With-Cloud-main%20AWS%20Sagemaker/Generative-AI-With-Cloud-main/Blog%20generation%20in%20aws/requirements.txt) only lists `boto3`). Give the function role Bedrock and S3 permissions, and set `s3_bucket` to your own bucket.

---

## Repository layout

```
mlproject-main/                 1. student score regression: notebooks, src/ pipeline, Flask app, artifacts/
datascienceproject/             2. wine quality: config/params/schema YAML, src/datascience/, research/ notebooks, Flask app
Textsummarizer_HuggingFace/     3. Pegasus + SAMSum: config/params YAML, src/textSummarizer/, research/ notebooks, FastAPI app
awssagemaker/                   4. SageMaker notebook, script.py, mobile-price CSVs
ETL/                            5. Astro Airflow project: dags/etl.py, Dockerfile, docker-compose.yml, tests/
Generative-AI-With-Cloud-main AWS Sagemaker/
  └── Generative-AI-With-Cloud-main/
      ├── Blog generation in aws/   6. Bedrock blog-generator Lambda
      ├── AWS sagemaker/            Hugging Face / Falcon SageMaker notebooks
      └── LICENSE                   GPL-3.0 (upstream)
```

## Limitations

- **These are course-style projects, not production systems.** Several READMEs inside the subfolders are workflow checklists from the course templates.
- **Known bugs in `mlproject-main`:**
  - `app.py` swaps the reading and writing scores when building the input.
  - The transformer saves `proprocessor.pkl`, but the predictor loads `preprocessor.pkl`. The committed artifact has the correct name, but a fresh training run will not overwrite it.
  - The ingestion path `notebook\data\stud.csv` uses a Windows separator.
  - `train_pipeline.py` is empty.
- **The Pegasus summariser is trained on the SAMSum test split** and evaluated on 10 of the same dialogues, and its ROUGE code is broken (see section 3).
- **Hard-coded external services.**
  - `datascienceproject` logs to a DagsHub MLflow URI belonging to the course author.
  - The Bedrock Lambda writes to a fixed bucket name (`aws_bedrock_course1`), which is not a valid S3 bucket name because it contains underscores.
  - The SageMaker notebooks contain account-specific bucket names and IAM role ARNs.
- **Empty Dockerfiles.** `datascienceproject/Dockerfile` and `Textsummarizer_HuggingFace/Dockerfile` are empty, and there are no CI/CD workflows in the repo.
- **Unpinned or old dependencies.** Most `requirements.txt` files are unpinned. The summariser pins FastAPI 0.78 / uvicorn 0.18. Bedrock's Llama 2 model IDs and SageMaker framework 0.23-1 are outdated.
- **Committed data and artifacts.** Datasets, pickled models, logs and `__pycache__` files are committed. Only unpickle `model.pkl` / `preprocessor.pkl` if you trust the source.

## Licence

MIT. See [LICENSE](LICENSE). The MIT licence covers this repository **except** two subfolders, which keep their own GPL-3.0 licences:

- **`Generative-AI-With-Cloud-main AWS Sagemaker/`:** GPL-3.0, see [its LICENSE](Generative-AI-With-Cloud-main%20AWS%20Sagemaker/Generative-AI-With-Cloud-main/LICENSE).
  - This looks like **third-party tutorial code**. The `-main` folder name matches a downloaded GitHub archive of a public "Generative AI On Cloud (Azure And AWS)" course repository; its original README (since removed) linked to the course author's YouTube channels.
  - The Falcon notebook is an AWS workshop lab ("Lab 1 (b)").
- **`Textsummarizer_HuggingFace/`:** GPL-3.0, see [its LICENSE](Textsummarizer_HuggingFace/LICENSE).
  - This looks like **code written by following a course tutorial**. The research notebooks were run from a local `Udemy … MLOPS` course folder, the data comes from the course author's dataset repository, and the GPL licence file appears to come from that course template.

Also note:
- `datascienceproject/` contains its own GPL-3.0 [LICENSE](datascienceproject/LICENSE) file and follows the same course template, with its MLflow tracking pointed at the course author's DagsHub. Treat that folder as GPL-3.0 as well.
- `mlproject-main/` and `awssagemaker/` appear to follow the same course's project walkthroughs.
- Datasets keep their original terms: Kaggle Students Performance in Exams, the red wine quality data, SAMSum (CC BY-NC-ND 4.0) and the mobile price classification data.
- The Llama 2 and Falcon models are under their providers' licences.
