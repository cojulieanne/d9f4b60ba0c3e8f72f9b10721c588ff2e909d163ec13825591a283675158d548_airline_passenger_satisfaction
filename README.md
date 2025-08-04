# MLOps Homework 2: Containerizing and Orchestrating ML Workflows

## Project Overview

This project addresses the problem of predicting airline passenger satisfaction using survey and service-related data. By modeling this classification task, we aim to help airlines better understand the factors that lead to higher satisfaction and act upon them to improve customer experience. The dataset includes labeled instances with demographic details, flight-related attributes, and in-flight service ratings. It is small (<100MB), well-structured, and ideal for training a single supervised ML model end-to-end. This makes it a good candidate for demonstrating a production-oriented MLOps workflow.

See model training pipeline in `branch/main`

In this phase, the focus shifts to **productionizing** the pipeline. We are currently in `branch/hw2-docker-airflow`. We use **Docker** to encapsulate dependencies and code in a consistent environment, ensuring reproducibility across systems. We orchestrate execution using **Apache Airflow**, which allows scalable workflow management with support for retries, logging, and dependency tracking — critical for managing complex, long-running ML pipelines.

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone -b hw2-docker-airflow https://github.com/cojulieanne/d9f4b60ba0c3e8f72f9b10721c588ff2e909d163ec13825591a283675158d548_airline_passenger_satisfaction.git
cd d9f4b60ba0c3e8f72f9b10721c588ff2e909d163ec13825591a283675158d548_airline_passenger_satisfaction
```

### 2. Build the Docker Image
 ```bash
 docker build -f deploy/docker/Dockerfile -t d9f4b60ba0c3e8f72f9b10721c588ff2e909d163ec13825591a283675158d548-ml-pipeline .
 ```

### 3. Run the Container
 ```bash
 docker run d9f4b60ba0c3e8f72f9b10721c588ff2e909d163ec13825591a283675158d548-ml-pipeline
 ```

 ### 4. Start Airflow with Docker Compose
 ``` bash
 docker-compose -f deploy/docker-compose.yaml up --build
 ```

 ### 5. Access Airflow UI
 ```bash
 http://localhost:8080
 ```
Login Credentials:
* Username: `airflow`
* Password: `airflow`

### 6. Trigger the DAG
In the Airflow UI:
* Unpause the DAG named `ml_pipeline_dag`
* Click **Trigger DAG**

You may track the logs of the pipeline either thru the Airflow UI in the `Logs` or in the folder `deploy/airflow/logs`.

## Docker Integration
To ensure environment consistency and reproducibility, we encapsulate all pipeline dependencies within a custom Docker image. The Dockerfile is located at deploy/Dockerfile, and it installs essential libraries such as `pandas`, `scikit-learn`, `matplotlib`, `shap`,  `hyperopt`, etc based on the `pyproject.toml`. Building and running the image is outline in Steps *2* and *3*.

Data directory is mounted as a volume as seen in the `docker-compose.yaml`:
``` bash
  volumes:
    - /var/run/docker.sock:/var/run/docker.sock
    - ${AIRFLOW_PROJ_DIR:-.}/deploy/airflow/dags:/opt/airflow/dags
    - ${AIRFLOW_PROJ_DIR:-.}/deploy/airflow/logs:/opt/airflow/logs
    - ${AIRFLOW_PROJ_DIR:-.}/src:/app/src
    - ${AIRFLOW_PROJ_DIR:-.}/data:/app/data
```
This ensures that updates to the host machine’s data are immediately reflected inside the container without rebuilding the image.

## Airflow DAG
* DAG ID: ml_pipeline_dag
* Start Date: Dynamically set to today's date
* Schedule: Manual trigger (schedule_interval=None)
* Task: run_pipeline (executes src/pipeline.py inside the container)

## Project Structure

```text
.
├── analysis/
│   └── eda.ipynb                  # Jupyter notebook for Exploratory Data Analysis
├── data/
│   ├── full_data.csv              # Merged dataset used for training and testing
│   ├── train.csv                  # Original training data (e.g., from Kaggle)
│   └── test.csv                   # Original test data (e.g., from Kaggle)
├── deploy/
│   ├── airflow/
│   │   ├── dags/                  # Airflow DAG definitions
│   │   └── logs/                  # Airflow task execution logs
│   ├── docker/
│   │   ├── Dockerfile             # Dockerfile for building custom image
│   ├── docker-compose.yaml        # Compose file to orchestrate Airflow services
│   └── .env                       # Environment variables used by Docker Compose
├── results/                       # Trained model and output artifacts
├── src/
│   ├── data_preprocessing.py      # Functions to load, clean, encode, and split dataset
│   ├── evaluation.py              # Model evaluation and plotting utilities
│   ├── model_training.py          # Model training, tuning, and saving logic
│   └── pipeline.py                # End-to-end ML pipeline script
├── .pre-commit-config.yaml        # Pre-commit hook configuration
├── pyproject.toml                 # Project metadata and dependency configuration (for `uv` or `pip`)
└── README.md                      # Project overview and documentation

```

## Pre-commit Configuration
To maintain code quality and style, the following pre-commit hooks are configured:

* Ruff: Fast Python linter and formatter that detects style violations and common errors.

* Trailing Whitespace: Removes unnecessary trailing spaces from code files.

* End-of-File Fixer: Ensures files end with a newline for readability and compatibility.

* Hadolint: Lints the `Dockerfile` to ensure Docker best practices, such as avoiding unnecessary root privileges and pinning base image versions.

* Yamllint: Validates `docker-compose.yaml` and other YAML files for structural and stylistic correctness.

These hooks automatically run before commits, helping maintain clean and consistent code.

## Reflection

One of the main challenges I encountered in this project was setting up Airflow to work smoothly with my existing pipeline, particularly around path handling inside the Dockerized Airflow environment. My scripts worked perfectly in a standalone Docker container, but once integrated into Airflow via the `DockerOperator`, the relative paths I was using began to break.

At first, I tried to mount the `data/` folder into the container, but it was difficult to verify the visibility and correctness of the paths from within Airflow's perspective. I then attempted to copy the data into the image itself as a quick workaround since the dataset was small. While that worked temporarily, I realized this approach may not be scalable or reproducible for larger or changing datasets.
