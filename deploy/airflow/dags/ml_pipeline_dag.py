from datetime import datetime
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
import os

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

with DAG(
    dag_id="ml_pipeline_dag",
    start_date=datetime.today().replace(hour=0, minute=0, second=0, microsecond=0),
    schedule_interval=None,
    catchup=False,
) as dag:
    run_pipeline = DockerOperator(
    task_id="run_pipeline",
    image="d9f4b60ba0c3e8f72f9b10721c588ff2e909d163ec13825591a283675158d548-ml-pipeline",
    command="python src/pipeline.py",
    docker_url="unix://var/run/docker.sock",
    network_mode="bridge",
    working_dir="/app",
    auto_remove=True,
    tty=True,
    mount_tmp_dir=False,
)
