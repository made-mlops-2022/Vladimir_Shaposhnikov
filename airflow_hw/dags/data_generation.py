from datetime import timedelta
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

from data_config import default_args, SOURCE, TARGET, RAW_PATH

with DAG(
        dag_id="data_gen",
        default_args=default_args,
        schedule_interval="@daily",
        tags=["Airflow"],
) as dag:
    get_data = DockerOperator(
        image="airflow-data-generation",
        command=RAW_PATH,
        task_id="docker-airflow-data-generation",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=SOURCE, target=TARGET, type='bind')],
    )

    get_data