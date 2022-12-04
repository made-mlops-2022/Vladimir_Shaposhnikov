import os
import datetime
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.python import PythonSensor
from docker.types import Mount

from data_config import default_args, SOURCE, TARGET, RAW_PATH, PROCESSED_PATH, MODEL_PATH


def wait_for_file(file_name):
    return os.path.exists(file_name)


with DAG(
        dag_id="data_train",
        default_args=default_args,
        schedule_interval="@daily",
        tags=["Airflow"],
) as dag:
    data_preprocessing = PythonSensor(
        task_id="data_preprocessing",
        python_callable=wait_for_file,
        op_args=["/opt/airflow/data/raw/{{ ds }}/data.csv"],
        timeout=6000,
        poke_interval=10,
        retries=100,
        mode="poke"
    )

    get_target = PythonSensor(
        task_id="get_target",
        python_callable=wait_for_file,
        op_args=["/opt/airflow/data/raw/{{ ds }}/target.csv"],
        timeout=6000,
        poke_interval=10,
        retries=100,
        mode="poke"
    )

    data_splitting = DockerOperator(
        image="airflow-split",
        command=f"--input-dir={RAW_PATH} --output-dir={PROCESSED_PATH}",
        network_mode="bridge",
        task_id="split",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=SOURCE, target=TARGET, type='bind')]
    )

    preprocess = DockerOperator(
        image="airflow-preprocess",
        command=f"--input-dir={PROCESSED_PATH} --output-dir={MODEL_PATH}",
        network_mode="bridge",
        task_id="airflow-preprocess",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=SOURCE, target=TARGET, type='bind')]
    )

    train = DockerOperator(
        image="airflow-train",
        command=f"--data-dir={PROCESSED_PATH} --scaler-dir={MODEL_PATH} --output-dir={MODEL_PATH}",
        network_mode="bridge",
        task_id="train",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=SOURCE, target=TARGET, type='bind')]
    )

    validate = DockerOperator(
        image="airflow-validate",
        command=f"--data-dir={PROCESSED_PATH} --scaler-dir={MODEL_PATH} --output-dir={MODEL_PATH}",
        network_mode="bridge",
        task_id="validate",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=SOURCE, target=TARGET, type='bind')]
    )

    [data_preprocessing, get_target] >> data_splitting >> preprocess >> train >> validate
