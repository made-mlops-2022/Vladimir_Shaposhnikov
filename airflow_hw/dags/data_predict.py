import os
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.python import PythonSensor
from docker.types import Mount

from data_config import default_args, SOURCE, TARGET, RAW_PATH, MODEL_PATH, RESULT_PATH

# def wait_for_file(file_name):
#     return os.path.exists(file_name)

with DAG(
        dag_id="data_predict",
        default_args=default_args,
        schedule_interval="@daily",
        tags=["Airflow"],
) as dag:

    data_preprocessing = PythonSensor(
        task_id="data_preprocessing",
        python_callable=lambda file: os.path.exists(file),
        op_args=["/opt/airflow/data/raw/{{ ds }}/data.csv"],
        timeout=6000,
        poke_interval=10,
        retries=100,
        mode="poke"
    )

    model_preprocessing = PythonSensor(
        task_id="model_preprocessing",
        python_callable=lambda file: os.path.exists(file),
        op_args=["/opt/airflow/data/models/{{ ds }}/model.pkl"],
        timeout=6000,
        poke_interval=10,
        retries=100,
        mode="poke"
    )
    scaler_preprocessing = PythonSensor(
        task_id="scaler_preprocessing",
        python_callable=lambda file: os.path.exists(file),
        op_args=["/opt/airflow/data/models/{{ ds }}/scaler.pkl"],
        timeout=6000,
        poke_interval=10,
        retries=100,
        mode="poke"
    )

    prediction = DockerOperator(
        image="airflow-predict",
        command=f"--input-dir={RAW_PATH} --scaler-path={MODEL_PATH} --output-dir={RESULT_PATH}",
        network_mode="bridge",
        task_id="predict",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=SOURCE, target=TARGET, type='bind')]
    )


    [data_preprocessing, model_preprocessing, scaler_preprocessing] >> prediction