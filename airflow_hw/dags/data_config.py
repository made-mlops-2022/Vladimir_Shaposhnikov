from datetime import timedelta
from airflow.utils.dates import days_ago

default_args = {
    "owner": "airflow",
    "email": ["vladimirshaposhnikovmade@gmail.com"],
    "email_on_failure": True,
    "email_on_retry": True,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "start_date": days_ago(2),
    "depends_on_past": False,

}
SOURCE = "/home/voland/Workspace/Vladimir_Shaposhnikov/airflow_hw/data"
TARGET = "/data"

RAW_PATH = "/data/raw/{{ ds }}"
PROCESSED_PATH = "/data/processed/{{ ds }}"

MODEL_PATH = "/data/models/{{ ds }}"
RESULT_PATH = "/data/predictions/{{ ds }}"