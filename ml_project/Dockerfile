FROM python:3.10-slim

COPY requirements.txt requirements.txt
RUN python -m pip install --upgrade pip && pip install -r requirements.txt --no-cache-dir && pip install pytest

COPY utils/fast_api.py utils/create_request.py utils/test_module.py app/

COPY configs/api_config.yml app/configs/api_config.yml
COPY dataset app/dataset
#COPY model/trained/model.pkl app/model.pkl
WORKDIR /app
#ENV PATH_TO_MODEL="model.pkl"
ENV PATH_TO_DATA='dataset/heart_cleveland_upload.csv'
ENV GLINK_FILE='1QHykZTa1tBkUsNAbj7oTIKTnx-_eXta0'
ENV PATH_TO_CONFIG="configs/api_config.yml"

CMD ["python3", "fast_api.py"]