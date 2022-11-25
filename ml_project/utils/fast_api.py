import os
import pickle

import pandas as pd
import uvicorn
import gdown
from fastapi import FastAPI

import logging
from pydantic import BaseModel, conlist
from typing import List, Union

from yaml import safe_load
app = FastAPI()

if not os.path.exists("logs"):
    os.mkdir("logs")
logger = logging.getLogger(__name__)
handler = logging.FileHandler("logs/fast_api.log")
formatter = logging.Formatter("%(asctime)s - %(name)s "
                              "" "- %(levelname)s - %(message)s")
handler.setFormatter(formatter)
handler.setLevel(logging.INFO)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

model = None
config = None


class ModelData(BaseModel):
    data: List[conlist(Union[float, str, None], min_items=13, max_items=13)]


class ModelResponse(BaseModel):
    result: List[int]


def load_config():
    env_conf = os.getenv("PATH_TO_CONFIG")
    conf = env_conf if env_conf else 'configs/api_config.yml'
    with open(conf, 'r') as stream:
        config = safe_load(stream)
    return config


@app.on_event('startup')
def model_init():
    global model
    global config
    logger.info('Loading config file')
    try:
        config = load_config()
        logger.info(f'Config file loaded succesfully, params - {config}')
    except Exception as err:
        logger.critical(f'Critical error in load_config, message - {err}')
        return 1

    try:
        if not os.getenv('GLINK_FILE'):
            env_model = os.getenv('PATH_TO_MODEL')
            model_path = env_model if env_model else config['path']['model_path']
        else:
            url = str(os.getenv('GLINK_FILE'))
            model_path = './model.pkl'
            gdown.download(id=url, output=model_path, quiet=False)

        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    except Exception as err:
        logger.critical(f'Critical error in model load, message - {err}')
        return 1
    logger.info('Model init completed')


@app.get("/")
async def root():
    return {"message": "/health - healthcheck \n /predict - prediction"}


@app.get('/health')
async def is_health():
    if model is not None:
        return 200
    else:
        return 400


@app.post("/predict/")
async def predict(request: dict):
    df_json = request['data']
    drop_columns = request['drop']
    try:
        data = pd.read_json(df_json)

        if drop_columns:
            data.drop(drop_columns, axis=1, inplace=True)
        logger.info('Data loaded successfully')
    except Exception as err:
        logger.critical(f'Critical error in model load, message - {err}')
        return 1

    try:
        prediction = model.predict(data)
        logger.info(f'Model get prediction - {prediction.tolist()}')
        return ModelResponse(result=prediction.tolist())

    except Exception as err:
        logger.critical(f'Critical error in predict, message - {err}')
        return 1


if __name__ == "__main__":
    logger.info("Server started")
    uvicorn.run("fast_api:app", host="0.0.0.0", port=os.getenv("PORT", 8000))
