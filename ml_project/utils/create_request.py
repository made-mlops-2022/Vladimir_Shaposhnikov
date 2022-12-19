import os

import pandas as pd
import requests

if __name__ == "__main__":
    env_data = os.getenv('PATH_TO_DATA')
    path = env_data if env_data else "dataset/heart_cleveland_upload.csv"
    data = pd.read_csv(path)
    req = {"data": data.to_json(), 'drop': ['condition']}
    response = requests.post("http://127.0.0.1:8000/predict/", json=req)
    print(response.status_code)
    print(response.json())

