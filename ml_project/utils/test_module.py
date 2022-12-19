from fastapi.testclient import TestClient
import pandas as pd
from fast_api import app

client = TestClient(app)


def test_read_main():
    with client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {'message': '/health - healthcheck \n /predict - prediction'}


def test_health_state():
    with client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == 200


def test_valid_request():
    with client:
        data = [[40, 0, 0, 120, 200, 0, 0, 110, 0, 0.1, 0, 0, 0, 0],
                [35, 1, 1, 180, 250, 1, 1, 180, 1, 4, 1, 1, 1, 1],
                [61, 1, 0, 150, 234, 0, 1, 165, 0, 2.6, 1, 2, 0, 1]]
        labels = ["age", "sex", "cp", "trestbps", "chol", "fbs",
                  "restecg", "thalach", "exang", "oldpeak",
                  "slope", "ca", "thal", "condition"]
        data = pd.DataFrame(data, columns=labels)
        req = {"data": data.to_json(), 'drop': ['condition']}
        response = client.post("http://127.0.0.1:8000/predict/", json=req)
        assert response.status_code == 200
        assert response.json() == {'result': [0, 1, 1]}

def test_invalid_rerquest():
    with client:
        data = [[69, 1, 0, 160, 234, 1, 2, 131, 0, 0.1, 1, 1, 0, 0],
                [66, 0, 0, 150, 226, 0, 0, 114, 0, 2.6, 2, 0, 0, 0],
                [61, 1, 0, 134, 234, 0, 0, 145, 0, 2.6, 1, 2, 0, 1]]
        response = client.post("/predict/", json=[data])
        assert response.status_code == 422

        response = client.post("/predict/", json=('some_data'))
        assert response.status_code == 422



if __name__ == '__main__':
    test_valid_request()