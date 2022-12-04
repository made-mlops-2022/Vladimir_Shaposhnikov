import os
import pandas as pd
import click
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


@click.command("train")
@click.option("--data-dir")
@click.option("--scaler-dir")
@click.option("--output-dir")
@click.option("--model_type", default='linear')
def train(data_dir, scaler_dir, output_dir, model_type):
    data = pd.read_csv(os.path.join(data_dir, "data_train.csv"))
    target = pd.read_csv(os.path.join(data_dir, "target_train.csv"))
    with open(os.path.join(scaler_dir, "scaler.pkl"), 'rb') as f:
        scaler = pickle.load(f)

    data = scaler.transform(data)

    model_dict = {'svc': SVC(), 'knn': KNeighborsClassifier(),
                'linear': LogisticRegression(), 'naive': GaussianNB()}

    model = model_dict[model_type]
    model.fit(data, target)

    os.makedirs(output_dir, exist_ok=True)
    with open(f'{output_dir}/model.pkl', "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    train()
