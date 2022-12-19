import os
import pandas as pd
import click
import pickle
import json

from sklearn.metrics import accuracy_score, precision_score, \
    recall_score, roc_auc_score


@click.command("validate")
@click.option("--data-dir")
@click.option("--scaler-dir")
@click.option("--output-dir")
def validate(data_dir, scaler_dir, output_dir):
    data = pd.read_csv(os.path.join(data_dir, "data_test.csv"))
    target = pd.read_csv(os.path.join(data_dir, "target_test.csv"))

    with open(os.path.join(scaler_dir, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    data = scaler.transform(data)

    with open(os.path.join(scaler_dir, "model.pkl"), "rb") as f:
        model = pickle.load(f)

    prediction = model.predict(data)

    metric_dict = {'accuracy': accuracy_score(prediction, target),
                   'precision': precision_score(prediction, target),
                   'recall': recall_score(prediction, target),
                   'roc_auc': roc_auc_score(prediction, target)}

    with open(f'{output_dir}/metrics.json', "w") as f:
        json.dump(metric_dict, f)


if __name__ == '__main__':
    validate()
