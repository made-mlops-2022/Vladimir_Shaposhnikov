from datetime import datetime
import os
import pickle

import click
import numpy as np
import pandas as pd


@click.command("predict")
@click.option("--input-dir")
@click.option("--output-dir")
@click.option("--scaler-path")
def predict(input_dir, output_dir, scaler_path):
    data = pd.read_csv(f'{input_dir}/data.csv')

    with open(f'{scaler_path}/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    with open(f'{scaler_path}/model.pkl', 'rb') as f:
        model = pickle.load(f)

    data = scaler.transform(data)
    prediction = model.predict(data)
    os.makedirs(output_dir, exist_ok=True)

    timestamp = str(datetime.now().date()) + \
                '_' + str(datetime.now().time()).replace(':', '.')
    np.savetxt(f'{output_dir}'
               f'/result_{timestamp}.csv',
               prediction, delimiter=",")


if __name__ == '__main__':
    predict()