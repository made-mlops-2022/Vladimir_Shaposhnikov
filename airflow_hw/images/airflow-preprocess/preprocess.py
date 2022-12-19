import os
from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd
import click


@click.command("preprocess")
@click.option("--input-dir")
@click.option("--output-dir")
def preprocess(input_dir: str, output_dir):
    data = pd.read_csv(os.path.join(input_dir, "data_test.csv"))
    scaler = StandardScaler()
    scaler.fit(data)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "scaler.pkl"), 'wb') as f:
        pickle.dump(scaler, f)


if __name__ == '__main__':
    preprocess()
