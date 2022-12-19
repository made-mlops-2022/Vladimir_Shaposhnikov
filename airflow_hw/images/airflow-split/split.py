
import os
import pandas as pd
import click
from sklearn.model_selection import train_test_split


@click.command("split")
@click.option("--input-dir")
@click.option("--output-dir")
@click.option("--test-size", default=0.3)
def split(input_dir, output_dir, test_size):
    data = pd.read_csv(f'{input_dir}/data.csv')
    target = pd.read_csv(f'{input_dir}/target.csv')
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=test_size, stratify=target)

    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(os.path.join(output_dir, "data_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "data_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "target_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "target_test.csv"), index=False)


if __name__ == '__main__':
    split()