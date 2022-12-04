import glob
import os
import time
import unittest
from random import randint

import click
import pandas as pd


@click.command("generation")
@click.argument("output_dir")
def get_data(output_dir):
    categorial = ['sex', 'cp', 'fbs', 'restecg','exang', 'slope', 'ca', 'thal']

    numerical = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    columns = list(categorial) + list(numerical)
    print(output_dir)
    data = [[randint(40, 80), randint(0, 1), randint(0, 3),
             randint(100, 180), randint(50, 300), randint(0, 1),
             randint(0, 2), randint(90, 180), randint(0, 1),
             randint(0, 9), randint(0, 2), randint(0, 3), randint(0, 2)]
            for _ in range(10000)]
    data = pd.DataFrame(data, columns=columns)
    target = pd.DataFrame([randint(0, 1) for _ in range(10000)], columns=['target'])

    print(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    data.to_csv(f'{output_dir}/data.csv', index=False)
    target.to_csv(f'{output_dir}/target.csv', index=False)


if __name__ == '__main__':
    get_data()




