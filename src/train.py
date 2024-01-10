from sklearn.pipeline import Pipeline
import pandas as pd
import yaml
from pathlib import Path

# Load the configuration file
config_path = Path.cwd() / 'config.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)


# read the data

input_path = config['preprocessed_train_filepath']

# read the house price prediction dataset
x = pd.read_csv(input_path)
traget_column = 'SalePrice'

# make a sklearn pipe line to predict house price 