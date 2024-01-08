import logging
from pathlib import Path

import pandas as pd
import yaml
from icecream import ic
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the configuration file
config_path = Path.cwd()/'config.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)
    
# print(config, type(config))
# x = pd.read_csv(config['train_data_path'])

def clean_data(input_path, output_path):
    # Load the dataset
    logger.info(f'input_paht: {input_path}')
    logger.info(f'output_path: {output_path}')
    
    df = pd.read_csv(input_path)
    rows, cols = df.shape
    logger.info(f'data shape: {df.shape}')

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Handle missing values
    def handle_missing_values(df):
        # Identify and drop columns with a high percentage of missing values
        threshold = 0.5
        logger.info(threshold)
        df = df[df.columns[df.isnull().mean() < threshold]]
        new_rows, new_cols = df.shape
        
        logger.info(f'Dropped {cols-new_cols} columns due to high null counts')

        # Impute missing values in numerical columns with mean
        numerical_cols = df.select_dtypes(include='number').columns
        imputer = SimpleImputer(strategy='mean')
        df[numerical_cols] = imputer.fit_transform(df[numerical_cols].to_numpy())

        # Impute missing values in categorical columns with the most frequent value
        categorical_cols = df.select_dtypes(exclude='number').columns
        imputer = SimpleImputer(strategy='most_frequent')
        df[categorical_cols] = imputer.fit_transform(df[categorical_cols].to_numpy())

        return df

    df = handle_missing_values(df)

    # Handle categorical variables (convert to numerical using one-hot encoding)
    df = pd.get_dummies(df, drop_first=True)

    # Scale numerical features
    numerical_cols = df.select_dtypes(include='number').columns
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols].to_numpy())

    logger.info(f'preprocessed data shape: {df.shape}')
    # Save the cleaned dataset
    df.to_csv(output_path, index=False)
    

if __name__ == '__main__':
    clean_data(config['train_data_file'], config['clean_train_data_filepath'])
    