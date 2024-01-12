import logging
import os
import pickle
from datetime import datetime
from pathlib import Path

import missingno as msno
import pandas as pd
from box import ConfigBox
from ruamel.yaml import YAML
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info(f"Started data preprocessing at {datetime.now()}.")

# Constants
NULL_THRESHOLD = 0.10
CARDINALITY_THRESHOLD= 0.80

# Data Loading
def load_data(file_path:str | Path):
    return pd.read_csv(file_path)

# Data Cleaning
def remove_high_null_columns(df, threshold=NULL_THRESHOLD):
    """
    Remove columns with missing values exceeding a threshold.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - threshold (float): Threshold for null values.

    Returns:
    - pd.DataFrame: DataFrame after removing high null columns.
    """
    null_percentage_count = {}
    high_null_counts = []
    
    for column in df.columns:
        x = df[column].isnull().sum() / df[column].shape[0]
        null_percentage_count[column] = x
        if x > threshold:
            high_null_counts.append(column)
        
    df.drop(high_null_counts, axis=1, inplace=True)
    return df

# Remove columns with high cardinality
def remove_high_cardinality_columns(df, threshold=CARDINALITY_THRESHOLD):
    """
    Remove columns with high cardinality.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - threshold (float): Threshold for cardinality.

    Returns:
    - pd.DataFrame: DataFrame after removing high cardinality columns.
    """
    cardinality_percentage_count = {}
    high_cardinal_columns = []
    
    for column in df.columns:
        x = df[column].nunique() / df[column].shape[0]
        cardinality_percentage_count[column] = x
        if x > threshold:
            high_cardinal_columns.append(column)
        
    df.drop(high_cardinal_columns, axis=1, inplace=True)
    return df

# Separate categorical and numerical columns
def get_categorical_numerical_columns(df):
    """
    Separate categorical and numerical columns in a DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.

    Returns:
    - pd.DataFrame: DataFrame containing only categorical columns.
    - pd.DataFrame: DataFrame containing only numerical columns.
    """
    cat_df = df.select_dtypes(exclude='number').columns
    num_df = df.select_dtypes(include='number').columns
    return { 
            'categorical_columns':cat_df, 
            'numerical_columns': num_df,
            }

# Preprocessing
def main_processing(file_path:str, params):
    """
    Main data processing function.

    Parameters:
    - file_path (str): Path to the input CSV file.
    - params (ConfigBox): Configuration parameters.

    Returns:
    - pd.DataFrame: Original DataFrame.
    - pd.DataFrame: Transformed DataFrame.
    - Pipeline: Data processing pipeline.
    """
    original_df = load_data(file_path)
    logger.info(f"Initial Dataframe shape: {original_df.shape}")
    
    # Data Cleaning
    original_df = remove_high_null_columns(original_df, threshold=params.data_cleaning.null_threshold)
    original_df.drop_duplicates(inplace=True)

    
    # Removing high cardinality columns
    # original_df = remove_high_cardinality_columns(original_df, threshold=params.data_cleaning.cardinality_threshold)
    
    # Separate categorical and numerical columns
    columns = get_categorical_numerical_columns(original_df)
    categorical_columns = columns['categorical_columns']
    numerical_columns = columns['numerical_columns']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('impute_null_nums', SimpleImputer(strategy='mean'), numerical_columns),
            ('impute_null_cats', SimpleImputer(strategy='most_frequent'), categorical_columns ),
            ('one_hot_encode', OneHotEncoder(), categorical_columns)
        ],
        remainder='passthrough', 
        verbose_feature_names_out=False,
    )

    # Create a pipeline with the preprocessor
    pipeline = Pipeline([
        ('preprocessor', preprocessor)
    ])
    
    
    # Apply the preprocessing steps to your DataFrame
    df_transformed = pipeline.fit_transform(original_df)

    # Converting all the numpy data to pandas 
    df_transformed = pd.DataFrame(df_transformed, columns=preprocessor.get_feature_names_out())
    logger.info(f'Shape after Data Transformation: {df_transformed.shape} ')

    # Dropping the categorical value that has been one hot encoded
    df_transformed.drop(categorical_columns, axis=1, inplace=True)
    
    logger.info(f"Dropping old categorical columns after OneHotEncoding: {categorical_columns}")
    logger.info(f'Shape after dropping categorical columns: {df_transformed.shape} ')
    
    return original_df, df_transformed, pipeline

# Execution
if __name__ == '__main__':
    yaml = YAML(typ="safe")

    # Load config and params
    params = ConfigBox(yaml.load(open("params.yaml", encoding="utf-8")))
    config = ConfigBox(yaml.load(open("config.yaml", encoding="utf-8")))
    
    # Preprocess the data
    df, df_transformed, pipeline = main_processing(config['train_filepath'], params=params)

    # Save the transformed data
    df_transformed.to_csv(config['preprocessed_train_filepath'], index=False)
    
    # Save the processing pipeline as a pickle 
    with open(config['preprocessing_pipeline'], 'wb') as file:
        pickle.dump(pipeline, file) 
