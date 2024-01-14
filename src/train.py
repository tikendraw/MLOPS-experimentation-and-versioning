import logging
import pickle
from datetime import datetime
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from box import ConfigBox
from funcyou.utils import DotDict
from ruamel.yaml import YAML
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.linear_model import (
    GammaRegressor,
    PassiveAggressiveRegressor,
    PoissonRegressor,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from utils import get_score, timeit

# Load the configuration file
config_path = Path.cwd() / 'config.yaml'


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info(f"Started data preprocessing at {datetime.now()}.")

# read the data

# Data Loading
def load_data(file_path:str | Path):
    return pd.read_csv(file_path)


def get_model(model_params):
    
    
    return VotingRegressor(
    estimators=[
        ('gradient_boosting_regression',GradientBoostingRegressor()), 
        ('random_forest_regression',RandomForestRegressor()), 
        ('poisson_regression',PoissonRegressor()), 
        ('gamma_regression',GammaRegressor()), 
        ('passive_aggressive_regression',PassiveAggressiveRegressor())
        ],
    n_jobs=-1
    )
    

@timeit
def train(model_params:DotDict, X:pd.DataFrame, y:pd.DataFrame) -> tuple[RegressorMixin]:
    model = get_model(model_params=model_params)
    pipeline = Pipeline([
        ('scaler',StandardScaler()),
        ('voting_regressor', model)
    ])

    pipeline.fit(X, y=y)
    
    return pipeline

# make a sklearn pipe line to predict house price     model_params= get_model_params()


# Execution
if __name__ == '__main__':
    yaml = YAML(typ="safe")
    config_path = "config.yaml"
    params_path = "params.yaml"
    # Load config and params
    config = DotDict.from_yaml(config_path)
    params = DotDict.from_yaml(params_path)
    
    target_column = params.base.target
    model_params = params.model.params

    # Preprocess the data    
    df = load_data(file_path=config.preprocessed_train_filepath)
    
    # Split the data
    xtrain, xtest, ytrain, ytest = train_test_split(
        df.drop(target_column, axis = 1),
        df[target_column],
        train_size=params.data_split.train_size,
        test_size =params.data_split.test_size,
        random_state = params.data_split.random_seed,    
        )
    
    pipeline = train(model_params=model_params, X=xtrain, y=ytrain)
    
    from dvclive import Live

    y_train_pred = pipeline.predict(xtrain)
    y_test_pred = pipeline.predict(xtest)
    
    train_scores = get_score(ytest=ytrain, y_pred=y_train_pred)
    test_scores = get_score(ytest=ytest, y_pred=y_test_pred)

    with Live(save_dvc_exp=True) as live:
        live.log_metric("train score", pipeline.score(xtrain, ytrain))
        live.log_metric("train mse", train_scores['mse'])
        live.log_metric("train mae", train_scores["mae"])
        live.log_metric("train r2 score", train_scores["r2"])
            
        live.log_metric("test score", pipeline.score(xtest, ytest))
        live.log_metric("test mse", test_scores['mse'])
        live.log_metric("test mae", test_scores["mae"])
        live.log_metric("test r2 score", test_scores["r2"])
            

    

    # Save the model pipeline as a pickle 
    with open(config.model_pipeline, 'wb') as file:
        pickle.dump(pipeline, file) 
