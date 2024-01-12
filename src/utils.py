import time
from functools import partial, wraps

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def timeit(func, logger=None, show_args_and_result=False):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        if logger:
            logger.info(f'{func.__name__} took {end - start: .5f} seconds')
            if show_args_and_result:
                logger.info(f'Arguments: {args}')
                logger.info(f'Kwargs: {kwargs}')
                logger.info(f'Returned: {result}')
        else:
            print(f'{func.__name__} took {end - start: .5f} seconds')
            if show_args_and_result:
                print(f'Arguments: {args}')
                print(f"Kwargs: {kwargs}")
                print(f'Returned: {result}')
            
        return result
    return wrapper

# get all metrics
def get_score(ytest, y_pred):
    try:
        mse = mean_squared_error(ytest, y_pred)
    except:
        mse = np.nan
        
    try:
        mae = mean_absolute_error(ytest, y_pred),
    except:
        mae = np.nan
    
    try:
        r2 = r2_score(ytest, y_pred)
    except:
        r2 = np.nan
    finally:
        return {
            'mse':mse,
            'mae':mae,
            'r2': r2
        }