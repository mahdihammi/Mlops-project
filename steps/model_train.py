import logging
import pandas as pd
from zenml import step 
from src.model_dev import LinearRegressionModel, RandomForestModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig
import mlflow
from sklearn.ensemble import RandomForestRegressor

from zenml.client import Client




experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def train_model(X_train : pd.DataFrame,
                X_test : pd.DataFrame, 
                y_train : pd.DataFrame ,
                y_test : pd.DataFrame,
                config : ModelNameConfig,
                ) -> RegressorMixin:
    try : 
        if config.model_name == "LinearRegression":
            mlflow.sklearn.autolog()
            model = LinearRegressionModel()
            trained_model = model.train(X_train, y_train)
            logging.info("Model trained")
            return trained_model
        elif config.model_name == "RandomForestRegressor" :
            mlflow.sklearn.autolog()
            model = RandomForestModel() 
            trained_model = model.train(X_train, y_train)
            return trained_model
        else:
            raise ValueError(f"Model {config.model_name} is not a Supported")
        
    except Exception as e:
        logging.error('Error while training model', e)
        raise e
    
    