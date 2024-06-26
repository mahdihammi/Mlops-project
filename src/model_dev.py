import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

class Model(ABC):
    
    @abstractmethod
    def train(self, X_train, y_train):
        pass
    

class LinearRegressionModel(Model):
    
    def train(self, X_train, y_train, **kwargs):
        try:
            reg = LinearRegression(**kwargs)
            reg.fit(X_train, y_train)
            return reg
        except Exception as e:
            logging.error('Error while training model', e)
            raise e
        
    # For linear regression, there might not be hyperparameters that we want to tune, so we can simply return the score
    def optimize(self, trial, x_train, y_train, x_test, y_test):
        reg = self.train(x_train, y_train)
        return reg.score(x_test, y_test)
        
class RandomForestModel(Model):
    """
    RandomForestModel that implements the Model interface.
    """

    def train(self, x_train, y_train, **kwargs):
        reg = RandomForestRegressor(**kwargs)
        reg.fit(x_train, y_train)
        return reg
    
    def optimize(self, trial, x_train, y_train, x_test, y_test):
        n_estimators = trial.suggest_int("n_estimators", 1, 200)
        max_depth = trial.suggest_int("max_depth", 1, 20)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        reg = self.train(x_train, y_train, n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split)
        return reg.score(x_test, y_test)

