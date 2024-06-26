import logging 
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

class Evaluation(ABC):
    
    @abstractmethod
    def calculate_score(self,y_true : np.ndarray, y_pred: np.ndarray):
        
        pass
    

class MSE(Evaluation):
    def calculate_score(self,y_true : np.ndarray, y_pred: np.ndarray):
        try : 
            logging.info("Calculating score MSE")
            mse = mean_squared_error(y_true, y_pred)
            logging.info("MSE : " + str(mse))
            return mse
        except Exception as e:
            logging.error('Error while calculating MSE', e)
            raise e
        
class R2(Evaluation):
    def calculate_score(self,y_true : np.ndarray, y_pred: np.ndarray):
        try : 
            logging.info("Calculating score r2")
            r2 = r2_score(y_true, y_pred)
            logging.info("r2 score :" + str(r2))
            return r2
        except Exception as e:
            logging.error('Error while calculating r2 score', e)
            raise e

class RMSE(Evaluation):
    def calculate_score(self,y_true : np.ndarray, y_pred: np.ndarray):
        try : 
            logging.info("Calculating score RMSE")
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            logging.info("RMSE :"  + str(rmse))
            return rmse
        except Exception as e:
            logging.error('Error while calculating RMSE', e)
            raise e
        
        