import logging
import pandas as pd
from zenml import step
from src.data_cleaning import DataDivideStrategy, DataPreprocessStrategy, DataCleaning
from typing import Annotated
from typing_extensions import Tuple
@step(enable_cache=True)

def clean_data(df : pd.DataFrame) ->Tuple[
    
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"]
] :
    
    try : 
        process_strategy = DataPreprocessStrategy()
        data_cleaning = DataCleaning(df, process_strategy)
        processed_data = data_cleaning.handle_data()
        
        
        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(processed_data, divide_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        
        logging.info("Data cleaning finished")
        return X_train, X_test, y_train, y_test
           
    except  Exception as e:
        logging.error('Error while dividing data', e)
        raise e



