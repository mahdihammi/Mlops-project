import logging
import pandas as pd 
from zenml import step


class IngestData:
    def __init__(self, data_path):
        self.data_path = data_path
        
    def get_data(self):
        logging.info(f'Ingesting data from {self.data_path}')
        return pd.read_csv(self.data_path)
    

@step 
def ingest_df(data_path: str) -> pd.DataFrame:
    """
    Ingesting data from data path
    
    """
    try :
        ingest_data = IngestData(data_path)
        df = ingest_data.get_data()
        logging.info("Ingested data")
        
        return df
    except Exception as e:
        logging.error('Error while ingesting', e)
        raise e
    
