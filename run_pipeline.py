from pipelines.training_pipeline import train_pipeline
from steps.clean_data import clean_data
from steps.ingest_data import ingest_df
from steps.model_train import train_model
from steps.evaluate import evaluate

from zenml.client import Client



if __name__ == "__main__":
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    train_pipeline(data_path = 'C:/Users/Mahdi/Desktop/workspase/Projects/MLOPS project/data/olist_customers_dataset.csv')
    
    