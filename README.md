# Predicting how a customer will feel about a product before they ordered it
For a given customer's historical data, we are tasked to predict the review score for the next order or purchase. We will be using the Brazilian E-Commerce Public Dataset by Olist. This dataset has information on 100,000 orders from 2016 to 2018 made at multiple marketplaces in Brazil. Its features allow viewing charges from various dimensions.
 In order to achieve this in a real-world scenario, we will be using ZenML to build a production-ready pipeline to predict the customer satisfaction score for the next order or purchase.
#Training Pipeline
Our standard training pipeline consists of several steps:
- ingest_data: This step will ingest the data and create a DataFrame.
- clean_data: This step will clean the data and remove the unwanted columns.
- train_model: This step will train the model and save the model using MLflow autologging.
- evaluation: This step will evaluate the model and save the metrics -- using MLflow autologging -- into the artifact store.

The picture below shows zenml dashboards for tracking pipelines with versions and with DAG visualizers 
![zenml pipeline](https://github.com/mahdihammi/Mlops-project/assets/89527502/ed9ce127-fe7f-47ce-ad6a-be96c02f7e2d)   ![zenml pipeline 2](https://github.com/mahdihammi/Mlops-project/assets/89527502/f4beaa14-067d-436b-929e-9de513659232)
