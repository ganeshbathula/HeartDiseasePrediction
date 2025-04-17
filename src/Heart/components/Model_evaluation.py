import os
import sys
import mlflow
import pickle
import numpy as np
import pandas as pd
import mlflow.sklearn
from urllib.parse import urlparse
from src.Heart.utils.utils import load_object
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from mlflow.models import infer_signature


class ModelEvaluation:
    def __init__(self):
        pass

    def eval_metrics(self, actual, pred):
        accuracy = accuracy_score(actual, pred)
        precision = precision_score(actual, pred)
        recall = recall_score(actual, pred)
        f1 = f1_score(actual, pred)
        conf_matrix = confusion_matrix(actual, pred)
        print("Confusion Matrix:\n", conf_matrix)
        return accuracy, precision, recall, f1

    def initate_model_evaluation(self, train_array, test_array):
        try:
            X_test,y_test=(test_array[:,:-1], test_array[:,-1])
            model_path=os.path.join("Artifacts","model.pkl")
            model=load_object(model_path)

            mlflow.set_registry_uri("http://localhost:5000")
            mlflow.set_tracking_uri("http://127.0.0.1:5000")
            mlflow.set_experiment("ganesh-experiment")

            
            # Sample data
            data = {
                "age": [38],
                "sex": [1],
                "cp": [3],
                "trestbps": [120],
                "chol": [340],
                "fbs": [0],
                "restecg": [0],
                "thalach": [90],
                "exang": [0],
                "oldpeak": [11],
                "slope": [0],
                "ca": [0],
                "thal": [0]
            }

            # Create DataFrame
            input_example = pd.DataFrame(data)

            # Infer the model signature
            signature = infer_signature(input_example)


            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            
            # Make predictions
            prediction = model.predict(input_example)[0]

            # Print the prediction
            print("Predicted Value:", prediction)


            with mlflow.start_run():

                predicted_qualities = model.predict(X_test)

                (accuracy, precision, recall, f1) = self.eval_metrics(y_test,predicted_qualities)
                
                mlflow.log_param("predictions", prediction)

                mlflow.log_metric("Testing Accuracy", accuracy)
                mlflow.log_metric("Precision Score", precision)
                mlflow.log_metric("Recall Score", recall)
                mlflow.log_metric("F1 Score", f1)

                # Model registry does not work with file store
                if tracking_url_type_store != "file":

                    # Register the model
                    # There are other ways to use the Model Registry, which depends on the use case,
                    # please refer to the doc for more information:
                    # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                    mlflow.sklearn.log_model(model, "Model", registered_model_name="ml_model", signature=signature, input_example=input_example)
                else:
                    mlflow.sklearn.log_model(model, "Model")
                
        except Exception as e:
            raise e


