import os
import sys
import pandas as pd
from src.Heart.logger import logging
from src.Heart.utils.utils import load_object
from src.Heart.exception import customexception
import shap
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid main thread issues

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class PredictPipeline:
    def __init__(self):
        pass

    def getFeaturePlot(self, model, preprocessor):
    # Get feature importance
        importances = model.feature_importances_

        print(":::::importances::::::::", importances)
        feature_names = preprocessor.get_feature_names_out()  # Assuming preprocessor supports this method
        
        # Rename feature names to remove both 'num_' and 'pipeline_' prefixes
        feature_names = [name.replace('num_', '').replace('pipeline_', '').replace('_', '') for name in feature_names]
        print(":::::feature_names::::::::", feature_names)
        
        # Plot feature importance
        
        plt.figure(figsize=(10, 6))
        plt.barh(feature_names, importances, color='skyblue', edgecolor='black')
        plt.gca().invert_yaxis()  # Invert Y-axis to display the most important feature at the top
        plt.xlabel('Feature Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importance Plot')

        # Save plot as image
        output_path = os.path.join("static", "images", "feature_importance.png")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()

    def getSHAPPlot(self, model, preprocessor, scaled_data):
        # Initialize SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(scaled_data)

        # Get feature names and clean them
        feature_names = preprocessor.get_feature_names_out()
        feature_names = [name.replace('num_', '').replace('pipeline_', '').replace('_', '') for name in feature_names]

        # Plot SHAP values
        shap.summary_plot(shap_values, scaled_data, feature_names=feature_names, show=False, plot_type="bar")
        
        # Save SHAP summary plot as an image
        shap_output_path = os.path.join("static", "images", "shap_summary_plot.png")
        os.makedirs(os.path.dirname(shap_output_path), exist_ok=True)
        plt.savefig(shap_output_path, bbox_inches='tight')
        plt.close()
    
    def getPredictionProbability(self, model, scaled_data):
         # Assuming y_pred_proba are your predicted probabilities
            y_pred_proba = model.predict_proba(scaled_data)[:, 1]  # Get probabilities for the positive class
            sns.histplot(y_pred_proba, kde=True)
            plt.xlabel('Predicted Probability')
            plt.ylabel('Frequency')
            plt.title('Prediction Probability Distribution')
            output_path = os.path.join("static", "images", "prediction_distribution.png")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path)
            plt.close()

    def predict(self,features):
        try:
            preprocessor_path = os.path.join("Artifacts", "Preprocessor.pkl")
            model_path = os.path.join("Artifacts", "Model.pkl")
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)
            scaled_data = preprocessor.transform(features)
            pred = model.predict(scaled_data)
            
                        
            # Assuming accuracy is calculated
            accuracy = 0.86 # Example accuracy value

            # Create a bar plot for accuracy
            plt.figure(figsize=(6, 4))
            plt.bar(['Accuracy'], [accuracy], color='blue')
            plt.ylim(0, 1)
            plt.ylabel('Accuracy')
            plt.title('Model Accuracy: Random Forest Classifier')

            # Save plot as image
            output_path = os.path.join("static", "images", "accuracy_plot.png")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path)
            plt.close()

           

            self.getPredictionProbability(model, scaled_data)

            self.getSHAPPlot(model, preprocessor, scaled_data)

            self.getFeaturePlot(model, preprocessor)
            
            return pred

        except Exception as e:
            raise customexception(e, sys)
    
class CustomData:
    def __init__(self,
                 age:int,
                 sex:int,
                 cp:int,
                 trestbps:int,
                 chol:int,
                 fbs:int,
                 restecg:int,
                 thalach:int,
                 exang:int,
                 oldpeak:float,
                 slope:int,
                 ca:int,
                 thal:int):
        
        self.age = age
        self.sex = sex
        self.cp = cp
        self.trestbps = trestbps
        self.chol = chol
        self.fbs = fbs
        self.restecg = restecg
        self.thalach = thalach
        self.exang = exang
        self.oldpeak = oldpeak
        self.slope = slope
        self.ca = ca
        self.thal = thal
            
                
    def get_data_as_dataframe(self):
            try:
                custom_data_input_dict = {
                    'age':[self.age],
                    'sex':[self.sex],
                    'cp':[self.cp],
                    'trestbps':[self.trestbps],
                    'chol':[self.chol],
                    'fbs':[self.fbs],
                    'restecg':[self.restecg],
                    'thalach':[self.thalach],
                    'exang':[self.exang],
                    'oldpeak':[self.oldpeak],
                    'slope':[self.slope],
                    'ca':[self.ca],
                    'thal':[self.thal]
                }
                df = pd.DataFrame(custom_data_input_dict)
                print(df)
                logging.info('Dataframe Gathered')
                return df
            except Exception as e:
                logging.info('Exception Occured in prediction pipeline')
                raise customexception(e,sys)
            
   