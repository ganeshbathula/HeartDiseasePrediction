# HeartDiseasePrediction

## About The Project

Heart disease prediction is a crucial aspect of preventive healthcare that involves the comprehensive analysis of diverse data points to evaluate an individual's susceptibility to cardiovascular diseases. This process integrates demographic details like age and gender with critical clinical information, including medical and family histories, lifestyle choices, and existing health conditions such as hypertension or diabetes. By examining biomarkers like blood pressure, cholesterol levels, and blood sugar, alongside results from medical tests and imaging studies, predictive models can identify patterns and trends indicative of potential heart issues. Machine learning algorithms play a pivotal role in processing this information, helping stratify individuals into risk categories. The ultimate goal is to enable timely interventions and personalized preventive strategies, empowering individuals to make lifestyle adjustments that can mitigate the risk of heart-related events like heart attacks or strokes. Continuous monitoring and updating of predictive models ensure ongoing accuracy and effectiveness in supporting proactive heart health management.

## About the Dataset

This dataset gives information related to heart disease. The dataset contains 13 columns, target is the class variable which is affected by the other 12 columns. Here the aim is to classify the target variable to (disease\non disease) using different machine learning algorithms and find out which algorithm is suitable for this dataset.
<br><be>

<h3>Attributes:</h3> 

 - Age 
 - Gender 
 - Chest Pain Type
 - Resting Blood Pressure
 - Serum Cholesterol 
 - Fasting Blood Sugar 
 - Resting Electrocardiographic Results
 - Maximum Heart Rate Achieved
 - Exercise-induced angina
 - Depression induced by exercise relative to rest
 - Slope of the Peak Exercise ST Segment
 - Number of Major Vessels Colored by Fluoroscopy
 - Thalassemia
 - Target

## Getting Started

This will help you understand how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

## Installation Steps

### Option 1: Installation from GitHub

Follow these steps to install and set up the project directly from the GitHub repository:

1. **Clone the Repository**
   - Open your terminal or command prompt.
   - Navigate to the directory where you want to install the project.
   - Run the following command to clone the GitHub repository:
     ```
     git clone https://github.com/ganeshbathula/HeartDiseasePrediction.git
     ```
2. **Create a Virtual Environment** (Optional but recommended)
   
   - **Python Version: 3.9.13**
   - It's a good practice to create a virtual environment to manage project dependencies. Run the following command:
     ```
     conda create -p <Environment_Name> python==<python version> -y
     ```

4. **Activate the Virtual Environment** (Optional)
   - Activate the virtual environment based on your operating system:
       ```
       conda activate <Environment_Name>/
       ```

5. **Install Dependencies**
   - Navigate to the project directory:
     ```
     cd [project_directory]
     ```
   - Run the following command to install project dependencies:
     ```
     pip install -r requirements.txt
     ```

6. **Run the Project**
   - Start the project by running the appropriate command.
     ```
     python app.py
     ```

7. **Access the Project**
   - Open a web browser or the appropriate client to access the project.
  
## MLFlow steps

Set the path to mlflow.exe and run the below command and execute Training_pipeline.py file to see the experiment details.
```
mlflow server --backend-store-uri sqlite:///mlruns.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000
```

