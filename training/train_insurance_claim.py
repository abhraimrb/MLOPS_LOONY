import pickle
import os
import numpy as np
import pandas as pd
import json
import subprocess

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
from typing import Tuple, List

import matplotlib.pyplot as plt
import seaborn as sns

from azureml.core import Workspace, Datastore, Dataset
from azureml.core.run import Run
from azureml.core.model import Model

run = Run.get_context()
exp = run.experiment
ws = run.experiment.workspace

print("Loading training data...")
datastore = ws.get_default_datastore()
datastore_paths = [(datastore, 'insurance/insurance_claim_data.csv')]
traindata = Dataset.Tabular.from_delimited_files(path=datastore_paths)
insurance_claim_data = traindata.to_pandas_dataframe()
print("Columns:", insurance_claim_data.columns) 
print("Diabetes data set dimensions : {}".format(insurance_claim_data.shape))

y = insurance_claim_data.pop('insuranceclaim')
X_train, X_test, y_train, y_test = train_test_split(insurance_claim_data, y, test_size=0.2, random_state=123)
data = {"train": {"X": X_train, "y": y_train}, "test": {"X": X_test, "y": y_test}}

print("Training the model...")
transformer = ColumnTransformer(transformers=[
                                ("one_hot_enc", OneHotEncoder(sparse=False, drop="first"), 
                                 ["sex", "smoker","region"])
    
                                ], remainder=StandardScaler())

model = RandomForestClassifier()

pipeline = Pipeline(steps=[("transformer", transformer), ("model", model)])

pipeline.fit(X_train, y_train)

print("Evaluate the model...")
preds = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, pipeline.predict(X_test))
precision = precision_score(y_test, pipeline.predict(X_test))
recall = recall_score(y_test, pipeline.predict(X_test))

print("Accuracy Score: ", accuracy)
print("F1 Score: ", f1_score)
print("Precision Score: ", precision)
print("Recall Score: ", recall)
run.log("Accuracy Score: ", accuracy)
run.log("F1 Score: ", f1_score)
run.log("Precision Score: ", precision)
run.log("Recall Score: ", recall)

# Save model as part of the run history
print("Exporting the pipeline as pickle file...")
outputs_folder = './rfc_pipeline'
os.makedirs(outputs_folder, exist_ok=True)

model_filename = "insurance_claim_prediction_model.pkl"
model_path = os.path.join(outputs_folder, model_filename)
dump(pipeline, model_path)

# upload the model file explicitly into artifacts
print("Uploading the model into run artifacts...")
run.upload_file(name="./outputs/models/" + model_filename, path_or_stream=model_path)
print("Uploaded the model {} to experiment {}".format(model_filename, run.experiment.name))
dirpath = os.getcwd()
print(dirpath)
print("Following files are uploaded ")
print(run.get_file_names())

run.complete()