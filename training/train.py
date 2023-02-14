import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier

# Splitting the dataframe into test and train data

def split_data(data_df):
    """Split a dataframe into training and validation datasets"""
    print("Columns:", data_df.columns) 
    print("Insurance claim data set dimensions : {}".format(data_df.shape))

    y = data_df.pop('insuranceclaim')
    X_train, X_test, y_train, y_test = train_test_split(data_df, y, test_size = 0.2, random_state = 123)
    data = {"train": {"X": X_train, "y": y_train}, "test": {"X": X_test, "y": y_test}}

    return data


# Train the model, return the model
def train_model(data, parameters):
    """Train a model with the given datasets and parameters"""

    rf_model = RandomForestClassifier(n_estimators = parameters["n_estimators"], criterion = parameters["criterion"], max_depth = parameters["max_depth"])
    print("Training the model...")
    transformer = ColumnTransformer(transformers = [
                                ("one_hot_enc", OneHotEncoder(sparse = False, drop = "first"), 
                                 ["sex", "region"])], remainder = StandardScaler())


    pipeline = Pipeline(steps = [("transformer", transformer), ("model", rf_model)])

    pipeline.fit(data["train"]["X"], data["train"]["y"])

    return pipeline


# Evaluate the metrics for the model
def get_model_metrics(pipeline, data):
    """Construct a dictionary of metrics for the model"""

    preds = pipeline.predict(data["test"]["X"])
    accuracy = accuracy_score(data["test"]["y"], preds)
    precision = precision_score(data["test"]["y"], preds)
    recall = recall_score(data["test"]["y"], preds)
    f1score = f1_score(data["test"]["y"], preds)

    model_metrics = {
        "Accuracy Score": (accuracy),
        "F1 Score": (f1score),
        "Precision Score": (precision),
        "Recall Score": (recall)
             }
    print(model_metrics)

    return model_metrics