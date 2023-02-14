import numpy as np
import pandas as pd

# functions to test are imported from train.py
from train import split_data, train_model, get_model_metrics

"""A set of simple unit tests for protecting against regressions in train.py"""


def test_split_data():
    test_data = {
        "insuranceclaim": [0, 0, 1, 0, 1],
        "col1": [10, 20, 30, 4, 5],
        "col2": [6, 7, 1, 13, 1]
        }

    data_df = pd.DataFrame(data=test_data)
    data = split_data(data_df)

    # verify that columns were removed correctly
    assert "insuranceclaim" not in data["train"]["X"].columns
    assert "col1" in data["train"]["X"].columns

    # verify that data was split as desired
    assert data["train"]["X"].shape == (4, 2)
    assert data["train"]["X"].shape == (1, 2)


def test_train_model():
    data = __get_test_datasets()

    params = {
        "n_estimators": 50,
        "criterion": "gini",
        "max_depth": 8
    }

    model = train_model(data, params)

    # verify that parameters are passed in to the model correctly
    for param_name in params.keys():
        assert param_name in model.named_steps["model"].get_params()
        assert params[param_name] == model.named_steps["model"].get_params()[param_name]


def test_get_model_metrics():
    class MockModel:

        @staticmethod
        def predict(data):
            return np.array([0, 0])

    data = __get_test_datasets()

    metrics = get_model_metrics(MockModel(), data)

    # verify that metrics is a dictionary containing the accuracy score.
    assert "Accuracy Score" in metrics
    accuracy = metrics["Accuracy Score"]
    np.testing.assert_almost_equal(accuracy, 0.5)


def __get_test_datasets():
    """This is a helper function to set up some test data"""
    X_train = pd.DataFrame(np.array([[1, 2, 3, 4, 5, 6, 7],[7, 8, 9, 10, 11, 12, 13]]).reshape(-1, 7), columns = ["age", "sex", "bmi", "children", "smoker", "region", "charges"])
    y_train = np.array([[1], [1]]).reshape(-1, 1)
    X_test = pd.DataFrame(np.array([[7, 8, 9, 10, 11, 12, 13],[7, 80, 90, 100, 110, 120, 130]]).reshape(-1, 7), columns = ["age", "sex", "bmi", "children", "smoker", "region", "charges"])
    y_test = np.array([[0], [1]]).reshape(-1, 1)

    data = {"train": {"X": X_train, "y": y_train}, "test": {"X": X_test, "y": y_test}}
    return data