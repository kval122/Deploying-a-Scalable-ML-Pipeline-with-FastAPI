import pytest
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from ml.model import train_model, compute_model_metrics
from ml.data import process_data
import numpy as np

# Sample data for testing
data = pd.DataFrame({
    "age": [25, 32, 47],
    "workclass": ["Private", "Self-emp-not-inc", "Private"],
    "education": ["Bachelors", "HS-grad", "Masters"],
    "marital-status": ["Never-married", "Married-civ-spouse", "Divorced"],
    "occupation": ["Tech-support", "Exec-managerial", "Prof-specialty"],
    "relationship": ["Not-in-family", "Husband", "Unmarried"],
    "race": ["White", "Black", "Asian-Pac-Islander"],
    "sex": ["Male", "Female", "Male"],
    "native-country": ["United-States", "United-States", "India"],
    "salary": ["<=50K", ">50K", ">50K"]
})

categorical_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


# Test 1: Check if the model returns the expected type of result
def test_apply_labels():
    """
    Verify if process_data returns the expected data types.
    """
    X, y, _, _ = process_data(data, categorical_features=categorical_features, label="salary", training=True)
    assert isinstance(X, np.ndarray), "Processed X is not a NumPy array"
    assert isinstance(y, np.ndarray), "Processed y is not a NumPy array" 


# Test 2: Check if the model uses the expected algorithm
def test_train_model():
    """
    Test that train_model returns an instance of RandomForestClassifier.
    """
    X, y, _, _ = process_data(data, categorical_features=categorical_features, label="salary", training=True)
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier), "The model is not of type RandomForestClassifier"


# Test 3: Verify computing metrics function returns expected values
def test_compute_model_metrics():
    """
    Test that compute_model_metrics returns metrics as a tuple of three floats.
    """
    X, y, _, _ = process_data(data, categorical_features=categorical_features, label="salary", training=True)
    model = train_model(X, y)
    preds = model.predict(X)
    precision, recall, f1 = compute_model_metrics(y, preds)
    assert isinstance(precision, float), "Precision is not a float"
    assert isinstance(recall, float), "Recall is not a float"
    assert isinstance(f1, float), "F1 score is not a float"
