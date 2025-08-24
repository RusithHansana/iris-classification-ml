import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import joblib

def load_and_engineer_data(test_size=0.2, random_state=42, engineer_features=True):
    """
    Load the Iris dataset and optionally engineer features.
    """

    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names

    if engineer_features:
        from .feature_engineering import create_engineered_features
        X, feature_names = create_engineered_features(X, feature_names)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_names,
        'target_names': target_names,
        'scaler': scaler,
        'engineered': engineer_features
    }

def cross_validation(model, X, y, cv=5):
    """
    Perform cross-validation on the given model using the provided features and target.
    """
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    cv_results = cross_validate(model, X, y, cv=cv_strategy, scoring=scoring, return_train_score=True)
    return cv_results

def save_model(model, scaler, feature_names, path="models/iris_model.pkl"):
    """
    Save the trained model, scaler, feature names, and target names to disk.
    """
    joblib.dump({
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
    }, path)

def load_model(path="models/iris_model.pkl"):
    """
    Load the trained model, scaler, feature names, and target names from disk.
    """
    return joblib.load(path)
