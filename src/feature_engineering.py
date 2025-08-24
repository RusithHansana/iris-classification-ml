import numpy as np
import pandas as pd

def create_engineered_features(X, feature_names):
    """
    Create engineered features from the original features.
    """

    df = pd.DataFrame(X, columns=feature_names)

       # Most impactful features based on correlation analysis
    df['petal_area'] = df['petal length (cm)'] * df['petal width (cm)']
    df['sepal_area'] = df['sepal length (cm)'] * df['sepal width (cm)']
    df['sepal_petal_length_ratio'] = df['sepal length (cm)'] / (df['petal length (cm)'] + 1e-8)
    df['sepal_aspect_ratio'] = df['sepal length (cm)'] / (df['sepal width (cm)'] + 1e-8)

    return df.values, list(df.columns)