from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import pandas as pd

class IrisClassifier:
    def __init__(self, random_state=42, **kwargs):
        self.model = LogisticRegression(
            multi_class='multinomial', 
            max_iter=1000, 
            random_state=random_state,
            **kwargs
        )
        self.cv_results = None
        
    def train(self, X_train, y_train):
        """Train the model"""
        self.model.fit(X_train, y_train)
        return self
    
    def cross_validate(self, X, y, cv=5):
        """Perform cross-validation"""
        from .utils import cross_validation
        self.cv_results = cross_validation(self.model, X, y, cv=cv)
        return self.cv_results
    
    def get_cv_summary(self):
        """Get summary of cross-validation results"""
        if self.cv_results is None:
            return None
        
        summary = {
            'mean_accuracy': np.mean(self.cv_results['test_accuracy']),
            'std_accuracy': np.std(self.cv_results['test_accuracy']),
            'mean_precision': np.mean(self.cv_results['test_precision_macro']),
            'mean_recall': np.mean(self.cv_results['test_recall_macro']),
            'mean_f1': np.mean(self.cv_results['test_f1_macro'])
        }
        return summary
    
    def evaluate(self, X_test, y_test, target_names):
        """Evaluate model performance on test set"""
        y_pred = self.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=target_names)
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'classification_report': report,
            'predictions': y_pred
        }
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.model.predict_proba(X)
    
    def predict_new_sample(self, features, scaler, feature_names):
        """Predict a new sample with proper preprocessing"""
        # Convert to DataFrame for feature engineering if needed
        if len(features) == 4 and len(feature_names) > 4:
            # We need to engineer features
            from .feature_engineering import create_engineered_features
            features_df = pd.DataFrame([features], columns=feature_names[:4])
            features_engineered, _ = create_engineered_features(features_df.values, feature_names[:4])
            features = features_engineered[0]
        
        features_scaled = scaler.transform([features])
        prediction = self.predict(features_scaled)[0]
        probabilities = self.predict_proba(features_scaled)[0]
        
        return prediction, probabilities
    
    def get_feature_importance(self, feature_names):
        """Get feature importance coefficients"""
        if hasattr(self.model, 'coef_'):
            importance = {}
            for i, class_name in enumerate(['setosa', 'versicolor', 'virginica']):
                importance[class_name] = dict(zip(feature_names, self.model.coef_[i]))
            return importance
        return None