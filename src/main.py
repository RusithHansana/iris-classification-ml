"""
Iris Flower Classification CLI Program
"""

import argparse
import numpy as np
import pandas as pd
from .utils import load_and_engineer_data, save_model, load_model
from .model import IrisClassifier

def train_model(engineer_features=True, cv_folds=5):
    """Train and save the model"""
    print("🚀 Loading and preprocessing data...")
    data = load_and_engineer_data(engineer_features=engineer_features)
    
    if data['engineered']:
        print(f"📊 Using engineered features: {data['feature_names']}")
    else:
        print("📊 Using original features")
    
    print("🤖 Training Logistic Regression model...")
    classifier = IrisClassifier()
    classifier.train(data['X_train'], data['y_train'])
    
    print("📈 Performing cross-validation...")
    classifier.cross_validate(data['X_train'], data['y_train'], cv=cv_folds)
    cv_summary = classifier.get_cv_summary()
    
    print(f"\n📊 Cross-Validation Results ({cv_folds}-fold):")
    print(f"   Mean Accuracy: {cv_summary['mean_accuracy']:.4f} (±{cv_summary['std_accuracy'] * 2:.4f})")
    print(f"   Mean Precision: {cv_summary['mean_precision']:.4f}")
    print(f"   Mean Recall: {cv_summary['mean_recall']:.4f}")
    print(f"   Mean F1-Score: {cv_summary['mean_f1']:.4f}")
    
    print("🧪 Evaluating on test set...")
    results = classifier.evaluate(data['X_test'], data['y_test'], data['target_names'])
    
    print(f"\n✅ Test Set Accuracy: {results['accuracy']:.4f}")
    print("\n📝 Classification Report:")
    print(results['classification_report'])
    
    # Show feature importance
    importance = classifier.get_feature_importance(data['feature_names'])
    if importance:
        print("\n🎯 Feature Importance Coefficients:")
        for class_name, coefs in importance.items():
            print(f"   {class_name.capitalize()}:")
            for feature, coef in coefs.items():
                print(f"     {feature}: {coef:+.4f}")
    
    # Save the trained model
    save_model(classifier.model, data['scaler'], data['feature_names'])
    print(f"\n💾 Model saved to models/iris_model.pkl")
    
    return classifier

def predict_sample(sepal_length, sepal_width, petal_length, petal_width):
    """Predict a single sample"""
    try:
        print(f"🔍 Predicting for: SL={sepal_length}, SW={sepal_width}, PL={petal_length}, PW={petal_width}")
        
        # Load model and scaler
        loaded_data = load_model()
        model = loaded_data['model']
        scaler = loaded_data['scaler']
        feature_names = loaded_data['feature_names']
        
        # Create classifier instance with loaded model
        classifier = IrisClassifier()
        classifier.model = model
        
        # Prepare features
        features = [sepal_length, sepal_width, petal_length, petal_width]
        
        # Predict
        prediction, probabilities = classifier.predict_new_sample(features, scaler, feature_names)
        
        target_names = ['setosa', 'versicolor', 'virginica']
        
        print(f"\n🎯 Prediction Results:")
        print(f"   Predicted class: {target_names[prediction]}")
        print("   Prediction probabilities:")
        for i, prob in enumerate(probabilities):
            print(f"     {target_names[i]}: {prob:.4f}")
            
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        print("💡 Make sure to train the model first with: python -m src.main --train")

def main():
    parser = argparse.ArgumentParser(
        description='Iris Flower Classification with Logistic Regression',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python -m src.main --train                 # Train model with engineered features
  python -m src.main --train --no-engineer   # Train with original features only
  python -m src.main --predict 5.1 3.5 1.4 0.2  # Predict a sample
  python -m src.main --predict 6.7 3.0 5.2 2.3  # Predict another sample
        '''
    )
    
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--no-engineer', action='store_false', dest='engineer', 
                       help='Disable feature engineering')
    parser.add_argument('--cv', type=int, default=5, help='Number of cross-validation folds')
    parser.add_argument('--predict', nargs=4, type=float, 
                       metavar=('SEPAL_L', 'SEPAL_W', 'PETAL_L', 'PETAL_W'),
                       help='Predict class for given measurements')
    
    args = parser.parse_args()
    
    if args.train:
        train_model(engineer_features=args.engineer, cv_folds=args.cv)
    elif args.predict:
        predict_sample(*args.predict)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()