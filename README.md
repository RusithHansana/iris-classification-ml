# 🌸 Iris Flower Classification

A comprehensive machine learning project that classifies iris flowers into three species using logistic regression with advanced feature engineering capabilities.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Model Details](#model-details)
- [Feature Engineering](#feature-engineering)
- [Results](#results)
- [API Reference](#api-reference)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a machine learning solution to classify iris flowers into three species:

- **Setosa** 🌺
- **Versicolor** 🌸
- **Virginica** 🌷

The classification is based on four morphological features: sepal length, sepal width, petal length, and petal width. The project includes advanced feature engineering capabilities that significantly improve model performance.

## ✨ Features

- **🤖 Logistic Regression Classifier**: Robust multinomial classification
- **⚡ Advanced Feature Engineering**: Creates derived features for better performance
- **📊 Cross-Validation**: Built-in k-fold cross-validation with comprehensive metrics
- **🎯 Real-time Predictions**: CLI interface for instant flower classification
- **📈 Performance Metrics**: Accuracy, precision, recall, F1-score, and confusion matrix
- **💾 Model Persistence**: Save and load trained models with preprocessing pipelines
- **📱 CLI Interface**: Easy-to-use command-line interface
- **🔍 Feature Importance**: Analyze model coefficients and feature contributions

## 📁 Project Structure

```
iris-classification/
│
├── src/                          # Source code
│   ├── __init__.py              # Package initialization
│   ├── main.py                  # CLI entry point and training/prediction logic
│   ├── model.py                 # IrisClassifier class implementation
│   ├── feature_engineering.py   # Feature engineering functions
│   └── utils.py                 # Utility functions (data loading, model saving)
│
├── models/                      # Trained models
│   └── iris_model.pkl          # Saved model with scaler and metadata
│
├── notebooks/                   # Jupyter notebooks
│   └── exploration.ipynb       # Data exploration and analysis
│
├── data/                        # Data directory (uses sklearn built-in dataset)
├── environment.yml              # Conda environment specification
└── README.md                    # This file
```

## 🚀 Installation

### Option 1: Using Conda (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd iris-classification

# Create and activate conda environment
conda env create -f environment.yml
conda activate iris-project
```

### Option 2: Using pip

```bash
# Clone the repository
git clone <repository-url>
cd iris-classification

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn jupyter scipy
```

## ⚡ Quick Start

### 1. Train the Model

```bash
# Train with engineered features (recommended)
python -m src.main --train

# Train with original features only
python -m src.main --train --no-engineer

# Train with custom cross-validation folds
python -m src.main --train --cv 10
```

### 2. Make Predictions

```bash
# Predict iris species for given measurements
python -m src.main --predict 5.1 3.5 1.4 0.2
# Expected: Setosa

python -m src.main --predict 6.7 3.0 5.2 2.3
# Expected: Virginica
```

## 📖 Usage

### Training Options

| Flag            | Description                      | Default |
| --------------- | -------------------------------- | ------- |
| `--train`       | Train the model                  | -       |
| `--no-engineer` | Disable feature engineering      | False   |
| `--cv`          | Number of cross-validation folds | 5       |

### Prediction Format

```bash
python -m src.main --predict [SEPAL_LENGTH] [SEPAL_WIDTH] [PETAL_LENGTH] [PETAL_WIDTH]
```

**Example measurements:**

- **Setosa**: `python -m src.main --predict 5.1 3.5 1.4 0.2`
- **Versicolor**: `python -m src.main --predict 5.9 3.0 4.2 1.5`
- **Virginica**: `python -m src.main --predict 6.5 3.0 5.8 2.2`

## 🤖 Model Details

### Algorithm

- **Logistic Regression** with multinomial classification
- **Solver**: Default (lbfgs for small datasets)
- **Regularization**: L2 (Ridge)
- **Max Iterations**: 1000
- **Random State**: 42 (for reproducibility)

### Data Preprocessing

- **Feature Scaling**: StandardScaler (zero mean, unit variance)
- **Train-Test Split**: 80/20 with stratified sampling
- **Cross-Validation**: Stratified K-Fold

### Performance Metrics

- **Accuracy**: Overall classification accuracy
- **Precision**: Per-class and macro-averaged
- **Recall**: Per-class and macro-averaged
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification breakdown

## 🔧 Feature Engineering

The project includes sophisticated feature engineering that creates additional features from the original measurements:

### Engineered Features

1. **Petal Area**: `petal_length × petal_width`

   - Captures the overall size of the petal

2. **Sepal Area**: `sepal_length × sepal_width`

   - Captures the overall size of the sepal

3. **Sepal-Petal Length Ratio**: `sepal_length ÷ petal_length`

   - Relative proportions between sepal and petal

4. **Sepal Aspect Ratio**: `sepal_length ÷ sepal_width`
   - Shape characteristics of the sepal

### Impact

These engineered features typically improve model performance by:

- Capturing non-linear relationships
- Providing shape and proportion information
- Reducing dimensionality complexity

## 📊 Results

### Typical Performance (with engineered features)

| Metric                        | Score    |
| ----------------------------- | -------- |
| **Test Accuracy**             | ~97-100% |
| **Cross-Validation Accuracy** | ~95-98%  |
| **Precision (Macro Avg)**     | ~97-100% |
| **Recall (Macro Avg)**        | ~97-100% |
| **F1-Score (Macro Avg)**      | ~97-100% |

### Per-Class Performance

- **Setosa**: Nearly perfect classification (100%)
- **Versicolor**: Excellent performance (~95-98%)
- **Virginica**: Excellent performance (~95-98%)

## 🔍 API Reference

### IrisClassifier Class

```python
from src.model import IrisClassifier

# Initialize classifier
classifier = IrisClassifier(random_state=42)

# Train model
classifier.train(X_train, y_train)

# Perform cross-validation
cv_results = classifier.cross_validate(X, y, cv=5)

# Get CV summary
summary = classifier.get_cv_summary()

# Evaluate on test set
results = classifier.evaluate(X_test, y_test, target_names)

# Make predictions
predictions = classifier.predict(X_new)
prediction, probabilities = classifier.predict_new_sample(features, scaler, feature_names)
```

### Utility Functions

```python
from src.utils import load_and_engineer_data, save_model, load_model

# Load and preprocess data
data = load_and_engineer_data(engineer_features=True)

# Save trained model
save_model(model, scaler, feature_names, path="models/my_model.pkl")

# Load saved model
loaded_data = load_model(path="models/my_model.pkl")
```

## 🛠 Development

### Project Setup for Development

```bash
# Clone repository
git clone <repository-url>
cd iris-classification

# Set up environment
conda env create -f environment.yml
conda activate iris-project

# Run in development mode
python -m src.main --help
```

### Running Tests

Currently, the project uses the built-in cross-validation for model validation. To add unit tests:

```bash
# Install pytest
pip install pytest

# Run tests (when implemented)
pytest tests/
```

### Code Structure

- **`src/main.py`**: CLI interface and high-level training/prediction logic
- **`src/model.py`**: Core IrisClassifier implementation
- **`src/utils.py`**: Data loading, model persistence, cross-validation
- **`src/feature_engineering.py`**: Feature engineering functions
- **`notebooks/`**: Jupyter notebooks for exploration and analysis

## 📝 Examples

### Complete Training and Prediction Workflow

```bash
# Step 1: Train the model with engineered features
python -m src.main --train
# Output: Model training progress, CV results, test accuracy

# Step 2: Make predictions for different samples
python -m src.main --predict 5.1 3.5 1.4 0.2
# Output: Setosa with high confidence

python -m src.main --predict 5.9 3.0 4.2 1.5
# Output: Versicolor with probability distribution

python -m src.main --predict 6.5 3.0 5.8 2.2
# Output: Virginica with probability distribution
```

### Expected Output Format

```
🚀 Loading and preprocessing data...
📊 Using engineered features: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)', 'petal_area', 'sepal_area', 'sepal_petal_length_ratio', 'sepal_aspect_ratio']
🤖 Training Logistic Regression model...
📈 Performing cross-validation...

📊 Cross-Validation Results (5-fold):
   Mean Accuracy: 0.9750 (±0.0500)
   Mean Precision: 0.9751
   Mean Recall: 0.9750
   Mean F1-Score: 0.9748

🧪 Evaluating on test set...
✅ Test Set Accuracy: 1.0000

💾 Model saved to models/iris_model.pkl
```

## 🤝 Contributing

Contributions are welcome! Here are some areas for improvement:

1. **Additional Models**: Implement Random Forest, SVM, or Neural Networks
2. **Hyperparameter Tuning**: Add GridSearchCV or RandomizedSearchCV
3. **Data Validation**: Input validation and error handling
4. **Visualization**: Add plotting functions for decision boundaries
5. **Web Interface**: Create a simple web app for predictions
6. **Docker Support**: Containerization for easy deployment

### Contributing Steps

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- **UCI Machine Learning Repository** for the Iris dataset
- **Scikit-learn** for machine learning tools
- **Pandas & NumPy** for data manipulation
- **The Python Community** for excellent documentation and resources

---

**Happy Classifying! 🌸🤖**

For questions or issues, please open an issue on the repository or contact the maintainers.
