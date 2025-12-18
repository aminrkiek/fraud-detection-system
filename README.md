# Financial Fraud Detection System

Advanced machine learning system for real-time fraud detection in financial transactions using XGBoost, achieving 97.5% precision with minimal false positives.

## Business Case

Credit card fraud costs the financial industry billions annually. This system addresses the challenge by:
- Detecting 74% of fraudulent transactions
- Maintaining 97.5% precision (only 2 false positives per 85,443 transactions)
- Reducing false positive rate to 0.002%
- Estimated savings: $9,750 per batch analyzed

## Live Demo

Interactive dashboard: https://858eea4719aa.ngrok-free.app/

## Key Results

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.960 |
| Precision | 97.56% |
| Recall | 74.07% |
| F1-Score | 0.842 |
| False Positive Rate | 0.002% |

## Technical Architecture

### Models Implemented

1. **Isolation Forest** (Unsupervised)
   - Detects outliers without labeled data
   - ROC-AUC: 0.948

2. **Autoencoder** (Deep Learning)
   - Neural network for anomaly detection
   - ROC-AUC: 0.955

3. **XGBoost** (Supervised - Production Model)
   - Gradient boosting with SMOTE balancing
   - Best performance: ROC-AUC 0.960

### Stack

**ML/Data Science:**
- Python 3.10+
- XGBoost, Scikit-learn
- TensorFlow/Keras
- SHAP (explainability)
- Imbalanced-learn

**Visualization:**
- Streamlit
- Plotly
- Seaborn

**Data Processing:**
- Pandas, NumPy
- Feature engineering pipeline

## Project Structure
```
fraud-detection-system/
├── data/
│   ├── raw/                      # Original dataset
│   └── processed/                # Processed features and models
├── notebooks/
│   └── 01_exploratory_analysis.ipynb
├── src/
│   ├── feature_engineering.py    # Feature creation pipeline
│   ├── models/                   # Model training scripts
│   └── dashboard.py              # Streamlit dashboard
├── README.md
└── requirements.txt
```

## Features Engineering

Created 47 features from original 31:

**Temporal Features:**
- Hour of day, day of week
- Time period categorization

**Amount Features:**
- Log transformation
- Robust scaling
- Category buckets

**Statistical Features:**
- Mean, std, range of PCA components
- Absolute sum of values

## Model Performance Comparison

| Model | Precision | Recall | F1-Score | False Positives |
|-------|-----------|--------|----------|-----------------|
| Isolation Forest | 0.91% | 1.85% | 1.22% | 217 |
| Autoencoder | 10.09% | 52.78% | 16.94% | 508 |
| **XGBoost** | **97.56%** | **74.07%** | **84.21%** | **2** |

## Explainability

Implemented SHAP (SHapley Additive exPlanations) for model interpretability:
- Global feature importance
- Individual prediction explanations
- Compliance-ready decision justification

## Installation
```bash
git clone https://github.com/aminrkiek/fraud-detection-system
cd fraud-detection-system

pip install -r requirements.txt
```

## Usage

### Training Models
```python
from src.feature_engineering import main
X_train, X_test, y_train, y_test = main()
```

### Making Predictions
```python
from src.predict import FraudDetector

detector = FraudDetector()
result = detector.predict(transaction)

print(f"Fraud Score: {result['fraud_score']:.4f}")
print(f"Recommendation: {result['recommendation']}")
```

### Running Dashboard
```bash
streamlit run src/dashboard.py
```

## Dataset

Credit Card Fraud Detection dataset from Kaggle (284,807 transactions)
- Highly imbalanced: 0.17% fraud rate (1:577 ratio)
- 28 PCA-transformed features (privacy protection)
- 2 days of transaction data

## Key Insights

1. **Class Imbalance Handling**: SMOTE oversampling improved recall significantly
2. **Threshold Optimization**: Custom threshold (not 0.5) maximized F1-score
3. **Feature Importance**: V14, V10, V17 are top discriminators
4. **Business Impact**: $122 average fraud amount vs $5 false positive review cost

## Future Enhancements

- Real-time streaming integration
- Automated model retraining pipeline
- A/B testing framework
- Multi-model ensemble
- API deployment (FastAPI)

## Author

**Amine Rkiek** - Data Scientist | Financial Analytics Specialist

- LinkedIn: https://www.linkedin.com/in/amine-rkiek-86871520b/
- E-mail: aminrkiek@gmail.com
- Github: github.com/aminrkiek

## License

This project is licensed under the MIT License.

## Acknowledgments

- Dataset: Kaggle Credit Card Fraud Detection
- Inspired by real-world fraud detection challenges in banking
