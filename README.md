# diabetes-readmission

A machine learning pipeline for predicting hospital readmission within 30 days for diabetes patients using clinical and demographic data from the UCI Diabetes 130-US Hospitals dataset.

## Overview

This project implements a complete classification pipeline that:
- Downloads and preprocesses the UCI diabetes dataset (130 US hospitals, 1999-2008)
- Performs comprehensive feature engineering with numeric and categorical features
- Trains multiple classifiers (Logistic Regression, Random Forest)
- Prioritizes recall to identify high-risk patients at risk of 30-day readmission
- Provides threshold tuning strategies for operational decision-making
- Generates calibration curves and precision-recall analysis

## Clinical Motivation

Unplanned hospital readmission within 30 days is costly and indicates suboptimal care. Early identification of high-risk patients enables targeted interventions:
- Post-discharge follow-up calls
- Medication reconciliation programs
- Scheduling outpatient appointments
- Care coordination with primary care providers

This model prioritizes **recall** to ensure high-risk patients are not missed, even at the cost of some false positives.

## Dataset

**Source**: UCI Machine Learning Repository - Diabetes 130-US Hospitals (1999-2008)  
**Size**: ~100K patient encounters from 130 US hospitals  
**Records**: Hospital readmission data with 50+ clinical and demographic features

**Target Variable**:  
- `readmit_30`: Binary classification (1 if readmitted within 30 days, 0 otherwise)
- Class distribution: Imbalanced (13-15% readmission rate)

**Data Sources** (tried in order):
1. Kaggle API (if configured with valid credentials)
2. UCI official download link
3. Local CSV file at `data/diabetes_ucireadmit/diabetic_data.csv`

## Key Features

### Feature Engineering
- **Numeric Features**: Time in hospital, lab procedures, medications, visits, diagnoses count
- **Categorical Features**: Race, gender, age group, admission type, discharge disposition, medical specialty, diagnosis codes, medication changes
- **Automatic Preprocessing**: OneHot encoding for categorical features, median imputation for numeric
- **Class Imbalance Handling**: Stratified train/test split and balanced class weights

### Machine Learning Models
1. **Logistic Regression**
   - Balanced class weights to handle imbalance
   - L2 regularization via liblinear solver
   - Output: Probability scores for threshold tuning

2. **Random Forest Classifier**
   - 200 estimators with balanced class weights
   - Feature importance extraction
   - Better handling of non-linear relationships

### Evaluation Metrics
- **Confusion Matrix**: TP, FP, TN, FN breakdown
- **Classification Report**: Precision, Recall, F1-score per class
- **Precision-Recall Curve**: Visualizes precision-recall tradeoff
- **Calibration Curves**: Model confidence vs actual probabilities
- **Threshold Tuning**: Find optimal decision threshold for recall maximization

## Project Structure

```
diabetes-readmission/
├── README.md                          # This file
├── diabetes_readmission_pipeline.py   # Main analysis pipeline
└── data/
    └── diabetes_ucireadmit/            # Data directory
        └── diabetic_data.csv          # UCI dataset (auto-downloaded)
```

## Requirements

Python 3.7+

Key Dependencies:
- numpy - Numerical computations
- pandas - Data manipulation and analysis
- scikit-learn - Machine learning models and preprocessing
- matplotlib - Visualization
- seaborn - Statistical visualizations
- requests - Data downloading
- joblib - Model persistence
- kaggle (optional) - Kaggle API for dataset download

Install dependencies:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn requests joblib
```

Optional Kaggle API:
```bash
pip install kaggle
# Configure: place kaggle.json in ~/.kaggle/
```

## Usage

### Automatic Data Download
The script automatically tries to download the dataset in this order:
1. **Kaggle API**: If `kaggle` is installed and configured
2. **UCI Download**: If Kaggle fails, downloads from official UCI link
3. **Local File**: If remote downloads fail, looks for local CSV

### Run the Pipeline
```bash
python diabetes_readmission_pipeline.py
```

### Output
The script generates:
- **Console Output**:
  - Data shape and column info
  - Target distribution (readmission rates)
  - Train/test split statistics
  - Model performance metrics (Precision, Recall, F1-score)
  - Classification reports
  - Best threshold recommendations
  - Top feature importances (Random Forest)

- **Visualizations**:
  - Calibration curves (model probability reliability)
  - Precision-Recall curves
  - Feature importance bar charts

- **Saved Models**:
  - `models/diabetes_rf_pipeline.joblib` - Chosen model (prioritizes recall)
  - `models/diabetes_lr_pipeline.joblib` - Logistic Regression pipeline

## Understanding the Results

### Key Performance Metrics

**Recall (Sensitivity)**: Proportion of actual readmissions correctly identified
- Higher recall = fewer high-risk patients missed
- Critical in healthcare to prevent missed interventions

**Precision**: Proportion of predicted readmissions that are correct
- Impacts outreach resource allocation and patient burden

**Calibration**: Model prediction probabilities vs actual outcomes
- Well-calibrated models allow trustworthy risk scores
- Guides threshold selection for clinical decision-making

### Threshold Tuning

The pipeline searches for optimal thresholds to maximize recall:
- Default threshold: 0.5
- Tuned threshold: Found value that maximizes recall on test set
- Trade-off: Higher recall = more false positives = increased intervention volume

### Interpreting Feature Importance

Random Forest feature importances show which factors most influence readmission risk:
- Higher values = stronger association with readmission
- Use to understand clinical drivers and target interventions
- Example: If "number of medications" is high importance, medication reconciliation becomes a priority

## Clinical Applications

1. **Risk Stratification**: Classify patients into risk groups at discharge
2. **Resource Allocation**: Focus limited intervention capacity on highest-risk patients
3. **Care Planning**: Tailor discharge planning based on predicted risk
4. **Quality Monitoring**: Track readmission prevention program effectiveness

## Model Selection Strategy

The pipeline chooses the model (Logistic Regression or Random Forest) that achieves the highest recall at its optimal threshold. This prioritizes:
- **Sensitivity**: Catching high-risk readmission cases
- **Operability**: Logistic Regression model also saved for interpretability

## Future Enhancements

- [ ] Temporal modeling (patient readmission history)
- [ ] Ensemble methods combining multiple model types
- [ ] Integration with EHR systems for real-time scoring
- [ ] Patient-level explanations (SHAP values)
- [ ] Cost-sensitive learning (false positive vs false negative costs)
- [ ] Continuous model monitoring and retraining

## Author

Ritvik(@ritvikvr)

## License

MIT License - Feel free to use, modify, and distribute

## References

- UCI Diabetes 130-US Hospitals dataset: https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008
- Kaggle mirror: https://www.kaggle.com/datasets
- scikit-learn pipeline documentation: https://scikit-learn.org/stable/modules/compose.html

## Disclaimer

This model is a demonstration project for educational purposes. In production healthcare settings:
- Always validate on your institutional data
- Integrate with clinical workflows and EHR systems
- Obtain necessary regulatory approvals (FDA, institutional review)
- Monitor for model drift and performance degradation
- Combine with clinical judgment from healthcare providers
