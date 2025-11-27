# Credit Card Fraud Detection


This project implements a credit card fraud detection system using Logistic Regression on a highly imbalanced dataset of European card transactions. The goal is to demonstrate a simple, effective workflow for fraud classification using supervised learning.

The dataset contains anonymized transaction features (V1–V28), the transaction amount, and a binary label indicating whether a transaction is fraudulent.

## Dataset

The dataset is **not included** in this repository due to its size.
Download it from Kaggle:

Credit Card Fraud Detection Dataset:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

After downloading, place `creditcard.csv` in the project directory.

## Project Structure
    ├── CreditCardFraudDetection.py     # Main project script
    ├── creditcard.csv       # Dataset (excluded from GitHub)
    └── README.md                     # Project documentation

## Overview of the Approach

**1. Data Loading**

   -  Load the dataset using pandas.
   -  Inspect head/tail, null values, data types, and class distribution.

**2. Class Imbalance Handling**

   - Fraud cases are extremely rare.

   - Undersample the majority class (legitimate transactions).
   - Create a balanced dataset of:

     - 492 legitimate transactions

     - 492 fraudulent transactions

**3. Feature/Label Preparation**

   - Drop the Class column to form feature matrix X.

   - Keep Class as target vector Y.

**4. Train–Test Split**

   - 80/20 split using stratification to preserve class ratio.

**5. Model Training**

   - Train a Logistic Regression model on the sampled dataset.

**6. Evaluation**

  -  Compute training and testing accuracy.

## Installation & Requirements

Install required packages:
```bash
pip install pandas scikit-learn
```
## Running the Project

1. Download the dataset from Kaggle.

2. Rename it to creditcard.csv and place it in the same directory as the script.

3. Run the script:
```bash
python CreditCardFraudDetection.py
```

The script will:

- Print dataset insights,

- Show class balancing results,

- Display training/testing accuracy.

## Key Notes

- Imbalanced Dataset:

    The original dataset is extremely skewed (fraudulent cases ≪ legitimate). This project uses undersampling for simplicity. In production, methods such as SMOTE, anomaly detection, or ensemble models should be considered.

- Baseline Model:

    Logistic Regression is used as an interpretable baseline. More sophisticated models often perform better.

## Future Improvements

- Add SMOTE or other oversampling techniques.

- Use Random Forest, XGBoost, or Neural Networks.

- Implement precision, recall, F1-score, and ROC-AUC metrics.

- Add cross-validation and model explainability (SHAP).

- Build a real-time prediction pipeline or API.

## License

This project is provided for educational and research purposes. Review the Kaggle dataset license before using the dataset commercially.

## Author
**Hrishikesh Suryawanshi**