CipherCloud Project Overview
===========================

This project is an end-to-end machine learning pipeline for automated risk assessment of AWS IAM policies. It is designed to help cloud security teams quickly identify risky policies using data-driven techniques.

Main Components
---------------

1. **Dataset Generation (`Dataset/generator.py`)**
   - Synthetic IAM policies are generated and labeled as "Benign" or "Risky".
   - The dataset covers a wide range of policy patterns, including privilege escalation, wildcard permissions, and admin-equivalent actions.

2. **Feature Extraction (`Dataset/feature_extractor.py`)**
   - Extracts statistical, risk-based, and text features from each policy.
   - Uses TF-IDF to capture semantic information from actions and resources.
   - Processes the dataset and saves feature matrices and labels for model training.

3. **Model Training & Evaluation (`Model/Binary.py`)**
   - Loads processed features and labels.
   - Trains multiple classifiers (Random Forest, Logistic Regression, SVM).
   - Evaluates models using accuracy, AUC, confusion matrix, and ROC curve.
   - Saves the best-performing model and evaluation plots for future use.

4. **Policy Scanning (`Scanners/Binary_Scanner.py`)**
   - Loads the trained model and scans new IAM policies.
   - Predicts risk level and provides explanations for flagged policies.
   - Includes an interactive CLI for quick policy analysis.

5. **Data & Artifacts**
   - All processed features, labels, and feature names are saved as CSV/JSON files.
   - Evaluation plots are saved as images for reporting and review.
