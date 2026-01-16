# High-level Analysis Pseudo-code (ICES-restricted data)

Input: ICES-derived analytic dataset (restricted access)
Output: Model performance, calibration, fairness, and explainability results

1. Data Preparation
   - Apply cohort inclusion/exclusion criteria
   - Define index date, look-back, and look-ahead windows
   - Construct outcome: delayed hospital discharge

2. Feature Construction
   - Derive socio-demographic, geographic, multimorbidity, and mobility features
   - Remove observations with negligible missing data (complete-case analysis)

3. Data Splitting
   - Randomly split data into training, validation, and test sets
   - Create a temporally separated hold-out cohort for validation

4. Model Development
   - Tune hyperparameters using k-fold cross-validation on training data
   - Train Logistic Regression and XGBoost models
   - Generate predicted probabilities

5. Model Evaluation and Calibration
   - Evaluate discrimination and calibration on the held-out test set
   - Apply post-hoc Platt scaling (global and group-stratified)
   - Re-evaluate calibrated models
   - Repeat evaluation on the temporal hold-out cohort

6. Fairness Assessment
   - Define protected groups (sex, residency, residential instability)
   - Compute threshold-free fairness metrics (AUC parity, xAUC parity, ECE parity)

7. Explainability Analysis
   - Local explainability: SHAP, breakdown, Ceteris Paribus profiles
   - Global explainability: permutation importance, partial dependence plots

8. Synthesis
   - Aggregate local explanations across high-risk patients
   - Perform PCA-based clustering of feature contributions
   - Generate figures and tables for reporting
