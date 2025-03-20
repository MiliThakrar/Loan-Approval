# Loan Approval Classification Project

## Overview
This project predicts loan approval status using machine learning techniques, analyzing a dataset containing personal details, loan specifics, and credit history of applicants.

## Data Dictionary

| Column Name | Data Type | Description |
|-------------|-----------|-------------|
| person_age | int64 | Age of the loan applicant |
| person_gender | object | Gender of the loan applicant |
| person_education | object | Education level of the loan applicant |
| person_income | float64 | Annual income of the loan applicant |
| person_emp_exp | int64 | Employment experience of the loan applicant in years |
| person_home_ownership | object | Home ownership status of the loan applicant |
| loan_amnt | float64 | Amount of loan applied for |
| loan_intent | object | Purpose of the loan |
| loan_int_rate | float64 | Interest rate of the loan |
| debt_to_income_percentage | float64 | Ratio of loan/debt amount to annual income in percentage |
| cb_person_cred_hist_length | float64 | Length of the applicant's credit history |
| credit_score | int64 | Credit score of the loan applicant |
| previous_loan_defaults_on_file | object | Whether the applicant has previous loan defaults on file |
| loan_status | int64 | Status of the loan (0 for not approved, 1 for approved) |

## Project Steps

1. **Data Cleaning**: Handled missing values and outliers.
2. **Exploratory Data Analysis (EDA)**: Visualized distributions and analyzed correlations.
3. **Model Development**:
   - Logistic Regression
   - SMOTE with Logistic Regression
   - PCA with Logistic Regression
   - KNN
   - ML Pipeline with Random Search CV

## Results

Our machine learning pipeline, utilizing Random Search Cross-Validation, identified the optimal model configuration for loan approval prediction. The best-performing model incorporated the following components:

1. **Feature Scaling**: StandardScaler
2. **Dimensionality Reduction**: Principal Component Analysis (PCA)
3. **Classifier**: Random Forest with 200 estimators

This combination significantly outperformed other tested configurations, demonstrating superior predictive power for loan approval classification. The use of StandardScaler ensured all features were on a comparable scale, while PCA helped in reducing dimensionality and capturing the most important aspects of the data. The Random Forest Classifier, with its ensemble of 200 decision trees, proved to be highly effective in capturing complex patterns within the dataset.

## Conclusion

The project successfully identified an effective approach for loan approval classification, leveraging advanced preprocessing techniques and a powerful ensemble learning method.

## Future Work

- Explore additional advanced models (e.g., XGBoost, Neural Networks)
- Investigate feature importance and selection techniques
- Deploy the model as a web application for real-time predictions

## Dependencies

- Python 3.x
- pandas
- numpy
- scikit-learn
- imbalanced-learn
- matplotlib
- seaborn
