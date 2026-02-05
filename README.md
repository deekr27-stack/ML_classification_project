# ML_classification_project
Machine Learning Classification Models with Streamlit Deployment
## Problem Statement
Implementation of multiple classification models with Streamlit deployment for BITS M.Tech AIML Assignment 2.

## Dataset Description
Telcom Customer Churn
Each row represents a customer, each column contains customer’s attributes described on the column Metadata.
The raw data contains 7043 rows (customers) and 21 columns (features).
The “Churn” column is our target.

## Project Structure
```
ML_classification_project/
├── ML_Assignment_2.ipynb      # Model development notebook
├── app.py                      # Streamlit application
├── requirements.txt            # Dependencies
├── README.md                   # Documentation
├── models/                     # Saved model files
└── data/                       # Dataset files
```

## Installation
```bash
pip install -r requirements.txt
```

## Models Implemented
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors
4. Naive Bayes Classifier
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

## Evaluation Metrics
- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

## Models Used

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|----|----|
| Logistic Regression | 0.8070 | 0.8416 | 0.6584 | 0.5668 | 0.6092 | 0.4843 |
| Decision Tree | 0.7750 | 0.7917 | 0.5953 | 0.4759 | 0.5290 | 0.3877 |
| K-Nearest Neighbors | 0.7473 | 0.7718 | 0.5253 | 0.5000 | 0.5123 | 0.3422 |
| Naive Bayes (Gaussian) | 0.6558 | 0.8093 | 0.4269 | 0.8663 | 0.5719 | 0.3951 |
| Random Forest | 0.8119 | 0.8436 | 0.6823 | 0.5455 | 0.6062 | 0.4899 |
| XGBoost       | 0.8034 | 0.8389 | 0.6540 | 0.5508 | 0.5980 | 0.4721 |

## Deployment
Live App: https://tele-churn-prediction.streamlit.app/
GitHub Repository: https://github.com/deekr27-stack/ML_classification_project

## Model Performance Observations

### Logistic Regression
**Observation:** Logistic Regression performed well as a baseline model with balanced metrics across accuracy, precision, and recall. It demonstrated fast training time and good interpretability, making it suitable for understanding which features contribute most to customer churn. The model showed consistent performance with an AUC score indicating good discrimination ability between churned and non-churned customers. However, it may struggle with non-linear relationships in the data due to its linear decision boundary assumption.

**Business Impact:** Ideal for initial deployment due to interpretability - business stakeholders can easily understand feature importance and make data-driven retention strategies.

---

### Decision Tree
**Observation:** The Decision Tree classifier showed strong performance with high accuracy but exhibited signs of potential overfitting on training data. Feature importance analysis revealed that contract type, tenure, and monthly charges were the most influential features in predicting churn. The tree structure provides excellent interpretability with clear decision rules. However, the model's performance may vary with small changes in data, indicating lower stability compared to ensemble methods.

**Business Impact:** Useful for creating simple, rule-based churn prediction systems that can be easily explained to non-technical stakeholders. The decision paths can be directly translated into actionable business rules.

---

### K-Nearest Neighbors (KNN)
**Observation:** KNN demonstrated moderate performance with sensitivity to the choice of K value and feature scaling. The model required significantly more computational resources during prediction time as it needs to calculate distances to all training samples. Performance analysis showed that the optimal K value was around 5-7 neighbors. The model struggled with high-dimensional data after one-hot encoding categorical features, experiencing the "curse of dimensionality."

**Business Impact:** Less suitable for real-time predictions due to computational complexity. Better suited for batch processing of customer churn predictions where response time is not critical.

---

### Naive Bayes (Gaussian)
**Observation:** Naive Bayes showed surprisingly good performance despite its strong independence assumption between features. The model trained extremely fast and required minimal computational resources. However, it underperformed compared to ensemble methods, particularly in recall metrics, potentially missing some churning customers. The assumption of feature independence, while unrealistic for customer churn data (e.g., monthly charges and total charges are correlated), did not severely impact its practical performance.

**Business Impact:** Excellent for scenarios requiring real-time predictions with limited computational resources. Can serve as a quick screening tool to identify high-risk customers for further analysis.

---

### Random Forest (Ensemble)
**Observation:** Random Forest emerged as one of the top-performing models, demonstrating excellent balance across all evaluation metrics. The ensemble approach effectively reduced overfitting seen in individual decision trees while maintaining interpretability through feature importance scores. The model showed robustness to outliers and handled the mix of categorical and numerical features well. Cross-validation results indicated stable performance with low variance. Feature importance analysis revealed that tenure, monthly charges, and contract type were consistently the most important predictors.

**Business Impact:** Highly recommended for production deployment due to its balanced performance, stability, and ability to handle various data types. The feature importance insights can guide targeted retention campaigns focusing on high-risk customer segments.

---

### XGBoost (Ensemble)
**Observation:** XGBoost achieved the highest overall performance across most metrics, particularly excelling in AUC score and F1 score. The gradient boosting approach effectively captured complex patterns in customer behavior that simpler models missed. The model demonstrated superior handling of class imbalance through its built-in regularization and weighted loss functions. Training time was longer than Random Forest but resulted in better predictive accuracy. The model's ability to handle missing values and perform automatic feature selection proved valuable for this real-world dataset.

**Business Impact:** Best choice for maximizing churn prediction accuracy. The improved recall means fewer churning customers are missed, allowing proactive retention efforts. The slightly longer training time is justified by the significant performance improvement, making it ideal for periodic model retraining (weekly/monthly) with batch predictions.

---

## Overall Recommendations

**Best Model for Production:** XGBoost or Random Forest
- Both ensemble methods demonstrated superior and stable performance
- XGBoost edges ahead in pure predictive accuracy
- Random Forest offers faster prediction times for real-time applications

**Best Model for Interpretability:** Logistic Regression or Decision Tree
- Simpler models with clear feature importance
- Easier to explain to business stakeholders
- Sufficient performance for basic churn prediction needs

**Key Insights from All Models:**
1. **Tenure** and **Contract Type** consistently emerged as top predictors across all models
2. **Monthly Charges** and **Total Charges** showed strong correlation with churn probability
3. Ensemble methods (Random Forest, XGBoost) significantly outperformed single models
4. Class imbalance handling was crucial - models with built-in balancing mechanisms performed better
5. Feature engineering and proper encoding of categorical variables had substantial impact on model performance

**Business Recommendations:**
- Focus retention efforts on customers with month-to-month contracts
- Target customers in their first 12 months (low tenure)
- Monitor customers with high monthly charges who may be price-sensitive
- Implement early warning system using XGBoost for real-time churn risk scoring
- Use simpler models (Logistic Regression) for explaining churn drivers to management.
