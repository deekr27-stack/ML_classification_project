# app.py - Streamlit Web Application for Customer Churn Classification

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    precision_score, 
    recall_score, 
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    h1 {
        color: #1f77b4;
    }
    h2 {
        color: #2ca02c;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and Description
st.title(" Customer Churn Prediction - ML Classification")
st.markdown("---")
st.markdown("""
### Machine Learning Assignment 2 - BITS Pilani M.Tech (AIML/DSE)
This application demonstrates **6 different classification models** for predicting customer churn
in the telecom industry.
""")

# Sidebar
st.sidebar.header(" Configuration")
st.sidebar.markdown("---")

# Model Selection
st.sidebar.subheader(" Select Model")
model_choice = st.sidebar.selectbox(
    "Choose a classification model:",
    [
        "Logistic Regression",
        "Decision Tree",
        "K-Nearest Neighbors",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

# Model file mapping
model_files = {
    "Logistic Regression": "models/logistic_regression.pkl",
    "Decision Tree": "models/decision_tree.pkl",
    "K-Nearest Neighbors": "models/knn.pkl",
    "Naive Bayes": "models/naive_bayes.pkl",
    "Random Forest": "models/random_forest.pkl",
    "XGBoost": "models/xgboost.pkl"
}

# Models that need scaling
models_needing_scaling = ["Logistic Regression", "K-Nearest Neighbors", "Naive Bayes"]

# Load model results
@st.cache_data
def load_model_results():
    try:
        results_df = pd.read_csv('models/all_model_results.csv')
        return results_df
    except:
        return None

# Load selected model
@st.cache_resource
def load_model(model_path):
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load scaler
@st.cache_resource
def load_scaler():
    try:
        scaler = joblib.load('models/scaler.pkl')
        return scaler
    except Exception as e:
        st.warning(f"Scaler not found: {e}")
        return None

# Load feature columns
@st.cache_data
def load_feature_columns():
    try:
        with open('models/feature_columns.txt', 'r') as f:
            features = [line.strip() for line in f.readlines()]
        return features
    except:
        return None

# Main App Logic
st.markdown("---")

# Display Model Information
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f" Selected Model: {model_choice}")
    
with col2:
    if model_choice in models_needing_scaling:
        st.info(" This model uses scaled features")
    else:
        st.success(" This model uses original features")

# Load the selected model
model = load_model(model_files[model_choice])
scaler = load_scaler()
feature_columns = load_feature_columns()

if model is None:
    st.error(" Could not load the model. Please check if model files exist in the 'models/' directory.")
    st.stop()

# File Upload Section
st.markdown("---")
st.subheader(" Upload Test Data")
st.markdown("Upload a CSV file with customer data for prediction. The file should contain the same features as the training data.")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=['csv'],
    help="Upload test dataset in CSV format"
)

if uploaded_file is not None:
    try:
        # Read the uploaded file
        test_data = pd.read_csv(uploaded_file)
        
        st.success(f" File uploaded successfully! Shape: {test_data.shape}")
        
        # Display first few rows
        with st.expander(" Preview Uploaded Data"):
            st.dataframe(test_data.head(10))
        
        # Check if target column exists
        target_col = 'Churn'
        has_target = target_col in test_data.columns
        
        if has_target:
            st.info(f" Target column '{target_col}' found in the dataset. Evaluation metrics will be calculated.")
            
            # Preprocess the data
            test_df = test_data.copy()
            
            # Drop customerID if exists
            if 'customerID' in test_df.columns:
                test_df = test_df.drop('customerID', axis=1)
            
            # Handle TotalCharges
            if 'TotalCharges' in test_df.columns and test_df['TotalCharges'].dtype == 'object':
                test_df['TotalCharges'] = pd.to_numeric(test_df['TotalCharges'], errors='coerce')
                test_df['TotalCharges'].fillna(test_df['TotalCharges'].median(), inplace=True)
            
            # Encode target
            if test_df[target_col].dtype == 'object':
                test_df[target_col] = test_df[target_col].map({'Yes': 1, 'No': 0})
            
            # Separate features and target
            X_test = test_df.drop(target_col, axis=1)
            y_test = test_df[target_col]
            
            # One-hot encode categorical variables
            categorical_cols = X_test.select_dtypes(include=['object']).columns.tolist()
            if len(categorical_cols) > 0:
                X_test_encoded = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)
            else:
                X_test_encoded = X_test
            
            # Align columns with training data
            if feature_columns is not None:
                # Add missing columns with 0
                for col in feature_columns:
                    if col not in X_test_encoded.columns:
                        X_test_encoded[col] = 0
                # Remove extra columns
                X_test_encoded = X_test_encoded[feature_columns]
            
            # Scale if needed
            if model_choice in models_needing_scaling and scaler is not None:
                X_test_processed = scaler.transform(X_test_encoded)
            else:
                X_test_processed = X_test_encoded
            
            # Make predictions
            y_pred = model.predict(X_test_processed)
            
            if hasattr(model, "predict_proba"):
                y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
            else:
                y_pred_proba = y_pred
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            mcc = matthews_corrcoef(y_test, y_pred)
            
            # Display Metrics
            st.markdown("---")
            st.subheader(" Evaluation Metrics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Accuracy", f"{accuracy:.4f}")
                st.metric("AUC Score", f"{auc:.4f}")
            
            with col2:
                st.metric("Precision", f"{precision:.4f}")
                st.metric("Recall", f"{recall:.4f}")
            
            with col3:
                st.metric("F1 Score", f"{f1:.4f}")
                st.metric("MCC Score", f"{mcc:.4f}")
            
            # Confusion Matrix
            st.markdown("---")
            st.subheader(" Confusion Matrix")
            
            cm = confusion_matrix(y_test, y_pred)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=['No Churn', 'Churn'],
                           yticklabels=['No Churn', 'Churn'],
                           ax=ax)
                ax.set_title(f'Confusion Matrix - {model_choice}', fontsize=14, fontweight='bold')
                ax.set_ylabel('Actual', fontsize=12)
                ax.set_xlabel('Predicted', fontsize=12)
                st.pyplot(fig)
            
            with col2:
                st.markdown("### Confusion Matrix Values")
                st.markdown(f"""
                - **True Negatives (TN):** {cm[0][0]}
                - **False Positives (FP):** {cm[0][1]}
                - **False Negatives (FN):** {cm[1][0]}
                - **True Positives (TP):** {cm[1][1]}
                
                **Total Predictions:** {cm.sum()}
                """)
                
                st.markdown("### Interpretation")
                st.markdown(f"""
                - **Correct Predictions:** {cm[0][0] + cm[1][1]} ({((cm[0][0] + cm[1][1]) / cm.sum() * 100):.2f}%)
                - **Incorrect Predictions:** {cm[0][1] + cm[1][0]} ({((cm[0][1] + cm[1][0]) / cm.sum() * 100):.2f}%)
                """)
            
            # Classification Report
            st.markdown("---")
            st.subheader(" Classification Report")
            
            report = classification_report(y_test, y_pred, 
                                          target_names=['No Churn', 'Churn'],
                                          output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.highlight_max(axis=0))
            
            # Predictions Distribution
            st.markdown("---")
            st.subheader(" Predictions Distribution")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(8, 6))
                pred_counts = pd.Series(y_pred).value_counts()
                ax.bar(['No Churn', 'Churn'], [pred_counts.get(0, 0), pred_counts.get(1, 0)], 
                      color=['skyblue', 'salmon'], edgecolor='black')
                ax.set_title('Predicted Class Distribution', fontsize=14, fontweight='bold')
                ax.set_ylabel('Count')
                ax.grid(axis='y', alpha=0.3)
                st.pyplot(fig)
            
            with col2:
                fig, ax = plt.subplots(figsize=(8, 6))
                actual_counts = y_test.value_counts()
                ax.bar(['No Churn', 'Churn'], [actual_counts.get(0, 0), actual_counts.get(1, 0)], 
                      color=['lightgreen', 'lightcoral'], edgecolor='black')
                ax.set_title('Actual Class Distribution', fontsize=14, fontweight='bold')
                ax.set_ylabel('Count')
                ax.grid(axis='y', alpha=0.3)
                st.pyplot(fig)
        
        else:
            st.warning(f" Target column '{target_col}' not found. Only predictions will be shown.")
            
            # Preprocess for prediction only
            test_df = test_data.copy()
            
            if 'customerID' in test_df.columns:
                customer_ids = test_df['customerID']
                test_df = test_df.drop('customerID', axis=1)
            else:
                customer_ids = None
            
            if 'TotalCharges' in test_df.columns and test_df['TotalCharges'].dtype == 'object':
                test_df['TotalCharges'] = pd.to_numeric(test_df['TotalCharges'], errors='coerce')
                test_df['TotalCharges'].fillna(test_df['TotalCharges'].median(), inplace=True)
            
            categorical_cols = test_df.select_dtypes(include=['object']).columns.tolist()
            if len(categorical_cols) > 0:
                test_df_encoded = pd.get_dummies(test_df, columns=categorical_cols, drop_first=True)
            else:
                test_df_encoded = test_df
            
            if feature_columns is not None:
                for col in feature_columns:
                    if col not in test_df_encoded.columns:
                        test_df_encoded[col] = 0
                test_df_encoded = test_df_encoded[feature_columns]
            
            if model_choice in models_needing_scaling and scaler is not None:
                test_processed = scaler.transform(test_df_encoded)
            else:
                test_processed = test_df_encoded
            
            predictions = model.predict(test_processed)
            
            if hasattr(model, "predict_proba"):
                pred_probas = model.predict_proba(test_processed)[:, 1]
            else:
                pred_probas = predictions
            
            # Display predictions
            st.markdown("---")
            st.subheader(" Predictions")
            
            results_df = pd.DataFrame({
                'Prediction': ['Churn' if p == 1 else 'No Churn' for p in predictions],
                'Churn Probability': [f"{prob:.4f}" for prob in pred_probas]
            })
            
            if customer_ids is not None:
                results_df.insert(0, 'Customer ID', customer_ids.values)
            
            st.dataframe(results_df)
            
            # Summary
            churn_count = (predictions == 1).sum()
            no_churn_count = (predictions == 0).sum()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Predictions", len(predictions))
            with col2:
                st.metric("Predicted Churn", churn_count)
            with col3:
                st.metric("Predicted No Churn", no_churn_count)
    
    except Exception as e:
        st.error(f" Error processing file: {e}")
        st.info("Please ensure your CSV file has the correct format and columns.")

else:
    st.info(" Please upload a CSV file to get started.")

# Model Comparison Section
st.markdown("---")
st.subheader(" All Models Comparison")

results_df = load_model_results()

if results_df is not None:
    st.dataframe(results_df.style.highlight_max(axis=0, subset=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']))
    
    # Visual comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    metrics = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        values = results_df[metric].values
        models = results_df['Model'].values
        
        bars = ax.barh(models, values, color=colors[idx], alpha=0.8, edgecolor='black')
        ax.set_xlabel('Score', fontsize=10)
        ax.set_title(f'{metric}', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.grid(axis='x', alpha=0.3)
        
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=8)
    
    plt.suptitle('Performance Comparison Across All Models', fontsize=16, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
else:
    st.warning("Model comparison data not available.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Machine Learning Assignment 2 | BITS Pilani M.Tech (AIML/DSE)</p>
    <p>Customer Churn Prediction using Classification Models</p>
</div>
""", unsafe_allow_html=True)