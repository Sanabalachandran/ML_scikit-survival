import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sksurv.datasets import make_gbsg2
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored

# Configure logging for production-ready traceability
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_preprocess_oncology_data():
    """
    Loads the GBSG2 (German Breast Cancer Study Group) dataset and preprocesses it.
    Features include: age, estrogen receptor (er), progesterone receptor (pgr), 
    tumor size, tumor grade, menopausal status, and hormone therapy.
    """
    logging.info("Step 1: Loading oncology clinical dataset...")
    X, y = make_gbsg2()
    
    # Standardize categorical variables using one-hot encoding
    logging.info("Step 2: Encoding categorical clinical features...")
    X_encoded = pd.get_dummies(X, columns=["horTh", "tgrade", "menostat"], drop_first=True)
    
    # Ensure all boolean columns from get_dummies are numeric (0 or 1)
    X_encoded = X_encoded.astype(float)
    
    return X_encoded, y

def split_and_scale_data(X, y):
    """
    Splits data into training and testing sets, and standardizes continuous clinical variables.
    """
    logging.info("Step 3: Splitting dataset into Stratified-style train/test sets...")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    
    # Scale continuous features to avoid magnitude bias in linear structures
    continuous_features = ["age", "tsize", "pgr", "er", "pnodes"]
    scaler = StandardScaler()
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[continuous_features] = scaler.fit_transform(X_train[continuous_features])
    X_test_scaled[continuous_features] = scaler.transform(X_test[continuous_features])
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Trains a baseline Cox Proportional Hazards model and an advanced Random Survival Forest.
    Evaluates both using the clinical Concordance Index (C-index).
    """
    # 1. Baseline Model: Cox Proportional Hazards
    logging.info("Step 4: Training baseline Cox Proportional Hazards model...")
    cox_model = CoxPHSurvivalAnalysis()
    cox_model.fit(X_train, y_train)
    
    # 2. Advanced Model: Random Survival Forest (Handles non-linear relationships)
    logging.info("Step 5: Training Advanced Random Survival Forest (RSF) model...")
    rsf_model = RandomSurvivalForest(
        n_estimators=100, min_samples_split=10, min_samples_leaf=5, n_jobs=-1, random_state=42
    )
    rsf_model.fit(X_train, y_train)
    
    # 3. Clinical Evaluation using C-index
    logging.info("Step 6: Evaluating models using the Concordance Index...")
    
    # Cox Evaluation
    cox_predictions = cox_model.predict(X_test)
    cox_cindex = concordance_index_censored(y_test["cens"], y_test["time"], cox_predictions)[0]
    
    # RSF Evaluation
    rsf_predictions = rsf_model.predict(X_test)
    rsf_cindex = concordance_index_censored(y_test["cens"], y_test["time"], rsf_predictions)[0]
    
    print("\n" + "="*40)
    print(f"Baseline Cox Model C-Index: {cox_cindex:.4f}")
    print(f"Advanced RSF Model C-Index:  {rsf_cindex:.4f}")
    print("="*40 + "\n")
    
    return rsf_model, X_test, y_test

def plot_patient_survival_curves(model, X_test, y_test):
    """
    Generates predicted survival curves for specific test patients to demonstrate 
    clinical risk stratification utility.
    """
    logging.info("Step 7: Generating patient-specific predicted survival curves...")
    
    # Select two distinct patients from the test set (e.g., patient index 0 and index 10)
    sample_patients = X_test.iloc[[0, 10]]
    
    # Predict step-function survival curves
    surv_funcs = model.predict_survival_function(sample_patients, return_array=True)
    
    plt.figure(figsize=(10, 6))
    for i, surv_func in enumerate(surv_funcs):
        plt.step(model.unique_times_, surv_func, where="post", label=f"Patient {i+1} Predicted Trajectory")
        
    plt.ylabel("Probability of Progression-Free Survival (PFS)")
    plt.xlabel("Time (Days)")
    plt.title("Clinical Risk Stratification: Individualized Oncology Survival Profiles")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    
    # Save the output visualization for your GitHub documentation
    output_filename = "patient_survival_trajectories.png"
    plt.savefig(output_filename, dpi=300)
    logging.info(f"Pipeline complete! Survival curves saved to '{output_filename}'.")
    plt.show()

if __name__ == "__main__":
    # Execute the end-to-end clinical machine learning pipeline
    X, y = load_and_preprocess_oncology_data()
    X_train, X_test, y_train, y_test = split_and_scale_data(X, y)
    trained_rsf, X_test_out, y_test_out = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    plot_patient_survival_curves(trained_rsf, X_test_out, y_test_out)
