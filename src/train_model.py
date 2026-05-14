import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


# Create output folder
os.makedirs("outputs", exist_ok=True)

# Load dataset
df = pd.read_csv("data/creditcard.csv")

print("Dataset loaded successfully!")
print(df.head())
print("\nDataset shape:", df.shape)

# Check fraud and non-fraud count
print("\nFraud and Non-Fraud Count:")
print(df["Class"].value_counts())

# Feature Engineering: Transaction time patterns
df["Hour"] = (df["Time"] / 3600) % 24
df["Hour"] = df["Hour"].astype(int)

df["Time_Period"] = pd.cut(
    df["Hour"],
    bins=[0, 6, 12, 18, 24],
    labels=["Night", "Morning", "Afternoon", "Evening"],
    include_lowest=True
)

# Convert category into numbers
df = pd.get_dummies(df, columns=["Time_Period"], drop_first=True)

# Scale Amount column
scaler = StandardScaler()
df["Scaled_Amount"] = scaler.fit_transform(df[["Amount"]])

# Drop original Time and Amount
df = df.drop(["Time", "Amount"], axis=1)

# Features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# Handle imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE:")
print(y_train_smote.value_counts())

# Handle imbalance using Undersampling
under = RandomUnderSampler(random_state=42)
X_train_under, y_train_under = under.fit_resample(X_train, y_train)

print("\nAfter Undersampling:")
print(y_train_under.value_counts())


# Function to train and evaluate model
def evaluate_model(model_name, model, X_train_data, y_train_data):
    model.fit(X_train_data, y_train_data)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    print("\nModel:", model_name)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("ROC AUC:", roc_auc)
    print("Confusion Matrix:")
    print(cm)

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC AUC": roc_auc
    }, model, y_pred, y_prob


results = []

# Logistic Regression with SMOTE
log_model = LogisticRegression(max_iter=1000)
result, trained_log, pred_log, prob_log = evaluate_model(
    "Logistic Regression + SMOTE",
    log_model,
    X_train_smote,
    y_train_smote
)
results.append(result)

# Random Forest with SMOTE
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
result, trained_rf, pred_rf, prob_rf = evaluate_model(
    "Random Forest + SMOTE",
    rf_model,
    X_train_smote,
    y_train_smote
)
results.append(result)

# XGBoost with SMOTE
xgb_model = XGBClassifier(eval_metric="logloss", random_state=42)
result, trained_xgb, pred_xgb, prob_xgb = evaluate_model(
    "XGBoost + SMOTE",
    xgb_model,
    X_train_smote,
    y_train_smote
)
results.append(result)

# Save comparison
comparison_df = pd.DataFrame(results)
comparison_df.to_csv("outputs/model_comparison.csv", index=False)

print("\nModel Comparison:")
print(comparison_df)

# Choose best model by F1 Score
best_model_name = comparison_df.sort_values(by="F1 Score", ascending=False).iloc[0]["Model"]

if best_model_name == "Logistic Regression + SMOTE":
    best_model = trained_log
    best_probs = prob_log
    best_preds = pred_log
elif best_model_name == "Random Forest + SMOTE":
    best_model = trained_rf
    best_probs = prob_rf
    best_preds = pred_rf
else:
    best_model = trained_xgb
    best_probs = prob_xgb
    best_preds = pred_xgb

print("\nBest Model:", best_model_name)

# Save model
joblib.dump(best_model, "outputs/fraud_model.pkl")
joblib.dump(scaler, "outputs/scaler.pkl")

# Business Layer
test_data = X_test.copy()
test_data["Actual_Class"] = y_test.values
test_data["Predicted_Class"] = best_preds
test_data["Risk_Score"] = best_probs

# Assumption
average_fraud_amount = 10000
false_positive_review_cost = 100

true_fraud_detected = len(test_data[(test_data["Actual_Class"] == 1) & (test_data["Predicted_Class"] == 1)])
false_positives = len(test_data[(test_data["Actual_Class"] == 0) & (test_data["Predicted_Class"] == 1)])
fraud_missed = len(test_data[(test_data["Actual_Class"] == 1) & (test_data["Predicted_Class"] == 0)])

money_saved = true_fraud_detected * average_fraud_amount
false_positive_cost = false_positives * false_positive_review_cost
net_savings = money_saved - false_positive_cost

print("\nBusiness Analysis:")
print("Fraud transactions detected:", true_fraud_detected)
print("False positives:", false_positives)
print("Fraud missed:", fraud_missed)
print("Estimated money saved:", money_saved)
print("False positive cost:", false_positive_cost)
print("Net savings:", net_savings)

test_data["Money_Saved"] = money_saved
test_data["False_Positive_Cost"] = false_positive_cost
test_data["Net_Savings"] = net_savings

# Save dashboard data
test_data.to_csv("outputs/dashboard_data.csv", index=False)

print("\nFiles saved successfully in outputs folder!")

# Save individual predictions with probabilities for the dashboard
test_data['Risk_Score'] = best_model.predict_proba(X_test)[:, 1] # Probability of fraud
test_data['Is_Fraud_Predicted'] = best_model.predict(X_test)
test_data.to_csv("outputs/fraud_predictions_full.csv", index=False)

# Save Feature Importance for the Dashboard
importances = best_model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({
    'Feature': feature_names, 
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

feature_importance_df.to_csv("outputs/feature_importance.csv", index=False)
print("Feature importance saved successfully!")