import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

# Load test data
df = pd.read_csv('ai_data.csv')  # Use different test CSV if available
X = df[['project_area', 'project_type']]
y_true = df['panel_name']

# Load both models
model_new = joblib.load('panel_model_balanced.joblib')
model_old = joblib.load('panel_model_unbalanced.joblib')

# Predict with both models
y_pred_new = model_new.predict(X)
y_pred_old = model_old.predict(X)

# Evaluate new model
print("🔵 Balanced Model (panel_model_balanced.joblib)")
print(f"Accuracy: {accuracy_score(y_true, y_pred_new):.4f}")
print("Classification Report:")
print(classification_report(y_true, y_pred_new))

print("")

# Evaluate old model
print("🟠 Unbalanced Model (panel_model_unbalanced.joblib)")
print(f"Accuracy: {accuracy_score(y_true, y_pred_old):.4f}")
print("Classification Report:")
print(classification_report(y_true, y_pred_old))
