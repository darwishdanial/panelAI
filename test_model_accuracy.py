import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load your training data
df = pd.read_csv('ai_data.csv')

# Select input features and target
X = df[['project_area', 'project_type']]
y = df['panel_name']

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Train the Random Forest model with class_weight='balanced'
model = RandomForestClassifier(n_estimators=20, random_state=10, max_leaf_nodes=15)
model.fit(train_X, train_y)
preds_val = model.predict(val_X)
accuracy = accuracy_score(val_y, preds_val)

print(f"Accuracy: {accuracy:.4f}")

# n_estimators=44, random_state=10, max_leaf_nodes=15 // Accuracy: 0.0532
# n_estimators=20, random_state=10, max_leaf_nodes=15 // Accuracy: 0.0544

