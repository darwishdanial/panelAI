import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load your training data
df = pd.read_csv('ai_data.csv')

# Select input features and target
X = df[['project_area', 'project_type']]
y = df['panel_name']

# Train the Random Forest model with class_weight='balanced'
model = RandomForestClassifier(n_estimators=10, random_state=5, max_leaf_nodes=15)
model.fit(X, y)

# Save the trained model
joblib.dump(model, 'panel_model_result_new_category.joblib')

print("✅ Model trained and saved")

# n_estimators is the number of decision trees the algorithm will build.
# random_state is a seed used by the random number generator, which ensures that the results are reproducible.
# max_leaf_nodes is the maximum number of leaf nodes in each decision tree.

# Use Model A (with max_leaf_nodes) if you care about :
# Faster inference
# Smaller model size
# Simpler model tuning (fewer trees)

# Use Model B (with max_depth) if you prefer:
# Potentially better generalization in larger or more complex datasets
# More flexible control over tree depth
