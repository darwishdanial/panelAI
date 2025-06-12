import joblib
import numpy as np

# Load the trained model
model = joblib.load('panel_model_old.joblib')

# Example input
test_data = np.array([[2, 0]])

# Predict the panel
prediction = model.predict(test_data)
probabilities = model.predict_proba(test_data)

print("🎯 Predicted panel:", prediction[0])
print("📊 Probabilities (sorted):")

# Combine class names and probabilities, then sort descending
sorted_probs = sorted(
    zip(model.classes_, probabilities[0]),
    key=lambda x: x[1],
    reverse=True
)

for class_name, prob in sorted_probs:
    print(f" - {class_name}: {prob:.4f}")
