import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load your data
df = pd.read_csv('ai_data.csv')

# Select input features and target
X = df[['project_area', 'project_type']]
y = df['panel_name']

# Split data once with fixed random_state
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

def get_accuracy_for_class_weight(class_weight):
    model = RandomForestClassifier(
        n_estimators=30, 
        random_state=0,
        class_weight=class_weight
    )
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    accuracy = accuracy_score(val_y, preds_val)
    return accuracy

# Test both None and 'balanced' for class_weight
candidate_class_weights = [None, 'balanced']
results = {}

for cw in candidate_class_weights:
    accuracy = get_accuracy_for_class_weight(cw)
    print(f"class_weight={cw}\t Accuracy: {accuracy:.4f}")
    results[cw] = accuracy

best_cw = max(results, key=results.get)
print(f"\nBest class_weight: {best_cw} with Accuracy = {results[best_cw]:.4f}")

# class_weight=None        Accuracy: 0.0405
# class_weight=balanced    Accuracy: 0.0304

# Best class_weight: None with Accuracy = 0.0405