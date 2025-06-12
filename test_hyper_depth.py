import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('ai_data_new.csv')
X = df[['project_area', 'project_type']]
y = df['panel_name']

# Consistent train-test split
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Define candidate values
n_estimators_list = [10, 20, 30, 40, 44, 50]
random_states = [0, 1, 5, 10, 20]
max_depth_list = [3, 5, 10, 15, 20, None]  # None = unlimited depth

# Track best result
best_accuracy = 0
best_params = {}

# Grid search
for n_estimators in n_estimators_list:
    for random_state in random_states:
        for max_depth in max_depth_list:
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=random_state,
                max_depth=max_depth
            )
            model.fit(train_X, train_y)
            preds = model.predict(val_X)
            acc = accuracy_score(val_y, preds)

            print(f"n_estimators={n_estimators}, random_state={random_state}, max_depth={max_depth} -> Accuracy: {acc:.4f}")

            if acc > best_accuracy:
                best_accuracy = acc
                best_params = {
                    'n_estimators': n_estimators,
                    'random_state': random_state,
                    'max_depth': max_depth
                }

print("\n✅ Best combination:")
print(best_params)
print(f"Highest Accuracy: {best_accuracy:.4f}")


# ✅ Best combination:
# {'n_estimators': 20, 'random_state': 10, 'max_depth': 10}
# Highest Accuracy: 0.0492