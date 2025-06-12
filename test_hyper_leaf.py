import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('ai_data.csv')
X = df[['project_area', 'project_type']]
y = df['panel_name']

# Consistent train-test split
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Define candidate values for each hyperparameter
n_estimators_list = [10, 20, 30, 40, 44, 50]
random_states = [0, 1, 5, 10, 20]
max_leaf_nodes_list = [10, 15, 20, 25, 30, None]  # None = no limit

# Tracking the best results
best_accuracy = 0
best_params = {}

# Grid search
for n_estimators in n_estimators_list:
    for random_state in random_states:
        for max_leaf_nodes in max_leaf_nodes_list:
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=random_state,
                max_leaf_nodes=max_leaf_nodes
            )
            model.fit(train_X, train_y)
            preds = model.predict(val_X)
            acc = accuracy_score(val_y, preds)

            print(f"n_estimators={n_estimators}, random_state={random_state}, max_leaf_nodes={max_leaf_nodes} -> Accuracy: {acc:.4f}")

            if acc > best_accuracy:
                best_accuracy = acc
                best_params = {
                    'n_estimators': n_estimators,
                    'random_state': random_state,
                    'max_leaf_nodes': max_leaf_nodes
                }

print("\n✅ Best combination:")
print(best_params)
print(f"Highest Accuracy: {best_accuracy:.4f}")

# ✅ Best combination:
# {'n_estimators': 10, 'random_state': 5, 'max_leaf_nodes': 15}
# Highest Accuracy: 0.0570
