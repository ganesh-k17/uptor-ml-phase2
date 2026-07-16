import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 1. Load a sample dataset (binary classification: malignant vs benign)
data = load_breast_cancer()
X, y = data.data, data.target

# 2. Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Convert data into XGBoost's optimized DMatrix format
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# 4. Set hyperparameters
params = {
    "objective": "binary:logistic",  # binary classification
    "max_depth": 4,                  # depth of each tree
    "eta": 0.1,                      # learning rate
    "eval_metric": "logloss"         # evaluation metric
}

# 5. Train the model
num_rounds = 100
bst = xgb.train(params, dtrain, num_boost_round=num_rounds)

# 6. Predict on test data
y_pred_prob = bst.predict(dtest)
y_pred = [1 if p > 0.5 else 0 for p in y_pred_prob]

# 7. Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))