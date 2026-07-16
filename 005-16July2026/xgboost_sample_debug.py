"""
XGBoost example — with a look INSIDE the training process.

Shows:
  1. Loss decreasing round-by-round (the actual boosting happening)
  2. The structure of individual trees (as text, and as a plot)
  3. Which features the model actually used, and how much
  4. A manual, from-scratch mini boosting loop so you can see
     the "fit residuals" idea with plain numbers (no XGBoost, just to build intuition)
"""

import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ------------------------------------------------------------------
# 1. Data
# ------------------------------------------------------------------
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=list(feature_names))
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=list(feature_names))

params = {
    "objective": "binary:logistic",
    "max_depth": 3,        # kept shallow so trees are easy to read below
    "eta": 0.3,
    "eval_metric": "logloss",
}

# ------------------------------------------------------------------
# 2. Train WHILE WATCHING the loss shrink each round
# ------------------------------------------------------------------
print("=" * 60)
print("STEP-BY-STEP TRAINING (watch logloss fall each round)")
print("=" * 60)

evals_result = {}  # will store the loss history
watchlist = [(dtrain, "train"), (dtest, "eval")]

bst = xgb.train(
    params,
    dtrain,
    num_boost_round=20,          # only 20 trees, kept small on purpose
    evals=watchlist,
    evals_result=evals_result,
    verbose_eval=True,           # <-- prints loss after every single round
)

# Plot how the loss dropped, round by round
plt.figure(figsize=(6, 4))
plt.plot(evals_result["train"]["logloss"], label="train logloss")
plt.plot(evals_result["eval"]["logloss"], label="test logloss")
plt.xlabel("Boosting round (tree #)")
plt.ylabel("Log loss")
plt.title("Loss decreasing as each new tree is added")
plt.legend()
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/loss_curve.png", dpi=120)
plt.close()
print("\nSaved loss curve -> loss_curve.png")

# ------------------------------------------------------------------
# 3. Look at the ACTUAL trees XGBoost built
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("WHAT DOES TREE #0 (the very first tree) LOOK LIKE?")
print("=" * 60)

tree_dump = bst.get_dump(with_stats=True)
print(f"Total trees built: {len(tree_dump)}\n")
print(tree_dump[0])  # raw text structure of the first tree

# Human-readable: each line is a split like
#   "mean radius < 16.8 -> go left/right, leaf value = ..."
# 'gain' = how much that split reduced the loss (higher = more useful split)
# 'cover' = how many training samples passed through that node

# Visual plot of the first tree (needs graphviz installed)
try:
    fig, ax = plt.subplots(figsize=(20, 10))
    xgb.plot_tree(bst, num_trees=0, ax=ax)
    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/tree_0.png", dpi=150)
    plt.close()
    print("\nSaved a picture of tree #0 -> tree_0.png")
except Exception as e:
    print(f"\n(Skipping tree plot, graphviz not available: {e})")

# ------------------------------------------------------------------
# 4. Which features actually mattered?
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("FEATURE IMPORTANCE (3 different ways to measure it)")
print("=" * 60)

for kind in ["weight", "gain", "cover"]:
    scores = bst.get_score(importance_type=kind)
    top5 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"\nTop 5 by '{kind}':")
    for feat, val in top5:
        print(f"  {feat:25s} {val:.2f}")

# 'weight' = how many times a feature was used to split, across all trees
# 'gain'   = average improvement in accuracy brought by splits on that feature
#            (usually the most meaningful one)
# 'cover'  = average number of samples affected by splits on that feature

fig, ax = plt.subplots(figsize=(8, 6))
xgb.plot_importance(bst, importance_type="gain", ax=ax, max_num_features=10)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/feature_importance.png", dpi=120)
plt.close()
print("\nSaved feature importance chart -> feature_importance.png")

# ------------------------------------------------------------------
# 5. Final accuracy check
# ------------------------------------------------------------------
y_pred = [1 if p > 0.5 else 0 for p in bst.predict(dtest)]
print("\n" + "=" * 60)
print(f"Final test accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("=" * 60)

# ------------------------------------------------------------------
# 6. BONUS: a tiny hand-built boosting loop (no XGBoost)
#    This is the core idea XGBoost is doing internally, stripped down
#    to plain numbers, for regression on a toy example.
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("BONUS: boosting 'by hand' on 5 toy numbers")
print("=" * 60)

y_true = np.array([10, 20, 30, 40, 50], dtype=float)
prediction = np.full_like(y_true, y_true.mean())  # start: predict the average
learning_rate = 0.5

print(f"Start prediction (just the mean): {prediction}")

for round_num in range(1, 4):
    residual = y_true - prediction          # what we still got wrong
    # a "tree" here is simplified to just: predict the residual directly
    tree_prediction = residual
    prediction = prediction + learning_rate * tree_prediction
    print(f"\nRound {round_num}:")
    print(f"  residual (errors so far): {residual}")
    print(f"  updated prediction:       {prediction}")

print(f"\nTrue values were:            {y_true}")
print("Notice how prediction creeps closer to the truth each round —")
print("that's exactly what XGBoost's 20 trees above were doing, just")
print("with real decision trees instead of 'predict the residual directly'.")