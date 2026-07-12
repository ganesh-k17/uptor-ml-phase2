import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# 1.Load CSV dataset
df = pd.read_csv('customer_churn.csv')

# drop columns that are not needed for the model
# df = df.drop(['Gender', 'Subscription Type', 'Contract Length'], axis=1)

obj  = LabelEncoder()
for x in df.columns:
    if df[x].dtype in ['object', 'str']:
        df[x] = obj.fit_transform(df[x])

# 2. Separate features (X) and target variable (y)
x = df.drop('Churn', axis=1).values
y = df['Churn'].values

# 3. Split into train/test

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)   

# display the values

print("X_train shape:", x_train.shape)
print("X_test shape:", x_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# display the actual data

print("X_train data:\n", x_train)
print("X_test data:\n", x_test)
print("y_train data:\n", y_train)
print("y_test data:\n", y_test) 

# Scale features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Build model
clf = MLPClassifier(
    hidden_layer_sizes=(100,), 
    activation='relu',
    max_iter=500, 
    random_state=42, 
    verbose=True 
)

# Train model
clf.fit(x_train, y_train)

# Predict/evaluate
y_pred = clf.predict(x_test)
# print("Predictions:", y_pred)
# print("Actual:", y_test)
