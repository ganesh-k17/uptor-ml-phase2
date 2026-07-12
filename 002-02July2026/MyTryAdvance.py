# ann_csv_keras.py — TRUE DEEP LEARNING with Keras
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder


# 1. Load CSV
df = pd.read_csv("customer_churn.csv")

# Format the columns (string columns are encoded to numeric)
obj  = LabelEncoder()
for x in df.columns:
    if df[x].dtype in ['object', 'str']:
        df[x] = obj.fit_transform(df[x])


X = df.drop("Churn", axis=1).values
y = df["Churn"].values

# 2. Split and scale
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 3. Build the model (this is the DEEP LEARNING style)
model = keras.Sequential([
    layers.Dense(64, activation=None, input_shape=(x_train.shape[1],)),
    layers.BatchNormalization(),        # 🔥 stabilizes learning
    layers.Activation('relu'),         # apply non-linearity after normalization
    layers.Dropout(0.3),

    layers.Dense(32, activation=None),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.3),

    layers.Dense(16, activation=None),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.3),

    layers.Dense(1, activation='sigmoid')  # output layer
])


# 4. COMPILE the model (this is what you mentioned!)
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',  # or 'mse' for some tasks [web:62]
    metrics=['accuracy']
)

# 5. Train with epochs and batch_size
model.fit(
    x_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# 6. Predict (this is model.predict() you mentioned!)
y_pred_prob = model.predict(x_test)
y_pred = (y_pred_prob > 0.5).astype(int)

print("Accuracy:", np.mean(y_pred == y_test))