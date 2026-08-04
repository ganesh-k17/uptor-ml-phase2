import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Social_Network_Ads.csv")
X = df[['Age', 'EstimatedSalary']]
Y = df['Purchased']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(X_train)

X_train = torch.tensor(
    X_train,
    dtype=torch.float32
)
print(X_train)


X_test = torch.tensor(
    X_test,
    dtype=torch.float32
)

y_train = torch.tensor(
    y_train.values.reshape(-1,1),
    dtype=torch.float32
)

y_test = torch.tensor(
    y_test.values.reshape(-1,1),
    dtype=torch.float32
)

model = nn.Sequential(

    nn.Linear(2,8),
    nn.ReLU(),

    nn.Linear(8,4),
    nn.ReLU(),

    nn.Linear(4,1),
    nn.Sigmoid() # activation function for binary classification
)

print(model)

criterion = nn.BCELoss()

optimizer = torch.optim.Adam(
    model.parameters(),     
    lr=0.001
)

epochs = 100

for epoch in range(epochs):
  # Forward Propagation
  outputs = model(X_train)
  # Loss Calculation
  loss = criterion(
    outputs, y_train
  )

  # Clear Old Gradients
  optimizer.zero_grad()

  # Backpropagation
  loss.backward()


  # Weight Update
  optimizer.step()

  if (epoch + 1) % 10 == 0:
  
          print(
              f"Epoch [{epoch+1}/{epochs}] "
              f"Loss = {loss.item():.4f}"
          )

# Testing the model
with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()

#acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
acc = y_predicted_cls.eq(y_test).sum().item() / len(y_test)
print(f"Accuracy = {acc:.4f}")