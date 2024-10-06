# By: Ali Behfarnia, Edited 09/2024
# Regression - Simple Neural Network
# The evaluation metrics MSE, NMSE
# 3-fold validation, Full Batch Learning

# ============================
# Step 0: Libraries
# ============================
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# ============================
# Step 1: Data Generation
# ============================
X = np.linspace(-20, 20, 1000).reshape(-1, 1)
y = 0.05*X**3 + X**2 - 6*X+ 20 * np.random.normal(size=X.shape)

# ============================
# Step 2: Normalization
# ============================
scaler_x = StandardScaler()
# scaler_y = StandardScaler()
x_norm = scaler_x.fit_transform(X)
# y_norm = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()  # Flatten y for compatibility

# Convert to PyTorch tensors
X_tensor = torch.tensor(x_norm, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# ============================
# Step 3: Train/Validation/Test Split
# ============================
X_train_val, X_test, y_train_val, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

# ============================
# Step 4: Define the Simple Neural Network
# ============================
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.hidden1 = nn.Linear(1, 100)  # First hidden layer
        self.hidden2 = nn.Linear(100, 100)  # Second hidden layer
        self.output = nn.Linear(100, 1)  # Output layer

    def forward(self, x):
        x = torch.relu(self.hidden1(x))  # Tanh activation for the first hidden layer
        x = torch.relu(self.hidden2(x))  # Tanh activation for the second hidden layer
        x = self.output(x)
        return x
    
# ============================
# Step 5: Instantiate the model, define loss and optimizer
# ==========================
model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Learning rate

# ============================
# Step 6: Train the model
# ============================
num_epochs = 2000  # Number of epochs
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    
    optimizer.zero_grad()  # Clear gradients
    outputs = model(X_train)  # Forward pass
    loss = criterion(outputs, y_train)  # Compute loss
    loss.backward()  # Backward pass
    optimizer.step()  # Update weights
    
    train_losses.append(loss.item())
    
    # Validate the model
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)
        val_losses.append(val_loss.item())
    
    # Print every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}')

# ============================
# Step 7: Evaluate the model on the test set
# ============================
model.eval()
with torch.no_grad():
    y_pred = model(X_test)


# Calculate NMSE
mse = mean_squared_error(y_test, y_pred)
nmse = mse / np.var(y_test.numpy())  # NMSE calculation

print(f'Normalized Mean Squared Error (NMSE) on test set: {nmse:.4f}')

# ============================
# Step 8: Plot the results
# ============================
plt.figure(figsize=(10, 6))
plt.scatter(X_test.numpy(), y_test.numpy(), label='Test data', color='green', alpha=0.5)
plt.scatter(X_test.numpy(), y_pred.numpy(), label='Predicted data', color='red', linewidth=2)
plt.title('Simple Neural Network Regression on Noisy Cubic Data')
plt.xlabel('X')
plt.ylabel('y')
plt.grid(True)
plt.legend()
plt.show()
