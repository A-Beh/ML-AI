# By: Ali Behfarnia, Edited 09/2024
# Transfer Learning - Part II: Transfer Learning
# 
# The evaluation metrics MSE, NMSE
# 3-fold validation, Full Batch Learning

# ============================
# Step 0: Libraries
# ============================
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt

# ============================
# Step 1: Data Generation: synthetic data for transfer learning
# ============================
num_samples = 100
X = np.random.rand(num_samples, 1) * 6  # Random values between 0 and 10
y_o = 2*(1 - np.exp(-X))  # Target function
noise = 0.2* np.random.normal(loc=0, scale=0.05, size=X.shape)
y = y_o +noise

# ============================
# Step 2: Data Prepration: Convert to PyTorch tensors
# ============================
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y)

# ============================
# Step 3: Define the Simple Neural Network
# ============================
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(1, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ============================
# Step 4: Instantiate and load the saved Part I model
# ============================
model_transfer = SimpleNN()
model_transfer.load_state_dict(torch.load('original_model.pth'))

# ============================
# Step 5: Freeze the layers except the last one
# ============================ 
for param in model_transfer.parameters():
    param.requires_grad = False
for param in model_transfer.fc3.parameters():  # Only train the last layer
    param.requires_grad = True

# Optimizer only updates the parameters that are trainable
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_transfer.parameters()))
criterion = nn.MSELoss()

# ============================
# Step 6: Train the model
# ============================
num_epochs = 1000
start_time = time.time()
for epoch in range(num_epochs):
    model_transfer.train()
    optimizer.zero_grad()
    outputs = model_transfer(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()

# Calculate elapsed time
elapsed_time = time.time() - start_time
print(f"Transfer Learning Time: {elapsed_time:.2f} seconds")


# ============================
# Step 7: Evaluate the model on the test set (prediction)
# ============================
num_samples = 100
X_test = np.linspace(0, 6, num_samples).reshape(-1, 1)
X_test_tensor = torch.FloatTensor(X_test)
y_pred = model_transfer(X_test_tensor).detach().numpy()

# ============================
# Step 8: Plot the results
# ============================
plt.figure(figsize=(10, 5))
plt.scatter(X, y, label='True Values', color='blue', alpha=0.5)
plt.plot(X_test, y_pred, label='Predictions', color='red')
plt.title(f'Transfer Learning: # samples {num_samples}, # epochs = {num_epochs}, training_time = {elapsed_time:.2f}s')
plt.xlabel('X')
plt.ylabel(r'$f(X) = 2 \cdot (1 - e^{-X})$')
plt.legend()
plt.grid()
plt.show()
