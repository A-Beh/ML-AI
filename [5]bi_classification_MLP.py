# By: Ali Behfarnia, Editted 09/2024
# Simple MLP
# Binary classification for a simple dataset
# The metrics (AUC, F1, etc) are shown on testset (not validation)

# ============================
# Step 0: Libraries
# ============================
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
# from torch.nn import ReLU
# from torch.nn import Sigmoid
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader 
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, accuracy_score
from sklearn.model_selection import KFold

# ============================
# Step I: Data Loading
# ============================
df = pd.read_csv('binary_classification_dataset.csv')
# print(df)

# khondane Mostaghim
# X = df[['age', 'income', 'marketing_spend']].values
# y = df['buy_product'].values

data = df.values
X = data[:,:-1]
y = data[:,-1].reshape(-1,1)

print("The sahpe of the input is: ", X.shape)
print("The shape of the label is: ", y.shape)


# ============================
# Section II: Standardization
# ============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_tensor = torch.tensor(X_scaled, dtype=torch.float)
y_tensor = torch.tensor(y, dtype=torch.float)


# ============================
# Section III: Data Split
# ============================
X_train, X_temp, y_train, y_temp = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# y_train = y_train.view(-1, 1)
# y_val = y_val.view(-1, 1)
# y_test = y_test.view(-1, 1)

y_train = y_train.reshape(-1, 1)
y_val = y_val.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# ============================
# Section IV: PyTorch data preparation 
# ============================
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)


# ============================
# Section V: mini-Batch
# ============================
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ============================
# Section V: NN Moldel Definition
# ============================
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(3, 16)  # 3 input features
        self.fc2 = nn.Linear(16, 1)  # 1 output (binary classification)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

model = MLP()
criterion = nn.BCELoss()    
optimizer = Adam(model.parameters(), lr=0.001)

# ============================
# Section VI: Loop
# ============================

n_epochs = 300
train_loss_history = []
val_loss_history = []

for epoch in range(n_epochs):
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_loss_history.append(train_loss)
    
    # Validation step
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            val_loss += loss.item()


    val_loss /= len(val_loader)
    val_loss_history.append(val_loss)
    if (epoch+1) % 20 == 0:
        print(f'Epoch {epoch+1}/{n_epochs}, \
            Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')


# ============================
# Section VI: Plots
# ============================
plt.plot(train_loss_history, label='Train Loss', color = 'green', linestyle='-')
plt.plot(val_loss_history, label='Val Loss', color ='blue', linestyle='--')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Prediction Error vs. Epochs')
plt.legend()
plt.grid(True)


# ============================
# Section VII: Test Evaluation: AUC, Precision, Recall, F1 Score
# ============================
all_test_preds = []
all_test_labels = []

model.eval()
test_loss = 0.0
# Disable gradient calculation for testing
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        test_loss += loss.item()
        
        # Collect predictions and labels
        all_test_preds.append(y_pred.numpy())  # Predictions
        all_test_labels.append(y_batch.numpy())  # True labels

# Calculate average test loss
test_loss /= len(test_loader)

# Convert lists to arrays for evaluation
all_test_preds = np.concatenate(all_test_preds, axis=0)
all_test_labels = np.concatenate(all_test_labels, axis=0)



# Binary classification: threshold predictions at 0.5
all_test_preds_binary = (all_test_preds > 0.5).astype(int)

# ============================
# Test Set Evaluation Metrics
# ============================
test_auc = roc_auc_score(all_test_labels, all_test_preds)
test_f1 = f1_score(all_test_labels, all_test_preds_binary)
test_precision, test_recall, _ = precision_recall_curve(all_test_labels, all_test_preds)

# Print the final test results
print(f'Test AUC: {test_auc:.4f}, \
       Test F1: {test_f1:.4f}, Test Precision: {test_precision[-1]:.4f}, \
        Test Recall: {test_recall[-1]:.4f}')

plt.show()
