
# Ali Behfarnia
# Created 2018, Editted 10/2024
# SimpleCNN Classifier
# Goal: To do binary classification to detect digits 0 or 8 or 5
# Dataset: MNIST. To reduce runtime, 2k samples are considered.

# =====================
# Step 0: Importing required libraries
# =====================
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import numpy as np
import warnings
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import callbacks  # Import callbacks here

warnings.filterwarnings("ignore")

# =====================
# Step 1: Loading data
# =====================
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"]
y = y.astype('int')

# Select only the first 500 samples
X = X[:2000]  # Keep only the first 500 samples
y = y[:2000]  # Keep only the first 500 samples

# Convert X to NumPy array for easier indexing
# X = X.values

# We need to turn this into a binary classification: is the number 8 or not?
y_binary = np.where((y == 8) | (y == 5) | (y == 3), 1, 0)  # Set 1 for '8' and 0 for everything else


# =====================
# Step 2: Plotting some samples (optional)
# =====================
# def plot_samples(X, y_binary, num_samples=9):
#     plt.figure(figsize=(10, 10))
#     for i in range(num_samples):
#         plt.subplot(3, 3, i + 1)
#         plt.imshow(X[i].reshape(28, 28), cmap="gray")
#         plt.title(f"Label: {y_binary[i]}")
#         plt.axis("off")
#     plt.show()

# plot_samples(X, y_binary)

# =====================
# Step 3: Train-validation-test split & Scaling
# =====================
# First, split into 90% train + 10% test
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.1, random_state=42)

# Now, further split the 90% train into 80% train + 10% validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)  # ~10% validation

# Scaling the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Reshape the data to be compatible with CNN (batch, height, width, channels)
X_train = X_train.reshape(-1, 28, 28, 1)
X_val = X_val.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# =====================
# Step 4: Defining the Highly Reduced CNN model
# =====================
def SimpleCNN():
    model = models.Sequential()
    model.add(layers.Input(shape=(28, 28, 1)))  # Input layer
    model.add(layers.Conv2D(8, (3, 3), activation='relu'))  # Reduced number of filters to 8
    model.add(layers.MaxPooling2D((2, 2)))  # Single pooling layer

    model.add(layers.Flatten())
    model.add(layers.Dense(16, activation='relu'))  # Reduced number of neurons in the dense layer to 16
    model.add(layers.Dense(1, activation='sigmoid'))  # Binary classification

    return model

model = SimpleCNN()

# =====================
# Step 5: Compiling the model
# =====================
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

# =====================
# Step 6: Defining the callback for epoch display every 20 epochs
# =====================
class EpochCallback(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: Loss = {logs['loss']:.4f}, Accuracy = {logs['accuracy']:.4f}")

epoch_callback = EpochCallback()

# =====================
# Step 7: Training the model (with validation and 20 epochs)
# =====================
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), 
                    callbacks=[epoch_callback], verbose=0)

# =====================
# Step 8: Evaluating the model on the test set
# =====================
# Predict on the test set
y_pred = model.predict(X_test)
y_pred_binary = np.where(y_pred > 0.5, 1, 0)

# Calculating evaluation metrics
auc = roc_auc_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred_binary)
precision = precision_score(y_test, y_pred_binary)
recall = recall_score(y_test, y_pred_binary)

print(f"AUC: {auc:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# =====================
# Step 9: Plotting training history
# =====================
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()
