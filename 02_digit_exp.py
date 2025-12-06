import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Load the digits dataset
digits = load_digits()
X, y = digits.data, digits.target

print("Dataset Overview:")
print(f"Number of samples: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")
print(f"Number of classes: {len(np.unique(y))}")
print(f"Classes: {np.unique(y)}")
print(f"Image shape: {digits.images[0].shape}")

# Visualize some sample digits
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap='gray')
    ax.set_title(f'Label: {digits.target[i]}')
    ax.axis('off')
plt.tight_layout()
plt.savefig('digits_samples.png')
plt.show()

# Data preprocessing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nAfter preprocessing:")
print(f"Training set shape: {X_train_scaled.shape}")
print(f"Test set shape: {X_test_scaled.shape}")

# Basic statistics
print("\nBasic Statistics:")
print(f"Mean pixel value: {np.mean(X):.2f}")
print(f"Std pixel value: {np.std(X):.2f}")
print(f"Min pixel value: {np.min(X)}")
print(f"Max pixel value: {np.max(X)}")

# Class distribution
unique, counts = np.unique(y, return_counts=True)
plt.figure(figsize=(8, 5))
plt.bar(unique, counts)
plt.xlabel('Digit')
plt.ylabel('Count')
plt.title('Class Distribution')
plt.xticks(unique)
plt.savefig('class_distribution.png')
plt.show()

print("\nClass distribution:")
for digit, count in zip(unique, counts):
    print(f"Digit {digit}: {count} samples")