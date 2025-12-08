import numpy as np
import os

base_path = '../data/timeseries/uci-har/UCI HAR Dataset'

# Load training data
X_train_path = os.path.join(base_path, 'train/X_train.txt')
y_train_path = os.path.join(base_path, 'train/y_train.txt')

# Load test data
X_test_path = os.path.join(base_path, 'test/X_test.txt')
y_test_path = os.path.join(base_path, 'test/y_test.txt')

print("Loading UCI-HAR training data...")
X_train = np.loadtxt(X_train_path, dtype=np.float32)
y_train = np.loadtxt(y_train_path, dtype=np.uint8) - 1  # Convert to 0-indexed

print(f"Training data shape: {X_train.shape}, labels shape: {y_train.shape}")
print(f"Training samples: {X_train.shape[0]}, features: {X_train.shape[1]}")

print("Loading UCI-HAR test data...")
X_test = np.loadtxt(X_test_path, dtype=np.float32)
y_test = np.loadtxt(y_test_path, dtype=np.uint8) - 1  # Convert to 0-indexed

print(f"Test data shape: {X_test.shape}, labels shape: {y_test.shape}")

# Save as binary
output_base = '../data/timeseries/uci-har/UCI HAR Dataset'

train_samples_path = os.path.join(output_base, 'train-samples.bin')
train_labels_path = os.path.join(output_base, 'train-labels.bin')
test_samples_path = os.path.join(output_base, 'test-samples.bin')
test_labels_path = os.path.join(output_base, 'test-labels.bin')

print(f"Saving training samples to {train_samples_path}...")
X_train.tofile(train_samples_path)

print(f"Saving training labels to {train_labels_path}...")
y_train.tofile(train_labels_path)

print(f"Saving test samples to {test_samples_path}...")
X_test.tofile(test_samples_path)

print(f"Saving test labels to {test_labels_path}...")
y_test.tofile(test_labels_path)

print("UCI-HAR preprocessing complete!")
print(f"Train: {X_train.shape[0]} samples x {X_train.shape[1]} features")
print(f"Test: {X_test.shape[0]} samples x {X_test.shape[1]} features")
