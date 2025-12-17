import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# --- 1. Load Data ---
# Using the well-known Iris dataset, which is ideal for multi-class classification
# and demonstrating SVM's performance on continuous features.
iris = datasets.load_iris()
X = iris.data  # Features (Sepal Length, Sepal Width, Petal Length, Petal Width)
y = iris.target # Target (0: setosa, 1: versicolor, 2: virginica)

print("--- Iris Dataset Loaded ---")
print(f"Total Samples: {len(X)}")
print(f"Total Features: {X.shape[1]}")
print(f"Target Classes: {iris.target_names}")
print("-" * 50)

# --- 2. Split Data ---
# Split the data into 70% training and 30% testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# --- 3. Initialize and Train Model ---
# Using Support Vector Classifier (SVC) with a Radial Basis Function (RBF) kernel,
# which is effective for non-linear decision boundaries.
# C=1.0 is the regularization parameter.
# gamma='scale' determines the influence of individual training samples.
model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)

print("Training SVM Model with RBF Kernel...")
model.fit(X_train, y_train)
print("Training complete.")
print("-" * 50)

# --- 4. Evaluate Model ---
# Predict on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"RBF SVM Model Accuracy on Test Set: {accuracy:.4f}")

# Display detailed performance metrics
report = classification_report(y_test, y_pred, target_names=iris.target_names)
print("\nClassification Report:")
print(report)
print("-" * 50)

# --- 5. Make a Prediction Example ---
# Sample data for a new Iris flower:
# (Sepal L=5.1, Sepal W=3.5, Petal L=1.4, Petal W=0.2) - looks like Setosa
new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])

prediction_index = model.predict(new_sample)[0]
predicted_class = iris.target_names[prediction_index]

print(f"New Sample Features: {new_sample[0]}")
print(f"Predicted Class Index: {prediction_index}")
print(f"Predicted Class Name: {predicted_class}")