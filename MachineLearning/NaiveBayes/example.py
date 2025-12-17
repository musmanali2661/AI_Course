import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder

# --- 1. Load Data ---
# The data.csv file is loaded into a pandas DataFrame.
try:
    # Assuming 'data.csv' is the uploaded file, which contains categorical data
    df = pd.read_csv('data.csv')
    df = df.iloc[:, 1:] # Drop the initial index column if present
except FileNotFoundError:
    print("Error: data.csv not found. Using mock data for demonstration.")
    data = {
        'Outlook': ['Rainy', 'Rainy', 'Overcast', 'Sunny', 'Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Sunny', 'Rainy', 'Overcast', 'Overcast', 'Sunny'],
        'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
        'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
        'Windy': [False, True, False, False, False, True, True, False, False, False, True, True, False, True],
        'Play Golf': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
    }
    df = pd.DataFrame(data)

print("--- Original Data Sample (First 5 Rows) ---")
print(df.head())
print("-" * 50)


# --- 2. Prepare Data (Categorical Encoding) ---
# Naive Bayes requires numerical input, so we must encode categorical features.
# OrdinalEncoder is suitable for converting strings to integers for nominal data.

# Separate features (X) and target (y)
X = df.drop('Play Golf', axis=1)
y = df['Play Golf']

# Initialize encoder
encoder = OrdinalEncoder()

# Fit and transform features (X)
# We handle the boolean 'Windy' column explicitly if it's the only non-string column
X_encoded = encoder.fit_transform(X)

# Map the columns back to a DataFrame for readability (optional)
X_encoded_df = pd.DataFrame(X_encoded, columns=X.columns)

print("--- Encoded Features Sample (First 5 Rows) ---")
print(X_encoded_df.head())
print("-" * 50)


# --- 3. Split Data ---
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.30, random_state=42
)

# --- 4. Initialize and Train Model ---
# CategoricalNB is used because all features are discrete categories.
model = CategoricalNB()
model.fit(X_train, y_train)


# --- 5. Evaluate Model ---
accuracy = model.score(X_test, y_test)
print(f"Model Training Complete. Categorical Naive Bayes Accuracy: {accuracy:.2f}")
print("-" * 50)


# --- 6. Make a Prediction Example ---
# Scenario: Rainy, Hot, High Humidity, Not Windy
new_sample_data = pd.DataFrame({
    'Outlook': ['Rainy'],
    'Temperature': ['Hot'],
    'Humidity': ['High'],
    'Windy': [False]
})

# Use the fitted encoder to transform the new sample
# Ensure columns match the training order
new_sample_encoded = encoder.transform(new_sample_data)

# Predict the result
prediction = model.predict(new_sample_encoded)
probabilities = model.predict_proba(new_sample_encoded)

print(f"New Sample: {new_sample_data.iloc[0].to_dict()}")
print(f"Prediction (Play Golf): {prediction[0]}")
# Show probabilities for each class (e.g., ['No', 'Yes'] - depends on internal class ordering)
class_labels = model.classes_
print(f"Probabilities: {probabilities[0]}")
for i, label in enumerate(class_labels):
    print(f"  P(Class={label}): {probabilities[0][i]:.4f}")