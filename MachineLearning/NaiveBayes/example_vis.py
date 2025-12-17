import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set a style for better visualization
sns.set_style("whitegrid")

# --- 1. Load Data ---
# The data.csv file is loaded into a pandas DataFrame.
try:
    # Assuming 'data.csv' is the uploaded file, which contains categorical data
    df = pd.read_csv('data.csv')
    df = df.iloc[:, 1:]  # Drop the initial index column if present
except FileNotFoundError:
    print("Error: data.csv not found. Using mock data for demonstration.")
    data = {
        'Outlook': ['Rainy', 'Rainy', 'Overcast', 'Sunny', 'Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Sunny',
                    'Rainy', 'Overcast', 'Overcast', 'Sunny'],
        'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild',
                        'Hot', 'Mild'],
        'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal',
                     'High', 'Normal', 'High'],
        'Windy': [False, True, False, False, False, True, True, False, False, False, True, True, False, True],
        'Play Golf': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
    }
    df = pd.DataFrame(data)

print("--- Original Data Sample (First 5 Rows) ---")
print(df.head())
print("-" * 50)

# --- 2. Prepare Data (Categorical Encoding) ---
# Naive Bayes requires numerical input, so we must encode categorical features.

# Separate features (X) and target (y)
X = df.drop('Play Golf', axis=1)
y = df['Play Golf']

# Initialize encoder
encoder = OrdinalEncoder()

# Fit and transform features (X)
X_encoded = encoder.fit_transform(X)

# Store the original features for plotting titles
X_df = df.drop('Play Golf', axis=1)

print("--- Encoded Features Sample (First 5 Rows) ---")
print(pd.DataFrame(X_encoded, columns=X.columns).head())
print("-" * 50)

# --- 3. Split Data ---
# Note: X_encoded is a numpy array.
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.30, random_state=42, stratify=y
)

# --- 4. Initialize and Train Model ---
model = CategoricalNB()
model.fit(X_train, y_train)

# --- 5. Evaluate Model ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Training Complete. Categorical Naive Bayes Accuracy: {accuracy:.2f}")
print("-" * 50)

# --- 6. Make a Prediction Example (Textual Output) ---
# Scenario: Rainy, Hot, High Humidity, Not Windy
new_sample_data = pd.DataFrame({
    'Outlook': ['Rainy'],
    'Temperature': ['Hot'],
    'Humidity': ['High'],
    'Windy': [False]
})

# Use the fitted encoder to transform the new sample
new_sample_encoded = encoder.transform(new_sample_data)

# Predict the result
prediction = model.predict(new_sample_encoded)
probabilities = model.predict_proba(new_sample_encoded)

print(f"New Sample: {new_sample_data.iloc[0].to_dict()}")
print(f"Prediction (Play Golf): {prediction[0]}")
class_labels = model.classes_
print(f"Probabilities: {probabilities[0]}")
for i, label in enumerate(class_labels):
    print(f"  P(Class={label}): {probabilities[0][i]:.4f}")
print("-" * 50)


# --- 7. Visualization Section ---

def plot_visualizations(df_full, y_test_labels, y_pred_labels, classes):
    """Generates and displays visualizations for the Naive Bayes model."""

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Naive Bayes Model Visualizations', fontsize=16)

    # --- Plot 1: Class Distribution (Target Balance) ---
    # Show the balance of the target variable in the dataset
    sns.countplot(x='Play Golf', data=df_full, ax=axes[0], palette='viridis')
    axes[0].set_title('1. Target Class Distribution (Full Data)')
    axes[0].set_xlabel('Play Golf?')
    axes[0].set_ylabel('Count')

    # --- Plot 2: Feature Distribution Example (Outlook) ---
    # Show how a key feature is distributed across the target classes
    sns.countplot(x='Outlook', hue='Play Golf', data=df_full, ax=axes[1], palette='flare')
    axes[1].set_title('2. Feature Distribution: Outlook vs. Target')
    axes[1].set_xlabel('Outlook')
    axes[1].set_ylabel('Count')
    axes[1].legend(title='Play Golf')

    # --- Plot 3: Confusion Matrix (Model Performance) ---
    # Show where the model made errors on the test set
    cm = confusion_matrix(y_test_labels, y_pred_labels, labels=classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=axes[2],
                cbar=False)
    axes[2].set_title('3. Confusion Matrix (Test Set)')
    axes[2].set_xlabel('Predicted Label')
    axes[2].set_ylabel('True Label')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to prevent overlap
    plt.show()


# Run the visualization function
plot_visualizations(df, y_test, y_pred, class_labels)