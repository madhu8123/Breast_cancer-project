# Breast_cancer-project using machine learning
# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the breast cancer dataset
data = load_breast_cancer()

# Create DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Split data into features and target
X = df.drop('target', axis=1)
y = df['target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Function to suggest foods for breast cancer prevention
def suggest_foods():
    print("Foods that may help prevent breast cancer:")
    foods = [
        "Fruits and vegetables (especially cruciferous vegetables like broccoli, kale, and cabbage)",
        "Berries (such as blueberries, strawberries, and raspberries)",
        "Healthy fats (such as those found in olive oil, avocado, and fatty fish like salmon)",
        "Green tea",
        "Turmeric",
        "Garlic",
        "Whole grains",
        "Nuts and seeds",
        "Legumes (beans, lentils, chickpeas)",
        "Limiting processed and red meats"
    ]
    for food in foods:
        print("-", food)

# Function to suggest tablets for breast cancer prevention
def suggest_tablets():
    print("Tablets/supplements that may help prevent or manage breast cancer:")
    tablets = [
        "Vitamin D",
        "Calcium",
        "Omega-3 fatty acids",
        "Coenzyme Q10 (CoQ10)",
        "Curcumin (the active ingredient in turmeric)",
        "Green tea extract",
        "DIM (diindolylmethane)",
        "Resveratrol",
        "Melatonin",
        "Probiotics"
    ]
    for tablet in tablets:
        print("-", tablet)

# Call the function to suggest foods for breast cancer prevention
suggest_foods()

# Call the function to suggest tablets for breast cancer prevention
suggest_tablets()
