import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import pickle


# Load dataset
df = pd.read_csv(r"C:\Users\saich\Downloads\Iris.csv")

# Display first few rows
print(df.head())

# Drop the 'Id' column if present (not needed for modeling)
if 'Id' in df.columns:
    df = df.drop(columns=['Id'])

# Separate features and target
X = df.drop(columns=['Species'])  # Features
y = df['Species']  # Target variable

# Encode target labels into numerical values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Initialize and train the Decision Tree Classifier
clf = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# # Predictions
# y_pred = clf.predict(X_test)

# # Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy:.2f}")

# # Classification Report
# print("Classification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))


# Save the trained model to a file
with open("iris_decisiontree.pkl", "wb") as file:
    pickle.dump(clf, file)

print("Model saved successfully!")

# # Load the model from the pickle file
# with open("decision_tree_model.pkl", "rb") as file:
#     loaded_model = pickle.load(file)

# print("Model loaded successfully!")

# # Verify by making predictions with the loaded model
# y_pred_loaded = loaded_model.predict(X_test)
# print(f"Loaded Model Accuracy: {accuracy_score(y_test, y_pred_loaded):.2f}")
