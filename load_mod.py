import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open("iris_decisiontree.pkl", "rb") as file:
    loaded_model = pickle.load(file)

# Title of the app
st.title("Iris Species Prediction")

# User inputs
sepal_length = st.number_input("Sepal Length", min_value=0.0, format="%.2f")
sepal_width = st.number_input("Sepal Width", min_value=0.0, format="%.2f")
petal_length = st.number_input("Petal Length", min_value=0.0, format="%.2f")
petal_width = st.number_input("Petal Width", min_value=0.0, format="%.2f")

# Predict button
if st.button("Predict"):
    # Convert inputs into a NumPy array
    input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Make a prediction
    prediction = loaded_model.predict(input_features)
    
    # Mapping numerical prediction back to original class labels
    species_mapping = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}
    predicted_species = species_mapping[prediction[0]]
    
    # Display the predicted species
    st.success(f"Predicted Iris Species: {predicted_species}")
