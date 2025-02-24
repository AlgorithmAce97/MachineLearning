import pickle

# Load the model from the pickle file
with open("iris_decisiontree.pkl", "rb") as file:
    loaded_model = pickle.load(file)

print(loaded_model)

print("Model loaded successfully!")

