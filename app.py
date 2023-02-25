import streamlit as st
import pandas as pd
import pickle
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

# Load the iris dataset
iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Load the pre-trained model
model = pickle.load(open("model.pkl", "rb"))

# Define the prediction function
def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    data = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(data)
    species = iris.target_names[prediction[0]]
    return species

# Create the Streamlit app
def main():
    st.title("Vehicle Engine Failure Prediction")
    st.sidebar.header("User Input Parameters")

    # Define the user input widgets
    sepal_length = st.sidebar.slider("Year", 4.3, 7.9, 5.4, 0.1)
    sepal_width = st.sidebar.slider("Manufacturer", 2.0, 4.4, 3.4, 0.1)
    petal_length = st.sidebar.slider("Cold Start RPM", 1.0, 6.9, 1.3, 0.1)
    petal_width = st.sidebar.slider("Engine Temperature", 0.1, 2.5, 0.2, 0.1)

    # Make predictions and display the results
    species = predict_species(sepal_length, sepal_width, petal_length, petal_width)
    st.write("The Knock sensor health is :", species)

if __name__ == "__main__":
    main()
