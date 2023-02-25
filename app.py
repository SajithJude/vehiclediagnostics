import streamlit as st
import pandas as pd
import pickle
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.write("""
The health of a knock sensor can be affected by several factors, including normal wear and tear, exposure to extreme temperatures or vibration, and contamination by oil, fuel, or other engine fluids. 

#Here are some indicators that may suggest that a knock sensor is degraded or malfunctioning:

1. Engine performance issues: A faulty knock sensor may cause the engine to misfire, hesitate, or run roughly. This is because the engine control module (ECM) uses the knock sensor signal to adjust the ignition timing and fuel delivery to prevent knocking or pinging.

2. Reduced fuel economy: A degraded knock sensor may cause the engine to operate less efficiently, resulting in reduced fuel economy.

3. Illuminated check engine light: A malfunctioning knock sensor can trigger the check engine light to come on. This is because the ECM relies on the knock sensor signal to detect engine knock and adjust the engine's operating parameters.

4. Knocking or pinging sounds: A failed knock sensor may not detect engine knock, leading to audible knocking or pinging sounds from the engine. However, it's worth noting that these sounds can also be caused by other issues, such as a damaged piston or a malfunctioning fuel injector.
""")

# Load the iris dataset
iris = datasets.load_iris()

df= pd.read_csv("senal dataset - Sheet1 (1).csv")
X = df[['year', 'manufacturer','Coldstart_rpm','heated_rpm','engine_temp']]
y = df['knock_sensor']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)



# Load the pre-trained model
# model = pickle.load(open("model.pkl", "rb"))

# Define the prediction function
def predict_species(sepal_length, sepal_width, petal_length, petal_width,heated_rpm):
    data = [[sepal_length, sepal_width, petal_length, petal_width,heated_rpm]]
    prediction = model.predict(data)
    species = st.write(prediction)
    return species

# Create the Streamlit app
def main():
    st.title("Vehicle Engine Failure Prediction")
    st.sidebar.header("User Input Parameters")

    # Define the user input widgets
    sepal_length = st.sidebar.slider("Year", min_value=2008,
    max_value=2022,
    value=(2010),
    step=1)

    sepal_width = st.sidebar.slider("Manufacturer", 1,2,3)
    petal_length = st.sidebar.slider("Cold Start RPM", min_value=1000,
    max_value=1400,
    value=(1000),
    step=1)
    heated_rpm = st.sidebar.slider("Heated RPM", min_value=2000,
    max_value=3000,
    value=(2010),
    step=1)
    petal_width = st.sidebar.slider("Engine Temperature", min_value=190,
    max_value=260,
    value=(200),
    step=1)

    # Make predictions and display the results
    if st.button("Predict"):
        
      species = predict_species(sepal_length, sepal_width, petal_length, petal_width,heated_rpm)
      st.write("The Knock sensor health is :", species)

if __name__ == "__main__":
    main()
