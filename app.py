import streamlit as st
import pandas as pd
import pickle
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from PIL import Image

st.set_page_config(page_title="Senal-K", page_icon=":car:", layout="wide", initial_sidebar_state="expanded")

# Set dark theme
st.markdown(
    """
    <style>
    .reportview-container {
        background: #0f141c;
        color: yellow;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the pre-trained model
# model = pickle.load(open("model.pkl", "rb"))

# Define the prediction function
def predict_species(sepal_length, sepal_width, petal_length, petal_width,heated_rpm):
    data = [[sepal_length, sepal_width, petal_length, petal_width,heated_rpm]]
    prediction = model.predict(data)
    # st.write( prediction)
    return prediction[0]

# Create the Streamlit app
def main():

    # Define the profile picture
    profile_image = Image.open("315770010_220541507026613_7570429809574094261_n.jpg").resize((150, 150))

    # Define the sidebar content
    st.sidebar.image(profile_image, use_column_width=True)
    st.sidebar.title("Senal Kariyawasam")
    st.sidebar.write("Final Year Undergraduate")
    st.sidebar.write("IIT (University of Westminster)")
    cola, colb = st.columns(2, gap="medium")
    # st.set_column_spacing(50)

    # Define the user input widgets
    sepal_length = cola.slider("Year", min_value=2008,
    max_value=2022,
    value=(2010),
    step=1)

    sepal_width = cola.slider("Manufacturer",  min_value=1,
    max_value=5,
    value=(2),
    step=1)
    petal_length = cola.slider("Cold Start RPM", min_value=1000,
    max_value=1500,
    value=(1000),
    step=1)
    heated_rpm = cola.slider("Heated RPM", min_value=1500,
    max_value=6000,
    value=(2010),
    step=1)
    petal_width = cola.slider("Engine Temperature", min_value=190,
    max_value=225,
    value=(200),
    step=1)
    arr = [" A faulty knock sensor may cause the engine to misfire, hesitate, or run roughly. This is because the engine control module (ECM) uses the knock sensor signal to adjust the ignition timing and fuel delivery to prevent knocking or pinging."," A degraded knock sensor may cause the engine to operate less efficiently, resulting in reduced fuel economy."," A malfunctioning knock sensor can trigger the check engine light to come on. This is because the ECM relies on the knock sensor signal to detect engine knock and adjust the engine's operating parameters."," A failed knock sensor may not detect engine knock, leading to audible knocking or pinging sounds from the engine. However, it's worth noting that these sounds can also be caused by other issues, such as a damaged piston or a malfunctioning fuel injector"]
    pred = ["Engine performance issues", "Reduced fuel economy", "Illuminated check engine light","Knocking or pinging sounds"]
    # Make predictions and display the results
    # if st.button("Predict"):
    rep = ["Colombo Auotomoile stores: +9452525252525", "Senal Garage stores: +94552525522"," Prkroad Service station : +827287282","Wellawate Vehicle Garage : +947828282"]
    sol = ["Replace the knock sensor: The most common solution to a faulty knock sensor is to replace it with a new one. This is a relatively simple repair that can be done by a mechanic or experienced DIYer. Make sure to use a high-quality replacement sensor to ensure proper function", "Check for damaged wiring: Sometimes the problem may not be the sensor itself, but damaged wiring leading to the sensor. Check for any frayed, corroded, or broken wires, and repair or replace them as needed.","Check the engine control module (ECM): In some cases, a faulty ECM may be causing issues with the knock sensor signal. If you've ruled out the knock sensor and wiring, it may be worth having a professional mechanic diagnose the issue with the ECM.","Clean the engine: A buildup of carbon deposits on the engine can sometimes cause issues with knock sensor signals. A thorough engine cleaning may help to resolve the problem.","Check the fuel quality: Poor quality fuel can sometimes cause engine knock and damage the knock sensor. Make sure to use high-quality fuel with the proper octane rating for your vehicle."]
    species = predict_species(sepal_length, sepal_width, petal_length, petal_width,heated_rpm)
   
    man =["Toyota ","Renault","Hyundai","Mercedes","Audi"]
    colb.header("Parameters Summary")
    colb.write("Model Manufactured Year :"+ str(sepal_length))
    colb.write("Manufactuerd Company :"+ str(man[sepal_width]))
    colb.write("Cold Start RPM :"+ str(petal_length))
    colb.write("Heated RPM :"+ str(heated_rpm))
    colb.write("Engine temperature :"+ str(petal_width)+" degree celsius")
    colb.header("Model Prediction:")
    colb.write(pred[species])

    with st.beta_expander("Reason for issue (click to expand)"):
      st.write( arr[species])

    with st.beta_expander("How to Fix this issue (click to expand)"):
      st.write(sol[species])

    with st.beta_expander("Mechanics and Parts Contact info (click to expand)"):
      st.write(rep[species])

if __name__ == "__main__":
    st.title("Vehicle Engine Failure Diagnosis System")
    st.markdown("#This Platform is the UI implementation made by selecting the highest performing Machine learning Model which is the Random forest Classifier")
    st.markdown("<a href='https://colab.research.google.com/drive/1i6lpmhUedo8ZnxQuOPSvj8A5ibI2LnCc?usp=sharing' target='_blank'>Click this link to access Colab notebook implementation of selecting the best model</a>", unsafe_allow_html=True)
    with st.beta_expander("How To Use This Tool (click to expand)"):
      st.write("Adjust the sliders and set the parameter with you Vehicles Manufacturing details and diagnostic readings, and the most suitable prediction will be shown on the right, expand the other buttons to reveal further information about diagnostics and recomendations from the system")
    df= pd.read_csv("senal dataset - Sheet1 (1).csv")
    X = df[['year', 'manufacturer','Coldstart_rpm','heated_rpm','engine_temp']]
    y = df['knock_sensor']
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)



    main()
