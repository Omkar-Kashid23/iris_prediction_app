import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open("best_model.pkl", "rb") as file:
    model = pickle.load(file)

# Streamlit app title
st.title("🌸 Iris Flower Species Prediction App")

st.write("Enter the flower measurements below to predict the Iris species:")

# Input fields for the features
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2)

# Prediction button
if st.button("🔍 Predict"):
    # Prepare the input array
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Make prediction
    prediction = model.predict(input_data)

    # Display result
    st.success(f"🌼 The predicted Iris species is: **{prediction[0]}**")
