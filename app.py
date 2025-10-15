import streamlit as st
import pickle
import numpy as np

# Load the trained model
try:
    with open('1_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    st.error("Error: The model file '1_model.pkl' was not found. Please make sure it's in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    st.stop()

# Set up the Streamlit app title and description
st.set_page_config(page_title="Iris Species Classifier", page_icon="🌸")
st.title("Iris Species Classifier")
st.write("Enter the measurements of the iris flower to classify its species.")

# Create input widgets in a sidebar for a cleaner layout
st.sidebar.header("User Input Features")

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal Length (cm)', 4.3, 7.9, 5.8)
    sepal_width = st.sidebar.slider('Sepal Width (cm)', 2.0, 4.4, 3.1)
    petal_length = st.sidebar.slider('Petal Length (cm)', 1.0, 6.9, 3.7)
    petal_width = st.sidebar.slider('Petal Width (cm)', 0.1, 2.5, 1.2)
    data = {
        'sepal_length': sepal_length,
        'sepal_width': sepal_width,
        'petal_length': petal_length,
        'petal_width': petal_width
    }
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    return features

# Get user input
user_data = user_input_features()

# Make prediction when the 'Predict' button is clicked
if st.sidebar.button('Predict'):
    prediction = model.predict(user_data)
    
    # Get the class names from the model
    # The saved model object has a 'classes_' attribute containing the species names.
    species_names = model.classes_
    predicted_species = species_names[prediction[0]]

    st.subheader('Prediction')
    st.write(f"The predicted Iris species is: **{predicted_species}**")
    
    # You can add more info, like the confidence score or visualizations, here.
    # The decision_function is useful for SVC.
    # It gives the distance of the sample to the separating hyperplane.
    
    try:
        decision_scores = model.decision_function(user_data)
        st.subheader('Prediction Confidence (Decision Scores)')
        scores_df = pd.DataFrame(decision_scores, columns=species_names)
        st.dataframe(scores_df)
    except:
        st.info("Decision scores are not available for this model type.")
