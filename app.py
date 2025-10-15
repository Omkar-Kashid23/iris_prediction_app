import streamlit as st
import pickle
import numpy as np
import pandas as pd

# --- Page Config ---
st.set_page_config(
    page_title="🌸 Iris Species Classifier",
    page_icon="🌸",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for better styling ---
st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.3rem 1rem;
    }
    .prediction-box {
        background-color: #e8f5e9;
        padding: 1.2rem;
        border-radius: 10px;
        text-align: center;
        margin-top: 1rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    }
    .confidence-header {
        color: #2e7d32;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# --- Load Model ---
@st.cache_resource
def load_model():
    try:
        with open('1_model.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("❌ Model file `1_model.pkl` not found. Please ensure it's in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        st.stop()

model = load_model()

# --- App Header ---
st.title("🌸 Iris Species Classifier")
st.markdown("🔍 Predict the species of an Iris flower based on its measurements.")

# --- Input Section ---
st.sidebar.header("📏 Input Flower Measurements")

# Use columns for a cleaner layout
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.slider("Sepal Length (cm)", 4.3, 7.9, 5.8, 0.1)
    petal_length = st.slider("Petal Length (cm)", 1.0, 6.9, 3.7, 0.1)

with col2:
    sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.4, 3.1, 0.1)
    petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.2, 0.1)

# Prepare input array
user_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# --- Real-time Prediction (no button needed) ---
try:
    prediction = model.predict(user_data)
    species_names = model.classes_
    predicted_species = species_names[prediction[0]]

    # Display prediction with styling
    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
    st.subheader("🎯 Prediction Result")
    st.markdown(f"### **{predicted_species}**")
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Confidence / Decision Scores (if available) ---
    try:
        decision_scores = model.decision_function(user_data)
        st.markdown("### 💯 Prediction Confidence (Decision Scores)")
        scores_df = pd.DataFrame(decision_scores, columns=species_names).T
        scores_df.columns = ["Score"]
        scores_df["Score"] = scores_df["Score"].round(3)
        scores_df = scores_df.sort_values("Score", ascending=False)
        st.dataframe(scores_df.style.highlight_max(axis=0, color='#e8f5e9'))
    except AttributeError:
        st.info("ℹ️ Confidence scores are not available for this model type.")

except Exception as e:
    st.error(f"❌ Prediction failed: {e}")

# --- Footer ---
st.markdown("---")
st.caption("Built with ❤️ using Streamlit | Trained on the Iris dataset")
