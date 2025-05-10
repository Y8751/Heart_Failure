import streamlit as st
import os
import joblib
from tensorflow.keras.models import load_model

st.title("📁 Load Saved Model")

# List saved models
model_files = os.listdir("saved_models")
selected_file = st.selectbox("Select a saved model", model_files)

# Load the model when button is clicked
if st.button("Load Model"):
    if selected_file.endswith(".pkl"):
        model = joblib.load(f"saved_models/{selected_file}")
        st.success(f"✅ Sklearn model loaded: {selected_file}")
    elif selected_file.endswith(".h5"):
        model = load_model(f"saved_models/{selected_file}")
        st.success(f"✅ Keras model loaded: {selected_file}")
    else:
        st.error("❌ Unsupported file type.")
