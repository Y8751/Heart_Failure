import os
import joblib
from tensorflow.keras.models import load_model
import streamlit as st

st.write("🗂 Current working directory:", os.getcwd())
st.write("📁 Files in saved_models/:", os.listdir("saved_models") if os.path.exists("saved_models") else "Folder not found.")

st.title("📁 Load Saved Model")

# Define the folder path
models_path = "saved_models"

# Create the folder if it doesn't exist
os.makedirs(models_path, exist_ok=True)

# Check for available models
model_files = os.listdir(models_path)

if model_files:
    selected_file = st.selectbox("Select a saved model", model_files)

    if st.button("Load Model"):
        try:
            if selected_file.endswith(".pkl"):
                model = joblib.load(os.path.join(models_path, selected_file))
                st.success(f"✅ Sklearn model loaded: {selected_file}")
            elif selected_file.endswith(".h5"):
                model = load_model(os.path.join(models_path, selected_file))
                st.success(f"✅ Keras model loaded: {selected_file}")
            else:
                st.error("❌ Unsupported file format.")
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
else:
    st.warning("⚠️ No models found in 'saved_models/'. Please train and save your models first.")
