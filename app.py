import joblib
from tensorflow.keras.models import load_model
import streamlit as st
import tempfile

st.title("📁 Upload and Load Saved Model")

# Upload model file
uploaded_file = st.file_uploader("Upload a model file (.pkl or .h5)", type=["pkl", "h5"])

if uploaded_file is not None:
    if st.button("Load Model"):
        try:
            if uploaded_file.name.endswith(".pkl"):
                # Load scikit-learn model
                model = joblib.load(uploaded_file)
                st.success(f"✅ Sklearn model loaded: {uploaded_file.name}")

            elif uploaded_file.name.endswith(".h5"):
                # Temporarily save and load Keras model
                with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    tmp_path = tmp.name
                model = load_model(tmp_path)
                st.success(f"✅ Keras model loaded: {uploaded_file.name}")

            else:
                st.error("❌ Unsupported file format. Only .pkl and .h5 are supported.")

        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
else:
    st.info("📂 Please upload a model file to begin.")
