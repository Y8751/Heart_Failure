import joblib
import pandas as pd
import streamlit as st
import tempfile
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

st.title("📊 Predict with Uploaded Model")

# Upload model
model_file = st.file_uploader("Upload a model file (.pkl or .h5)", type=["pkl", "h5"])

# Upload dataset
data_file = st.file_uploader("Upload a CSV file for prediction", type=["csv"])

if model_file is not None and data_file is not None:
    if st.button("Run Prediction"):
        try:
            # Load dataset
            df = pd.read_csv(data_file)

            # Backup original to display results later
            original_data = df.copy()

            # Identify target if present (optional)
            if 'HeartDisease' in df.columns:
                df = df.drop('HeartDisease', axis=1)

            # Try loading model
            if model_file.name.endswith(".pkl"):
                model = joblib.load(model_file)

                # Check if pipeline with preprocessor
                if hasattr(model, "predict"):
                    y_pred = model.predict(df)
                    st.success("✅ Prediction completed using Sklearn model.")
                else:
                    st.error("❌ Model format not supported.")

            elif model_file.name.endswith(".h5"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
                    tmp.write(model_file.getbuffer())
                    tmp_path = tmp.name
                keras_model = load_model(tmp_path)

                # Assume preprocessor is in another file (for real use) — here, we add a dummy one
                # This should match how you trained the keras model!
                numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                categorical_features = df.select_dtypes(include=['object']).columns.tolist()

                preprocessor = ColumnTransformer([
                    ("num", Pipeline([
                        ('imputer', SimpleImputer(strategy='median')),
                        ('scaler', StandardScaler())
                    ]), numeric_features),
                    ("cat", Pipeline([
                        ('imputer', SimpleImputer(strategy='most_frequent')),
                        ('onehot', OneHotEncoder(handle_unknown='ignore'))
                    ]), categorical_features)
                ])

                df_transformed = preprocessor.fit_transform(df)
                predictions = (keras_model.predict(df_transformed) > 0.5).astype("int32").flatten()
                y_pred = predictions
                st.success("✅ Prediction completed using Keras model.")

            else:
                st.error("❌ Unsupported model format.")
                y_pred = []

            # Display results
            if len(y_pred):
                original_data['Prediction'] = ["Heart Disease" if i == 1 else "No Heart Disease" for i in y_pred]
                st.subheader("📄 Predictions")
                st.dataframe(original_data)

        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")

else:
    st.info("📂 Please upload both a model file and a dataset to begin.")
