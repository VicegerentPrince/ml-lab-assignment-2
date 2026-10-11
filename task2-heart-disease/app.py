import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Heart Disease Prediction")

# Load the trained model saved by the notebook
model_path = Path(__file__).parent / "svc_trained_model.pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)

# Same label encoders as the notebook
chest_pain_label_encoder = LabelEncoder().fit(["Typical", "Atypical", "NonAnginal", "Asymptomatic"])
exercise_angina_label_encoder = LabelEncoder().fit(["Yes", "No"])
vessels_label_encoder = LabelEncoder().fit(["Zero", "One", "Two", "Three"])
thal_label_encoder = LabelEncoder().fit(["Normal", "Fixed", "Reversible"])

st.title("Heart Disease Prediction System")
st.write("Select the details of the patient and click Predict.")

# Take input from the user
chest_pain = st.selectbox("Chest Pain Type", ["Typical", "Atypical", "NonAnginal", "Asymptomatic"])
exercise_angina = st.selectbox("Chest Pain During Exercise", ["Yes", "No"])
vessels = st.selectbox("Number of Main Blood Vessels (X-ray test)", ["Zero", "One", "Two", "Three"])
thal = st.selectbox("Thallium Heart Scan Result", ["Normal", "Fixed", "Reversible"])

if st.button("Predict"):
    # Convert the input into a feature vector
    user_input = pd.DataFrame({
        "ChestPain": [chest_pain],
        "ExerciseAngina": [exercise_angina],
        "Vessels": [vessels],
        "Thal": [thal]
    })

    # Label encode the feature vector
    unseen_data_features = user_input.copy()
    unseen_data_features["ChestPain"] = chest_pain_label_encoder.transform(user_input["ChestPain"])
    unseen_data_features["ExerciseAngina"] = exercise_angina_label_encoder.transform(user_input["ExerciseAngina"])
    unseen_data_features["Vessels"] = vessels_label_encoder.transform(user_input["Vessels"])
    unseen_data_features["Thal"] = thal_label_encoder.transform(user_input["Thal"])

    # Apply the model and show the prediction
    prediction = model.predict(unseen_data_features)

    if prediction[0] == 1:
        st.error("Prediction: HEART DISEASE")
    else:
        st.success("Prediction: NO HEART DISEASE")

st.caption("Made for a class assignment. This is not medical advice.")
