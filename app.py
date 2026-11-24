import streamlit as st
import pandas as pd
import joblib


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Phishing Email Detector",
    page_icon="🛡️",
    layout="centered"
)


# --------------------------------------------------
# Load Model
# --------------------------------------------------

@st.cache_resource
def load_model():
    saved = joblib.load("phishing_email_model.pkl")
    return saved["model"], saved["features"]


model, features = load_model()


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🛡️ Phishing Email Detection")

st.write(
    "Enter the characteristics of an email to predict "
    "whether it is legitimate or phishing."
)

st.divider()


# --------------------------------------------------
# User Input
# --------------------------------------------------

st.subheader("Email Information")

has_link = st.selectbox(
    "Does the email contain a link?",
    ["No", "Yes"]
)

has_attachment = st.selectbox(
    "Does the email contain an attachment?",
    ["No", "Yes"]
)

urgency_score = st.slider(
    "Urgency Score",
    min_value=0,
    max_value=10,
    value=5
)

spelling_errors = st.number_input(
    "Number of Spelling Errors",
    min_value=0,
    max_value=100,
    value=0
)

email_length_words = st.number_input(
    "Email Length (words)",
    min_value=1,
    max_value=5000,
    value=100
)


# --------------------------------------------------
# Convert Inputs
# --------------------------------------------------

has_link_value = 1 if has_link == "Yes" else 0

has_attachment_value = 1 if has_attachment == "Yes" else 0


input_data = pd.DataFrame([{
    "has_link": has_link_value,
    "has_attachment": has_attachment_value,
    "urgency_score": urgency_score,
    "spelling_errors": spelling_errors,
    "email_length_words": email_length_words
}])


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("🔍 Detect Email", use_container_width=True):

    prediction = model.predict(input_data[features])[0]

    st.divider()

    if prediction == 1:

        st.error("🚨 PHISHING EMAIL")

        st.warning(
            "This email has characteristics associated "
            "with phishing emails. Be careful before "
            "clicking links or opening attachments."
        )

    else:

        st.success("✅ LEGITIMATE EMAIL")

        st.info(
            "The model predicts that this email is legitimate."
        )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Phishing Email Detection using Machine Learning"
)