import streamlit as st
import numpy as np
import re
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ===============================
# Page Config
# ===============================

st.set_page_config(page_title="Sentiment Analysis App", layout="centered")

st.title("🧠 Twitter Sentiment Analysis")
st.write("Bidirectional GRU Model with GloVe Embeddings")

# ===============================
# Load Saved Files
# ===============================

@st.cache_resource
def load_all():
    model = load_model("model.h5")

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    return model, tokenizer, label_encoder


model, tokenizer, label_encoder = load_all()

# ===============================
# Text Cleaning Function
# ===============================

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    return text

# ===============================
# Prediction Function
# ===============================

max_len = 50

def predict_sentiment(text):
    text = clean_text(text)
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')

    prediction = model.predict(padded)
    predicted_class = np.argmax(prediction, axis=1)

    label = label_encoder.inverse_transform(predicted_class)[0]
    confidence = np.max(prediction)

    return label, confidence

# ===============================
# UI Section
# ===============================

user_input = st.text_area("Enter a Tweet:", height=150)

if st.button("Predict Sentiment"):

    if user_input.strip() == "":
        st.warning("⚠ Please enter some text.")
    else:
        label, confidence = predict_sentiment(user_input)

        st.success(f"Predicted Sentiment: **{label}**")
        st.info(f"Confidence Score: {confidence:.2f}")

        if label.lower() == "positive":
            st.markdown("😊 This looks like a Positive tweet!")
        elif label.lower() == "negative":
            st.markdown("😡 This looks like a Negative tweet!")
        else:
            st.markdown("😐 This looks like a Neutral tweet!")

st.markdown("---")
st.caption("Built using Bidirectional GRU + GloVe Embeddings")
