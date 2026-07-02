import streamlit as st
import numpy as np
import pandas as pd
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
TEXT_COLUMN_CANDIDATES = ("text", "tweet", "content", "message", "body")


def find_text_column(dataframe):
    normalized_columns = {column.strip().lower(): column for column in dataframe.columns}
    for candidate in TEXT_COLUMN_CANDIDATES:
        if candidate in normalized_columns:
            return normalized_columns[candidate]
    return None


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
st.subheader("Batch Predict From CSV")
uploaded_file = st.file_uploader(
    "Upload a CSV with a text, tweet, content, message, or body column",
    type=["csv"],
)

if uploaded_file is not None:
    try:
        uploaded_df = pd.read_csv(uploaded_file)
    except pd.errors.EmptyDataError:
        st.warning("The uploaded CSV is empty.")
    except pd.errors.ParserError:
        st.warning("The uploaded CSV could not be parsed.")
    else:
        text_column = find_text_column(uploaded_df)
        if text_column is None:
            st.warning("Add a text, tweet, content, message, or body column.")
        else:
            rows = uploaded_df.copy()
            rows[text_column] = rows[text_column].fillna("").astype(str)
            valid_rows = rows[rows[text_column].str.strip() != ""].copy()

            if valid_rows.empty:
                st.warning("No non-empty text rows found in the CSV.")
            else:
                labels = []
                confidences = []
                for text in valid_rows[text_column]:
                    label, confidence = predict_sentiment(text)
                    labels.append(label)
                    confidences.append(round(float(confidence), 4))

                valid_rows["predicted_sentiment"] = labels
                valid_rows["confidence"] = confidences
                st.dataframe(valid_rows, use_container_width=True)
                st.download_button(
                    "Download Predictions",
                    valid_rows.to_csv(index=False).encode("utf-8"),
                    file_name="sentiment_predictions.csv",
                    mime="text/csv",
                )

st.markdown("---")
st.caption("Built using Bidirectional GRU + GloVe Embeddings")
