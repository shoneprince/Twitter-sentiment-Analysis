# 🧠 Twitter Sentiment Analysis

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-App-red?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/NLP-Sentiment-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge"/>
</p>

<p align="center">
  An end-to-end NLP project that classifies tweets into <strong>Positive 😊</strong>, <strong>Negative 😡</strong>, or <strong>Neutral 😐</strong> sentiments using a Bidirectional GRU deep learning model — deployed with an interactive Streamlit web application.
</p>

---

## 📌 Table of Contents

- [Web Application](#-web-application)
- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Tech Stack](#-tech-stack)

---

## 🚀 Live Web Application

The predictive model is deployed as an interactive web application using Streamlit.
Try it live here: https://twitter-sentiment-analysis-app-duib.onrender.com/

## 🔍 Overview

This project builds a complete machine learning pipeline — from raw tweet data to a live web application — for real-time Twitter sentiment classification. The model is trained on labeled tweet data and learns to detect the emotional tone of any text input.

```
Tweet Input  ──►  Text Cleaning  ──►  Tokenization  ──►  BiGRU Model  ──►  Sentiment Label
```

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧹 **Text Cleaning** | Strips URLs, mentions, hashtags, punctuation, and numbers |
| 🔢 **Tokenization** | Keras Tokenizer with vocabulary of 10,000 words + OOV token |
| 📐 **Padding** | Sequences padded/truncated to fixed length of 50 tokens |
| 🤖 **Bidirectional GRU** | Captures both forward and backward context in sequences |
| 🎯 **3-Class Output** | Predicts Positive, Negative, or Neutral with a confidence score |
| ⚡ **Real-time Prediction** | Instant results via cached model loading in Streamlit |
| 📊 **Confidence Score** | Shows probability score alongside the predicted label |
| 🌐 **Web Deployment** | Fully deployed as a Streamlit web app on Render |

---

## 📁 Project Structure

```
Twitter-sentiment-Analysis/
│
├── Twitter_data_main.py       # Model training pipeline
├── Twitter_data_app.py        # Streamlit web application
├── End to end NLP Project.py  # Complete notebook-style pipeline
│
├── model.h5                   # Saved trained Keras model
├── tokenizer.pkl              # Saved tokenizer (fitted on training data)
├── label_encoder.pkl          # Saved label encoder
│
├── requirements.txt           # Python dependencies
├── .python-version            # Python 3.11 pin (for Render)
└── README.md
```


**App Features:**
- 📝 Text area input for any tweet or sentence
- 🔮 One-click **Predict Sentiment** button
- 🏷️ Displays predicted class: **Positive / Negative / Neutral**
- 📈 Shows confidence score (0.00 – 1.00)
- 😊😡😐 Emoji feedback for each prediction
- ⚡ `@st.cache_resource` for fast model loading (loaded once, reused)

---

## 🛠️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/shoneprince/Twitter-sentiment-Analysis.git
cd Twitter-sentiment-Analysis
```

### 2. Set up Python environment
```bash
# Python 3.11
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the web app
```bash
streamlit run Twitter_data_app.py
```

---

### Python Version
Pin Python to **3.11** by adding a `.python-version` file to the repo root:
```
3.11.9
```
> ⚠️ TensorFlow does not support Python 3.12+ yet. Without this file, Render defaults to the latest Python version and the build will fail.

### Required `requirements.txt`
```
pandas
numpy
scikit-learn
tensorflow
streamlit
```

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.11 |
| **Deep Learning** | TensorFlow / Keras |
| **Model Type** | Bidirectional GRU |
| **Text Processing** | Keras Tokenizer, Regex |
| **Encoding** | Scikit-learn LabelEncoder |
| **Web Framework** | Streamlit |
| **Serialization** | Pickle (tokenizer, label encoder), HDF5 (model) |
| **Deployment** | Render |

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<p align="center">Built with ❤️ using TensorFlow & Streamlit</p>
