import streamlit as st
import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Spam Detector", page_icon="📧", layout="wide")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stTextArea textarea {
        border-radius: 10px;
    }
    .result-box {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- TEXT CLEANING ----------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# ---------------- TITLE ----------------
st.markdown("<h1 style='text-align:center; color:#4A90E2;'>📧 Spam Email Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Check if your message is spam or safe 🚀</p>", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("spam.csv", encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'text']
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    return df

df = load_data()
df['text'] = df['text'].apply(clean_text)

# ---------------- MODEL ----------------
X = df['text']
y = df['label']

vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2)

model = MultinomialNB(alpha=0.1)
model.fit(X_train, y_train)

# ---------------- LAYOUT ----------------
col1, col2 = st.columns(2)

# LEFT SIDE → INPUT
with col1:
    st.subheader("✉️ Enter Message")
    user_input = st.text_area("Type your message here...", height=150)

    threshold = st.slider("🎯 Detection Sensitivity", 0.0, 1.0, 0.5)

    predict_btn = st.button("🔍 Analyze Message")

# RIGHT SIDE → RESULT
with col2:
    st.subheader("📊 Result")

    if predict_btn:
        if user_input.strip() != "":
            cleaned = clean_text(user_input)
            vec = vectorizer.transform([cleaned])

            prob = model.predict_proba(vec)[0][1]

            st.metric("Spam Probability", f"{round(prob*100,2)} %")

            if prob >= threshold:
                st.markdown(
                    "<div class='result-box' style='background-color:#ff4d4d; color:white;'>🚨 SPAM DETECTED</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    "<div class='result-box' style='background-color:#4CAF50; color:white;'>✅ SAFE MESSAGE</div>",
                    unsafe_allow_html=True
                )
        else:
            st.warning("⚠️ Please enter a message")

# ---------------- MODEL PERFORMANCE ----------------
st.markdown("---")
st.subheader("📊 Model Performance")

y_pred = model.predict(X_test)

col3, col4 = st.columns(2)

with col3:
    st.metric("Accuracy", round(accuracy_score(y_test, y_pred)*100, 2))

with col4:
    st.write("Confusion Matrix")
    st.write(confusion_matrix(y_test, y_pred))

# ---------------- WORD CHART ----------------
st.markdown("---")
st.subheader("🔥 Most Common Words")

from collections import Counter
words = " ".join(df['text']).split()
common = Counter(words).most_common(10)

chart_df = pd.DataFrame(common, columns=["Word", "Count"]).set_index("Word")

st.bar_chart(chart_df)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("<center>✨ Made By Group 15(Pragya Srivastava and Ankit Kumar Jha) with Streamlit ✨</center>", unsafe_allow_html=True)
