import streamlit as st
import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------- TEXT CLEANING ----------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)   # remove URLs
    text = re.sub(r'\d+', '', text)       # remove numbers
    text = re.sub(r'[^a-z\s]', '', text)  # remove special chars
    return text

# ---------------- SIDEBAR ----------------
st.sidebar.title("📊 Spam Detection Project")
st.sidebar.info("ML Model trained on real dataset")

# ---------------- TITLE ----------------
st.title("📧 Spam Email Detection App")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("spam.csv", encoding='latin-1')

    # Keep only required columns
    df = df[['v1', 'v2']]
    df.columns = ['label', 'text']

    # Convert labels
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    return df

df = load_data()

# Clean text
df['text'] = df['text'].apply(clean_text)

# ---------------- DATA PREVIEW ----------------
st.subheader("📄 Dataset Preview")
st.write(df.head())

# ---------------- MODEL TRAINING ----------------
X = df['text']
y = df['label']

vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1,2),
    max_df=0.9,
    min_df=2
)

X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

model = MultinomialNB(alpha=0.1)
model.fit(X_train, y_train)

# ---------------- MODEL EVALUATION ----------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

st.subheader("📊 Model Performance")
st.write("Accuracy:", round(accuracy, 2))
st.write("Confusion Matrix:")
st.write(cm)

# ---------------- USER INPUT ----------------
st.subheader("✉️ Test Your Own Email")

user_input = st.text_area("Enter Email Text")

# Threshold slider
threshold = st.slider("Spam Detection Threshold", 0.0, 1.0, 0.5)

# ---------------- PREDICTION ----------------
if st.button("Predict"):
    if user_input.strip() != "":
        cleaned_input = clean_text(user_input)
        input_data = vectorizer.transform([cleaned_input])

        prob = model.predict_proba(input_data)[0][1]

        st.write("📈 Spam Probability:", round(prob, 2))

        if prob >= threshold:
            st.error("🚨 This is SPAM!")
        else:
            st.success("✅ This is NOT SPAM")
    else:
        st.warning("⚠️ Please enter some text.")

# ---------------- WORD FREQUENCY ----------------
st.subheader("📊 Top Words in Dataset")

from collections import Counter

all_words = " ".join(df['text']).split()
word_counts = Counter(all_words)

common_words = dict(word_counts.most_common(10))

st.bar_chart(pd.DataFrame(common_words.values(), index=common_words.keys()))

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Made By Pragya Srivastava with ❤️ using Streamlit")
