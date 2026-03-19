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
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# ---------------- SIDEBAR ----------------
st.sidebar.title("📊 Spam Detection Project")
st.sidebar.info("Improved ML Model with Better Accuracy")

# ---------------- TITLE ----------------
st.title("📧 Spam Email Detection App")

# ---------------- DATASET ----------------
data = {
    'text': [
        # Spam
        "Win money now", "Free offer just for you",
        "Claim your prize", "Earn cash fast",
        "Limited time offer", "Congratulations you won",
        "Click here to claim reward", "Get rich quick",
        "Exclusive deal just for you", "You have won cash prize",

        # Non-Spam
        "Hello friend", "Meeting tomorrow",
        "Let's study together", "Project discussion",
        "Call me later", "Assignment submission is due today",
        "Please review the document", "Lunch at 2 PM",
        "See you soon", "Good morning",
        "Are you coming to class?", "Let's meet in library"
    ],
    'label': [1,1,1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0,0,0,0,0]
}

df = pd.DataFrame(data)

# Clean text
df['text'] = df['text'].apply(clean_text)

# ---------------- PREVIEW ----------------
st.subheader("📄 Dataset Preview")
st.write(df.head())

# ---------------- MODEL ----------------
X = df['text']
y = df['label']

vectorizer = TfidfVectorizer(stop_words='english')
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.3, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

# ---------------- EVALUATION ----------------
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

# Adjustable threshold
threshold = st.slider("Spam Detection Threshold", 0.0, 1.0, 0.7)

if st.button("Predict"):
    if user_input.strip() != "":
        cleaned_input = clean_text(user_input)
        input_data = vectorizer.transform([cleaned_input])

        prediction = model.predict(input_data)
        prob = model.predict_proba(input_data)

        spam_prob = prob[0][1]

        st.write("📈 Spam Probability:", round(spam_prob, 2))

        # Decision using threshold
        if spam_prob > threshold:
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
st.caption("Made with ❤️ using Streamlit")
