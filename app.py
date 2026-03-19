import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

from collections import Counter

# ---------------- Sidebar ----------------
st.sidebar.title("📊 Spam Detection Project")
st.sidebar.info("Machine Learning + Statistics using Streamlit")

# ---------------- Title ----------------
st.title("📧 Spam Email Detection App")

# ---------------- File Upload ----------------
uploaded_file = st.file_uploader("Upload CSV Dataset (columns: text, label)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    # Default dataset
    data = {
        'text': [
            "Win money now", "Hello friend",
            "Free offer just for you", "Meeting tomorrow",
            "Claim your prize", "Let's study together",
            "Earn cash fast", "Project discussion",
            "Limited time offer", "Call me later"
        ],
        'label': [1,0,1,0,1,0,1,0,1,0]
    }
    df = pd.DataFrame(data)

# ---------------- Dataset Preview ----------------
st.subheader("📄 Dataset Preview")
st.write(df.head())

# ---------------- Train Model ----------------
X = df['text']
y = df['label']

vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.3, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

# ---------------- Evaluation ----------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

st.subheader("📊 Model Performance")
st.write("Accuracy:", round(accuracy, 2))

st.write("Confusion Matrix:")
st.write(cm)

# ---------------- Prediction Section ----------------
st.subheader("✉️ Test Your Own Email")

user_input = st.text_area("Enter Email Text")

if st.button("Predict"):
    if user_input.strip() != "":
        input_data = vectorizer.transform([user_input])
        prediction = model.predict(input_data)
        prob = model.predict_proba(input_data)

        st.write("📈 Spam Probability:", round(prob[0][1], 2))

        if prediction[0] == 1:
            st.error("🚨 This is SPAM!")
        else:
            st.success("✅ This is NOT SPAM")
    else:
        st.warning("⚠️ Please enter some text.")

# ---------------- Word Frequency ----------------
st.subheader("📊 Top 10 Words in Dataset")

all_words = " ".join(df['text']).lower().split()
word_counts = Counter(all_words)

common_words = dict(word_counts.most_common(10))

st.bar_chart(pd.DataFrame(common_words.values(), index=common_words.keys()))

# ---------------- Footer ----------------
st.markdown("---")
st.caption("Made with ❤️ using Streamlit")
