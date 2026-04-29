import streamlit as st
import pickle
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
nltk.download('punkt')
nltk.download('stopwords')

# -------------------------
# LOAD MODEL (ONLY THIS)
# -------------------------
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


st.title("📩 SMS Spam Classifier with Probability")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])

    prob = model.predict_proba(vector_input)[0]

    spam_prob = prob[1] * 100
    ham_prob = prob[0] * 100

    if prob[1] > 0.5:
        st.error("🚨 Spam Message")
    else:
        st.success("✅ Not Spam")

    st.write(f"📊 Spam Probability: {spam_prob:.2f}%")
    st.write(f"📊 Not Spam Probability: {ham_prob:.2f}%")
