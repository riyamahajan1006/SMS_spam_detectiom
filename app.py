import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

def transform_text(text):
    text = text.lower()
    words = nltk.word_tokenize(text)
    words = [w for w in words if w.isalnum()]
    words = [w for w in words if w not in stop_words]
    words = [ps.stem(w) for w in words]
    return " ".join(words)

st.title("📩 SMS Spam Classifier")
st.subheader("Enter a message below to check if it's Spam or Not:")

input_sms = st.text_area("Your Message:")

if st.button("Classify"):
    if input_sms:
        transformed_text = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_text])
        result = model.predict(vector_input)[0]
        if result == 1:
            st.header("🚫 Spam detected!")
        else:
            st.header("✅ This message is safe (Ham).")
    else:
        st.warning("Please enter a message to classify.")
