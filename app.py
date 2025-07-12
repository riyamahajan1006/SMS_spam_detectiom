import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
ps= PorterStemmer()

tfidf=pickle.load(open('vectorizer.pkl', 'rb'))
model=pickle.load(open('model.pkl', 'rb'))

def transform_text(text):
    text = text.lower()
    words = nltk.word_tokenize(text)
    words = [w for w in words if w.isalnum()]
    words = [w for w in words if w not in stopwords.words('english')]
    words = [ps.stem(w) for w in words]
    return " ".join(words)


st.title("SMS Spam Classifier")

input_sms= st.text_input("Enter the SMS Message")

#preprocess
transformed_text=transform_text(input_sms)
#vectorise
vector_input=tfidf.transform([transformed_text])
#predict
result=model.predict(vector_input)[0]
#display
if(result==1):
    st.header("Spam")

else:
    st.header("Ham")
