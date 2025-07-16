import streamlit as st
import pickle
import cloudpickle

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))
with open('transform.pkl', 'rb') as f_in:
    transform_text = cloudpickle.load(f_in)

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
