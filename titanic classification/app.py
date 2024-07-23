import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))



import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')

# TRANSFORM FUNCTION FOR PREPROCESSING
ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
   
    #removing stopwords
    text=[word for word in text if word.isalnum()]

    text=[word for word in text if word not in stopwords.words('english') and word not in string.punctuation]

    text=[ps.stem(word) for word in text]

 


    return " ".join(text)




st.title("EMAIL Spam Classifier")
input_sms = st.text_area("Enter TEXT here")

if st.button('Predict'):
    # 1. PREPROCESS TEXT
    transformed_sms = transform_text(input_sms)

    # 2. VECTORIZE USING TFIDF
    vector_input = tfidf.transform([transformed_sms])

    # 3. PREDICT USING MULTINOMIAL NAIVE BAYES MODEL
    result = model.predict(vector_input)

    # 4. DISPLAY THE OUTPUT AS HAM-SPAM TEXT
    if result == 1:
        st.header("SPAM TEXT")
    else:
        st.header("NOT A SPAM TEXT - HAM TEXT")
