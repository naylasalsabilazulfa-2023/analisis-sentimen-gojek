import streamlit as st
import pickle
import re
import string
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk

nltk.download('stopwords')

# Load model
model = pickle.load(open('model_sentimen.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

# Preprocessing
stop_words = set(stopwords.words('indonesian'))

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def preprocess(text):

    text = text.lower()
    text = re.sub(r'http\\S+', '', text)
    text = re.sub(r'@\\w+', '', text)
    text = re.sub(r'#\\w+', '', text)
    text = re.sub(r'\\d+', '', text)
    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )

    text = ' '.join(
        [word for word in text.split()
         if word not in stop_words]
    )

    text = stemmer.stem(text)

    return text

# Tampilan aplikasi
st.title("Analisis Sentimen Ulasan Gojek")

st.write(
    "Masukkan ulasan aplikasi Gojek untuk mengetahui sentimennya."
)

review = st.text_area("Masukkan Ulasan")

if st.button("Prediksi Sentimen"):

    review_bersih = preprocess(review)

    data = tfidf.transform([review_bersih])

    hasil = model.predict(data)

    if hasil[0] == "positif":
        st.success("Sentimen Positif 😊")

    elif hasil[0] == "negatif":
        st.error("Sentimen Negatif 😞")

    else:
        st.warning("Sentimen Netral 😐")