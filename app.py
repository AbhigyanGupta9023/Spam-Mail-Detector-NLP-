import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Configure NLTK data path
nltk.data.path.append("C:/nltk_data")  # Create this folder if it doesn't exist

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('punkt', download_dir="C:/nltk_data")
    nltk.download('stopwords', download_dir="C:/nltk_data")
    nltk.download('averaged_perceptron_tagger', download_dir="C:/nltk_data")
    nltk.download('punkt_tab', download_dir="C:/nltk_data")

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y.copy()
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y.copy()
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load models
try:
    tfidf = pickle.load(open('vectorizer.pkl','rb'))
    model = pickle.load(open('model.pkl','rb'))
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.stop()

# Streamlit interface
st.title("Email and SMS Spam Classifier")
input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    try:
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]
        st.header("Spam 🚨" if result == 1 else "Not Spam ✅")
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")