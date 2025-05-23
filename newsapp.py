import streamlit as st
import pickle
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

# Set page config
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide"
)

# Download NLTK resources
@st.cache_resource
def download_nltk_resources():
    nltk.download('stopwords')

download_nltk_resources()

# Load the saved model and vectorizer
@st.cache_resource
def load_model():
    try:
        model = pickle.load(open('fake_news_model.pkl', 'rb'))
        vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
        return model, vectorizer
    except FileNotFoundError:
        st.error("Model files not found. Please ensure fake_news_model.pkl and tfidf_vectorizer.pkl are in the same directory as this app.")
        return None, None

model, vectorizer = load_model()

# Text preprocessing function
def preprocess_text(text):
    # Convert to lowercase and remove non-alphanumeric characters
    text = re.sub('[^a-zA-Z]', ' ', text.lower())
    
    # Tokenize
    text = text.split()
    
    # Remove stopwords and stemming
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    text = [ps.stem(word) for word in text if word not in stop_words]
    
    # Join back to string
    return ' '.join(text)

# Prediction function
def predict_news(input_text, model, vectorizer):
    # Preprocess the input text
    processed_text = preprocess_text(input_text)
    
    # Transform text using the loaded vectorizer
    text_vector = vectorizer.transform([processed_text])
    
    # Make prediction
    prediction = model.predict(text_vector)[0]
    probability = model.predict_proba(text_vector)[0]
    
    return prediction, probability

# Main app
def main():
    st.title("📰 Fake News Detector")
    st.write("This app uses machine learning to predict whether news is real or fake.")
    
    # Check if model and vectorizer loaded successfully
    if model is None or vectorizer is None:
        st.warning("Please place your model files in the app directory and restart.")
        return
    
    # Input options
    input_option = st.radio("Choose input method:", ["Enter Text", "Upload File"])
    
    if input_option == "Enter Text":
        news_text = st.text_area("Enter the news article text:", height=250)
        
        if st.button("Predict"):
            if news_text.strip() == "":
                st.warning("Please enter some text to analyze.")
            else:
                with st.spinner("Analyzing..."):
                    prediction, probability = predict_news(news_text, model, vectorizer)
                    display_prediction(prediction, probability)
    
    else:  # Upload File
        uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
        
        if uploaded_file is not None:
            news_text = uploaded_file.read().decode("utf-8")
            st.text_area("File Content:", news_text, height=250, disabled=True)
            
            if st.button("Predict"):
                with st.spinner("Analyzing..."):
                    prediction, probability = predict_news(news_text, model, vectorizer)
                    display_prediction(prediction, probability)

# Function to display prediction results
def display_prediction(prediction, probability):
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if prediction == 0:  # Assuming 0 = Real, 1 = Fake
            st.success("Prediction: REAL NEWS")
            probability_value = probability[0]
        else:
            st.error("Prediction: FAKE NEWS")
            probability_value = probability[1]
        
        st.metric("Confidence", f"{probability_value*100:.2f}%")
    
    with col2:
        # Visualization of confidence
        st.write("Confidence Distribution:")
        chart_data = pd.DataFrame({
            'Category': ['Real News', 'Fake News'],
            'Probability': [probability[0], probability[1]]
        })
        st.bar_chart(chart_data.set_index('Category'))
        
        # Additional context
        st.info("""
        - Real News: Factual, verifiable content from reputable sources
        - Fake News: Misleading, false information designed to deceive
        """)

# Run the app
if __name__ == "__main__":
    main()
