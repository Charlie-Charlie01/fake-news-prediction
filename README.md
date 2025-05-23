# Fake News Prediction App
A Streamlit web application that uses machine learning to predict whether a news article is real or fake.

## Overview
This application loads a pre-trained Logistic Regression model to classify news articles as either real or fake. Users can input text directly or upload a text file for analysis.

## Features
- Text preprocessing with NLTK
- Real-time prediction using a trained Logistic Regression model
- Confidence score visualization
- Support for both direct text input and file upload
- Responsive UI design

## Setup Instructions
### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Installation
- Clone this repository or download the files
- Install the required dependencies:
  pip install -r requirements.txt
- Make sure your model files are in the same directory:
  fake_news_model.pkl - Your trained Logistic Regression model
  tfidf_vectorizer.pkl - The TF-IDF vectorizer used during training

### Running the App
Run the following command in your terminal:
  streamlit run app.py
The app should open automatically in your default web browser. If not, you can access it at http://localhost:8501

## Deployment
To deploy this Streamlit app to a production environment:
### Streamlit Cloud
- Push your code to a GitHub repository
- Sign up for Streamlit Cloud
- Connect your GitHub repository and deploy

## Model Training
The model was trained using Logistic Regression on a dataset of labeled news articles. The training process involved:
- Text preprocessing (removing stopwords, stemming)
- Feature extraction using TF-IDF vectorization
- Training a Logistic Regression classifier
- Saving the model using pickle

## Project Structure
- app.py: Main Streamlit application
- requirements.txt: List of Python dependencies
- fake_news_model.pkl: Saved Logistic Regression model (you need to provide this)
- tfidf_vectorizer.pkl: Saved TF-IDF vectorizer (you need to provide this)

## Customization
You can customize the application by:
- Modifying the app.py file to change UI elements
- Adding additional preprocessing steps
- Incorporating more features into the prediction algorithm

## License
This project is open source and available under the MIT License.
