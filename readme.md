# Language Identification Model using Streamlit

![Language Identification img](https://your_image_url_here)
<br/>
[View Project](https://lang-identification-model.streamlit.app)

## Overview
This Streamlit app is designed to demonstrate a language identification model. Given an input text, the model predicts the language of the text from a set of supported languages. This app is built using Streamlit, a popular Python library for building web applications.

## Features
- Identify the language of input text
- Supports up to 17 languages
- Simple and intuitive user interface

## Demo
Gif here demonstrating the usage of the app.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/samuelajala01/language-identification-model.git
    cd language-identification-model
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```
2. Open your web browser and navigate to the provided URL (typically http://localhost:8501).
3. Enter a text in the provided input box.
4. Click on the "Identify Language" button to see the predicted language.

## Supported Languages
English, Malayalam, Hindi, Tamil, Portugeese, French, Dutch, Spanish, Greek, Russian, Danish, Italian, Turkish, Sweedish, Arabic, German, Kannada.

## Model Details
- Algorithm: MultinomialNB
- Vectorizer: CountVectorizer()
- test/train ratio: 0.2
- Accuracy metric: Confusion Matrix, Accuracy Score

## Contributing
Contributions are welcome!
