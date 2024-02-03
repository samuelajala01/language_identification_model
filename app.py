import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import datasets, linear_model, model_selection
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import streamlit as st


st.title("Language Identification Model")

st.write("This model can classify up to 17 different languages these are: English, Malayalam, Hindi, Tamil, Portugeese, French, Dutch, Spanish, Greek, Russian, Danish, Italian, Turkish, Sweedish, Arabic, German, Kannada.")

df = pd.read_csv("dataset.csv")

texts = df["Text"]
languages = df["Language"]

vectorizer = CountVectorizer()
encoder = LabelEncoder()

# Convert text data into numerical features
X = vectorizer.fit_transform(texts)
y = encoder.fit_transform(languages)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3)

clf = SVC()  # Using Support Vector Classifier as an example
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# New text
paragraph_input = st.text_area("Enter text to be identified: ")

# Make prediction when the user clicks the button
if st.button("Predict Language"):
    if paragraph_input:
        # Apply the same vectorizer and label encoder transformations to new data
        X_new = vectorizer.transform([paragraph_input])  # Assuming vectorizer is already defined

        # Predict the language of the new text data
        predicted_labels = clf.predict(X_new)  # Assuming clf is already defined

        # Convert predicted labels back to original language names using the encoder
        predicted_language = encoder.inverse_transform(predicted_labels)  # Assuming encoder is already defined

        # Display the predicted languages for the text
        st.write(f"Predicted Language: {predicted_language[0]}")
    else:
        st.warning("Please enter text before predicting.")
