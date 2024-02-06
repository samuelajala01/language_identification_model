import pandas as pd
import numpy as np
from sklearn import model_selection
# from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
# from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
import streamlit as st

st.set_page_config(
    page_title="Langauge Identification Model",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

df = pd.read_csv("dataset.csv")

texts = df["Text"]
languages = df["Language"]

vectorizer = CountVectorizer()
encoder = LabelEncoder()

# Convert text data into numerical features
X = vectorizer.fit_transform(texts)
y = encoder.fit_transform(languages)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=4)

@st.cache_data()
def load_model():
    model = MultinomialNB()
    # model = SVC() perfoms very poor for this task
    model.fit(X_train, y_train)  # Assuming X_train and y_train are pre-defined
    return model

clf = load_model()

st.markdown("<h1 style='text-align: center; color: white;'>Language Identification Model</h1>", unsafe_allow_html=True)


st.write("This model can classify up to 17 different languages these are: English, Malayalam, Hindi, Tamil, Portugeese, French, Dutch, Spanish, Greek, Russian, Danish, Italian, Turkish, Sweedish, Arabic, German, Kannada.")

st.write("[![Star](https://img.shields.io/github/stars/samuelajala01/language_identification_model.svg?logo=github&style=social)](https://gitHub.com/samuelajala01/language_identification_model)")

st.write("[![Follow](https://img.shields.io/twitter/follow/samuelajala01?style=social)](https://www.twitter.com/samuelajala01)")

# New text
paragraph_input = st.text_area("Enter text/sentence to be identified: ")

# Make prediction when the user clicks the button
if st.button("Identify Language"):
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




# ... (rest of your code)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
# classification_rep = classification_report(y_test, y_pred, target_names=encoder.classes_)
conf_matrix = confusion_matrix(y_test, y_pred)

# Display the evaluation metrics
# st.write(f"Accuracy: {accuracy:.4f}")
# st.write("Confusion Matrix:")
# st.write(conf_matrix)


st.markdown("<h2 style='color:#82c9ff';>Dataset Statistics</h2>",unsafe_allow_html=True)

st.subheader("Class Distribution")
st.bar_chart(df['Language'].value_counts())

st.write(f"Number of Samples: {len(df)}")
st.write(df['Language'].value_counts())

