import streamlit as sl
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

model2 = PorterStemmer()
def message_transform(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    req_words = []
    for i in text:
        if i.isalnum() == True:
            req_words.append(i)
    req_words1 = []
    for i in req_words:
        if i not in stopwords.words("english") and i not in string.punctuation:
            req_words1.append(i)
    req_words2 = []
    for i in req_words1:
        req_words2.append(model2.stem(i))

    return " ".join(req_words2)
tfidf = pickle.load(open("email_spam_classifier\TfidfVectorizer.pkl", "rb"))
mbn = pickle.load(open("email_spam_classifier\MultinomialNB.pkl", "rb"))

sl.title("Email/SMS Spam Classifier by Abhinav")
input_msg = sl.text_input("Enter the message to check: ")

if sl.button("Predict"):
    message_update = message_transform(input_msg)
    model3 = tfidf.transform([message_update])
    output = mbn.predict(model3)[0]

    if output == 1:
        sl.header("Spam Message")
    else:
        sl.header("Not a Spam Message")
    