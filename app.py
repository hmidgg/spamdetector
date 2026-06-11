import pandas as pd
import urllib.request
import zipfile
import streamlit as str
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# 1. Page Configuration
str.set_page_config(page_title="Mail Spam Detector", page_icon="🚨", layout="centered")
str.title("📩 Automated Email Spam Classifier")
str.subheader("Client-Ready AI Solution")

# 2. Cache Model Training (لـ سرعة الـ واجهة)
def load_and_train_model():
    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    urllib.request.urlretrieve(url, "sms.zip")
    with zipfile.ZipFile("sms.zip", "r") as zip_ref:
        zip_ref.extractall(".")
    
    df = pd.read_csv("SMSSpamCollection", sep="\t", names=["label", "message"])
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    
    vectorizer = TfidfVectorizer(stop_words="english")
    X_tfidf = vectorizer.fit_transform(df["message"])
    
    model = MultinomialNB()
    model.fit(X_tfidf, df["label"])
    return vectorizer, model

vectorizer, model = load_and_train_model()

# 3. User Interface
user_input = str.text_area("Paste the email or SMS content here:", height=150, placeholder="Type or paste text...")

if str.button("Analyze Message", type="primary"):
    if user_input.strip() != "":
        # Predict
        input_tfidf = vectorizer.transform([user_input])
        prediction = model.predict(input_tfidf)[0]
        confidence = model.predict_proba(input_tfidf)[0][prediction] * 100
        
        # Display Result
        str.write("---")
        if prediction == 1:
            str.error(f"🚨 **Result: SPAM Detected!**")
            str.warning(f"Confidence Level: {confidence:.2f}%")
        else:
            str.success(f"✅ **Result: HAM (Clean Message)**")
            str.info(f"Confidence Level: {confidence:.2f}%")
    else:
        str.info("Please enter some text to analyze.")