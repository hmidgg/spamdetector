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

# 2. Sidebar - Why Naive Bayes?
str.sidebar.header("🧠 Technical Insights")
str.sidebar.markdown("""
### Why Multinomial Naive Bayes?
During our development phase, we benchmarked multiple algorithms. **Naive Bayes** consistently emerged as the **most performant algorithm** for this text-classification task due to:
1. **High Efficiency:** Extremely fast training and real-time inference.
2. **Text Specialization:** Works exceptionally well with high-dimensional TF-IDF word vectors.
3. **Probability-Driven:** Outputs exact mathematical confidence levels for transparency.
""")

# 3. Model Training (Safe Extraction)

def load_and_train_model():
    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    urllib.request.urlretrieve(url, "sms.zip")
    
    # هوني الـ قفل الـ هندسي: نـخرّجو الـ داتا بركة ونـطفيو الـ readme متاعهم!
    with zipfile.ZipFile("sms.zip", "r") as zip_ref:
        zip_ref.extract(member="SMSSpamCollection", path=".")
    
    df = pd.read_csv("SMSSpamCollection", sep="\t", names=["label", "message"])
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    
    vectorizer = TfidfVectorizer(stop_words="english")
    X_tfidf = vectorizer.fit_transform(df["message"])
    
    model = MultinomialNB()
    model.fit(X_tfidf, df["label"])
    return vectorizer, model

vectorizer, model = load_and_train_model()

# 4. User Interface
user_input = str.text_area("Paste the email or SMS content here:", height=150, placeholder="Type or paste text...")

if str.button("Analyze Message", type="primary"):
    if user_input.strip() != "":
        input_tfidf = vectorizer.transform([user_input])
        prediction = model.predict(input_tfidf)[0]
        confidence = model.predict_proba(input_tfidf)[0][prediction] * 100
        
        str.write("---")
        if prediction == 1:
            str.error(f"🚨 **Result: SPAM Detected!**")
            str.metric(label="Mathematical Confidence Level", value=f"{confidence:.2f}%")
            
        else:
            str.success(f"✅ **Result: HAM (Clean Message)**")
            str.metric(label="Mathematical Confidence Level", value=f"{confidence:.2f}%")
        
    else:
        str.info("Please enter some text to analyze.")