import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample dataset
emails = [
    "Congratulations! You won a lottery",
    "Claim your free prize now",
    "Meeting at 10 AM tomorrow",
    "Project submission deadline",
    "Win money instantly",
    "Hello, how are you?",
    "Limited time offer",
    "Let's have lunch today",
    "Free recharge available",
    "Happy Birthday!"
]

labels = [
    "Spam",
    "Spam",
    "Ham",
    "Ham",
    "Spam",
    "Ham",
    "Spam",
    "Ham",
    "Spam",
    "Ham"
]

# Convert into DataFrame
df = pd.DataFrame({
    "Email": emails,
    "Label": labels
})

# Convert text into numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["Email"])

# Train model
model = MultinomialNB()
model.fit(X, df["Label"])

# Streamlit UI
st.title("📧 Spam Email Classifier")

st.write("Enter an email message to check whether it is Spam or Ham.")

user_email = st.text_area("Email")

if st.button("Predict"):

    if user_email.strip() == "":
        st.warning("Please enter an email.")
    else:
        test = vectorizer.transform([user_email])
        prediction = model.predict(test)[0]

        if prediction == "Spam":
            st.error("🚨 This email is SPAM!")
        else:
            st.success("✅ This email is NOT SPAM.")