
import streamlit as st
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

st.title("🎬 Movie Sentiment Analyzer")

text = st.text_area("Enter your movie review:")

if st.button("Analyze Sentiment"):

    if text.strip():

        result = classifier(text)[0]

        st.subheader("Prediction")
        st.write(result["label"])

        st.subheader("Confidence")
        st.write(f"{result['score']:.2%}")

    else:
        st.warning("Please enter some text.")
