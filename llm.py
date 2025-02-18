import torch
import pandas as pd
import re
import nltk
import numpy as np
import streamlit as st
from nltk.corpus import stopwords
from transformers import BertTokenizer, BertForSequenceClassification
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download stopwords if not available
nltk.download('stopwords')
STOPWORDS = set(stopwords.words("english"))

# Load dataset
file_path = "data.csv"
df = pd.read_csv(file_path)

# Load the fine-tuned BERT classification model
model_path = "fine_tuned_bert"
classification_model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Load BERT embedding model for similarity ranking
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Ensure models are in evaluation mode
classification_model.eval()

# Label mapping
label_map = {0: "Tactic", 1: "Technique"}

# Extract unique keywords from descriptions
def extract_keywords(text):
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())  
    words = [word for word in words if word not in STOPWORDS]  
    return list(set(words))  

df["Keywords"] = df["Description"].apply(extract_keywords)

# Compute TF-IDF similarity
def compute_tfidf_similarity(input_text, corpus):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([input_text] + corpus)
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

# Compute BERT similarity
def compute_bert_similarity(input_text, corpus):
    input_embedding = embedding_model.encode([input_text])
    corpus_embeddings = embedding_model.encode(corpus)
    return cosine_similarity(input_embedding, corpus_embeddings).flatten()

# Compute keyword match score
def keyword_match_score(input_text, keyword_lists):
    input_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", input_text.lower()))
    return np.array([len(input_words & set(keywords)) for keywords in keyword_lists])  

# Predict and fetch best match
def predict_and_fetch_best_match(text):
    inputs = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    
    with torch.no_grad():
        outputs = classification_model(**inputs)
    
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    predicted_label = label_map[predicted_class]

    filtered_df = df[df["Type"] == predicted_label]

    if filtered_df.empty:
        return predicted_label, "No matching records found."

    corpus = filtered_df.apply(lambda row: f"{row['Name']} {row['Description']}", axis=1).tolist()
    keyword_lists = filtered_df["Keywords"].tolist()

    tfidf_scores = compute_tfidf_similarity(text, corpus)
    bert_scores = compute_bert_similarity(text, corpus)
    keyword_scores = keyword_match_score(text, keyword_lists)

    combined_scores = 0.4 * tfidf_scores + 0.5 * bert_scores + 0.1 * keyword_scores

    best_index = combined_scores.argmax()
    best_match = filtered_df.iloc[best_index]

    return predicted_label, best_match.to_dict()

# 🎨 Streamlit UI
st.set_page_config(page_title="ATT&CK Threat Intelligence", layout="wide")
st.title("🔍 ATT&CK Threat Intelligence Predictor")

# User Input
user_input = st.text_area("📝 Enter a Threat Report:", "", height=150)

if st.button("🔍 Predict ATT&CK Category"):
    if user_input.strip():
        category, best_match = predict_and_fetch_best_match(user_input)

        st.subheader("🎯 Prediction Result")
        st.write(f"**Predicted ATT&CK Category:** `{category}`")

        if isinstance(best_match, dict):
            st.markdown(f"**🆔 ID:** `{best_match['ID']}`")
            st.markdown(f"**📌 Name:** `{best_match['Name']}`")
            st.markdown(f"**🔍 Description:** {best_match['Description']}")
        else:
            st.warning("❌ No strong match found.")
    else:
        st.warning("⚠️ Please enter a valid threat report.")

st.markdown("---")
st.info("Powered by **Fine-Tuned BERT, TF-IDF, and BERT Similarity Matching** for accurate ATT&CK predictions.")
