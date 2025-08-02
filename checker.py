import streamlit as st
import spacy
import requests
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Google Custom Search API Key and Search Engine ID
GOOGLE_API_KEY = "#########YOUR_API_KEY##########"  # Replace with your valid API key
SEARCH_ENGINE_ID = "##########YOUR_ID#######"  # Replace with your search engine ID

# Load SpaCy model for text processing
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.error("The SpaCy model 'en_core_web_sm' is not available. Install it using: python -m spacy download en_core_web_sm")
    raise

# Search the web using Google Custom Search API
def search_web(query):
    if not GOOGLE_API_KEY or not SEARCH_ENGINE_ID:
        st.error("Google API Key and Search Engine ID are required to use the app.")
        return []

    try:
        url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={GOOGLE_API_KEY}&cx={SEARCH_ENGINE_ID}"
        response = requests.get(url)
        response.raise_for_status()
        search_results = json.loads(response.text)
        if "items" in search_results:
            return search_results["items"]
        else:
            st.info("No results found for the query.")
            return []
    except requests.exceptions.RequestException as e:
        st.error(f"Error accessing Google Custom Search API: {e}")
        return []

# Compute text embedding
def compute_embedding(text):
    doc = nlp(text)
    vectors = [token.vector for token in doc if token.has_vector and not token.is_stop]
    if vectors:
        return np.mean(vectors, axis=0)
    st.warning("No valid vectors computed from the text.")
    return np.zeros(nlp.vocab.vectors_length)

# Check similarity for plagiarism detection
def check_plagiarism(input_text, search_results, threshold):
    input_embedding = compute_embedding(input_text)
    if np.all(input_embedding == 0):
        return False, []

    plagiarized_snippets = []
    for result in search_results:
        snippet = result.get("snippet", "")
        snippet_embedding = compute_embedding(snippet)
        if np.all(snippet_embedding == 0):
            continue
        similarity = cosine_similarity([input_embedding], [snippet_embedding])[0][0]
        if similarity > threshold:
            plagiarized_snippets.append((result['title'], result['link'], similarity))

    return bool(plagiarized_snippets), plagiarized_snippets

# Main function
def main():
    # Add custom background and styling
    st.markdown(
        """
        <style>
            body {
                background-color: #f0f2f6;
            }
            .stApp {
                font-family: "Helvetica Neue", Arial, sans-serif;
            }
            .title {
                color: #4a90e2;
                font-weight: bold;
                text-align: center;
            }
            .result-box {
                background-color: #ffffff;
                border-radius: 8px;
                padding: 10px;
                margin-bottom: 10px;
                box-shadow: 0 0 5px rgba(0, 0, 0, 0.1);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<h1 class="title">AI-based Plagiarism Checker</h1>', unsafe_allow_html=True)
    st.markdown("### Check if your content is original or plagiarized using AI-powered tools.")

    # Text input
    text = st.text_area("Enter the text to check for plagiarism:", height=200, placeholder="Type or paste your text here...")

    # Fixed similarity threshold
    threshold = 0.7

    if st.button("Check for Plagiarism"):
        if not text.strip():
            st.warning("Please enter some text to check for plagiarism.")
        else:
            with st.spinner("Analyzing the text and searching for similar content online..."):
                search_results = search_web(text)
                if search_results:
                    st.subheader("Search Results:")
                    for result in search_results:
                        st.markdown(f"""
                        <div class="result-box">
                            <a href="{result['link']}" target="_blank"><b>{result['title']}</b></a>
                            <p>{result.get('snippet', '')}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    # Check for plagiarism
                    plagiarized, plagiarized_snippets = check_plagiarism(text, search_results, threshold)
                    if plagiarized:
                        st.warning("⚠ *Plagiarism Detected:* The text closely matches online sources.")
                        for title, link, similarity in plagiarized_snippets:
                            st.markdown(f"- *[{title}]({link})* (Similarity: {similarity:.2f})")
                    else:
                        st.success("✅ *No plagiarism detected:* The text appears to be original.")
                else:
                    st.info("No similar content found online.")

# Entry point
if __name__ == "__main__":
    main()