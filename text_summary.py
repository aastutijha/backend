import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
from bs4 import BeautifulSoup
import re
import math

def get_text_from_link(link):
    try:
        response = requests.get(link)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        text_content = soup.get_text(separator=' ', strip=True)
        return text_content
    except requests.RequestException as e:
        return f"Error fetching content from link: {e}"

def textrank_summarizer(rawdocs, percentage=30):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(rawdocs)
    sentences = [sent.text for sent in doc.sents]  # Use spaCy's sentence segmentation

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sentences)
    similarity_matrix = (X * X.T).toarray()

    G = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(G)

    ranked_sentences = sorted(((scores[i], sent) for i, sent in enumerate(sentences)), reverse=True)
    select_len = int(len(ranked_sentences) * (percentage / 100))
    selected_sentences = [sent for _, sent in ranked_sentences[:select_len]]

    # Return the sentences in their original order
    summary = ' '.join(sorted(selected_sentences, key=sentences.index))

    return {
        "summary": summary,
        "original_txt": doc.text,
        "len_orig_txt": len(rawdocs.split(' ')),
        "len_summary": len(summary.split(' '))
    }
