import streamlit as st
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from collections import Counter
import math

# Download required NLTK data
@st.cache(allow_output_mutation=True)
def download_nltk_data():
    try:
        nltk.data.find('punkt')
        nltk.data.find('stopwords')
        nltk.data.find('wordnet')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
    return True

# Initialize NLTK
download_nltk_data()

# Cache the stopwords and lemmatizer
@st.cache(allow_output_mutation=True)
def get_text_processors():
    return {
        'stop_words': set(stopwords.words('english')),
        'lemmatizer': WordNetLemmatizer()
    }

text_processors = get_text_processors()

def preprocess_text(text):
    """
    Preprocess the input text by cleaning, tokenizing, removing stopwords and lemmatizing
    """
    # Convert to lowercase and remove special characters
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    tokens = [token for token in tokens if token not in text_processors['stop_words']]
    
    # Lemmatize
    tokens = [text_processors['lemmatizer'].lemmatize(token) for token in tokens]
    
    return tokens

def get_word_freq(tokens):
    """
    Get word frequencies from tokens
    """
    return Counter(tokens)

def calculate_cosine_similarity(freq1, freq2):
    """
    Calculate cosine similarity between two frequency dictionaries
    """
    # Get all unique words
    all_words = set(freq1.keys()) | set(freq2.keys())
    
    # Calculate dot product and magnitudes
    dot_product = sum(freq1.get(word, 0) * freq2.get(word, 0) for word in all_words)
    magnitude1 = math.sqrt(sum(freq1.get(word, 0) ** 2 for word in all_words))
    magnitude2 = math.sqrt(sum(freq2.get(word, 0) ** 2 for word in all_words))
    
    # Avoid division by zero
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
        
    return dot_product / (magnitude1 * magnitude2)

def calculate_similarity_metrics(text1, text2):
    """
    Calculate various similarity metrics between two texts
    """
    # Preprocess texts
    tokens1 = preprocess_text(text1)
    tokens2 = preprocess_text(text2)
    
    # Get word frequencies
    freq1 = get_word_freq(tokens1)
    freq2 = get_word_freq(tokens2)
    
    # Calculate cosine similarity
    similarity = calculate_cosine_similarity(freq1, freq2)
    
    # Calculate additional metrics
    words1 = set(tokens1)
    words2 = set(tokens2)
    
    common_words = len(words1.intersection(words2))
    unique_words1 = len(words1)
    unique_words2 = len(words2)
    
    return {
        'cosine_similarity': similarity,
        'common_words': common_words,
        'unique_words_text1': unique_words1,
        'unique_words_text2': unique_words2
    }

st.title("Text Similarity Analyzer")
st.write("Compare two texts and analyze their similarity for SEO purposes.")

# Input text areas
col1, col2 = st.columns(2)
with col1:
    text1 = st.text_area("Enter first text:", height=200)
with col2:
    text2 = st.text_area("Enter second text:", height=200)

if st.button("Analyze Similarity") and text1 and text2:
    with st.spinner("Analyzing texts..."):
        # Calculate metrics
        metrics = calculate_similarity_metrics(text1, text2)
        
        # Display results
        st.subheader("Analysis Results")
        
        # Create a DataFrame for better presentation
        results_df = pd.DataFrame({
            'Metric': ['Similarity Score', 'Common Words', 'Unique Words (Text 1)', 'Unique Words (Text 2)'],
            'Value': [
                f"{metrics['cosine_similarity']:.2%}",
                metrics['common_words'],
                metrics['unique_words_text1'],
                metrics['unique_words_text2']
            ]
        })
        
        st.table(results_df)
        
        # Interpretation
        st.subheader("Interpretation")
        similarity_score = metrics['cosine_similarity']
        if similarity_score >= 0.8:
            st.warning("High similarity detected! These texts might be considered duplicate content.")
        elif similarity_score >= 0.5:
            st.info("Moderate similarity detected. Some content overlap exists.")
        else:
            st.success("Low similarity. These texts are substantially different.")
