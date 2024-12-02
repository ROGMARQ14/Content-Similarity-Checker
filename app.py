import streamlit as st
import nltk
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')

download_nltk_data()

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
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)

def calculate_similarity_metrics(text1, text2):
    """
    Calculate various similarity metrics between two texts
    """
    # Preprocess texts
    processed_text1 = preprocess_text(text1)
    processed_text2 = preprocess_text(text2)
    
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([processed_text1, processed_text2])
    
    # Calculate cosine similarity
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    
    # Calculate additional metrics
    words1 = set(processed_text1.split())
    words2 = set(processed_text2.split())
    
    common_words = len(words1.intersection(words2))
    unique_words1 = len(words1)
    unique_words2 = len(words2)
    
    return {
        'cosine_similarity': similarity,
        'common_words': common_words,
        'unique_words_text1': unique_words1,
        'unique_words_text2': unique_words2
    }

def plot_similarity_metrics(metrics):
    """
    Create a visualization of similarity metrics
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create a heatmap-style visualization
    similarity_score = metrics['cosine_similarity']
    sns.heatmap([[similarity_score]], 
                annot=True, 
                cmap='RdYlGn', 
                vmin=0, 
                vmax=1, 
                center=0.5,
                cbar_kws={'label': 'Similarity Score'},
                ax=ax)
    
    ax.set_title('Text Similarity Visualization')
    ax.set_xticks([])
    ax.set_yticks([])
    
    return fig

def main():
    st.title("Text Similarity Analyzer")
    st.write("This tool analyzes the similarity between two texts and provides detailed metrics.")
    
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
            
            # Similarity score with color coding
            similarity_score = metrics['cosine_similarity']
            st.metric("Similarity Score", f"{similarity_score:.2%}")
            
            # Additional metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Common Words", metrics['common_words'])
            with col2:
                st.metric("Unique Words (Text 1)", metrics['unique_words_text1'])
            with col3:
                st.metric("Unique Words (Text 2)", metrics['unique_words_text2'])
            
            # Visualization
            st.subheader("Similarity Visualization")
            fig = plot_similarity_metrics(metrics)
            st.pyplot(fig)
            
            # Interpretation
            st.subheader("Interpretation")
            if similarity_score >= 0.8:
                st.warning("⚠️ High similarity detected! These texts might be considered duplicate content.")
            elif similarity_score >= 0.5:
                st.info("ℹ️ Moderate similarity detected. Some content overlap exists.")
            else:
                st.success("✅ Low similarity. These texts are substantially different.")

if __name__ == "__main__":
    main()
