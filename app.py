import streamlit as st
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from docx import Document
from PyPDF2 import PdfReader
import io

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Cache the stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def read_file_content(uploaded_file):
    """
    Read content from uploaded file based on its type
    """
    if uploaded_file is None:
        return ""
        
    file_type = uploaded_file.name.split('.')[-1].lower()
    content = ""
    
    try:
        if file_type == 'txt':
            content = uploaded_file.getvalue().decode('utf-8')
        elif file_type == 'docx':
            doc = Document(io.BytesIO(uploaded_file.getvalue()))
            content = ' '.join([paragraph.text for paragraph in doc.paragraphs])
        elif file_type == 'pdf':
            pdf = PdfReader(io.BytesIO(uploaded_file.getvalue()))
            content = ' '.join([page.extract_text() for page in pdf.pages if page.extract_text()])
        else:
            st.error(f"Unsupported file type: {file_type}")
            return ""
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return ""
        
    return content

@st.cache_data
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
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)

@st.cache_data
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

def main():
    st.title("Text Similarity Analyzer")
    st.write("Compare two texts and analyze their similarity for SEO purposes.")
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["Upload Files", "Direct Text Input"]
    )
    
    text1 = ""
    text2 = ""
    
    if input_method == "Upload Files":
        st.info("Supported file formats: TXT, DOCX, PDF")
        col1, col2 = st.columns(2)
        with col1:
            file1 = st.file_uploader("Upload first file", type=['txt', 'docx', 'pdf'])
            if file1:
                text1 = read_file_content(file1)
        with col2:
            file2 = st.file_uploader("Upload second file", type=['txt', 'docx', 'pdf'])
            if file2:
                text2 = read_file_content(file2)
    else:
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
            df = pd.DataFrame({
                'Metric': ['Similarity Score', 'Common Words', 'Unique Words (Text 1)', 'Unique Words (Text 2)'],
                'Value': [
                    f"{metrics['cosine_similarity']:.2%}",
                    metrics['common_words'],
                    metrics['unique_words_text1'],
                    metrics['unique_words_text2']
                ]
            })
            
            st.dataframe(df, use_container_width=True)
            
            # Interpretation
            st.subheader("Interpretation")
            similarity_score = metrics['cosine_similarity']
            if similarity_score >= 0.8:
                st.warning("⚠️ High similarity detected! These texts might be considered duplicate content.")
            elif similarity_score >= 0.5:
                st.info("ℹ️ Moderate similarity detected. Some content overlap exists.")
            else:
                st.success("✅ Low similarity. These texts are substantially different.")

if __name__ == "__main__":
    main()
