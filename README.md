# Text Similarity Analyzer

A streamlined Streamlit application for analyzing text similarity, specifically designed for SEO professionals to detect duplicate content issues.

## Features

- Advanced text preprocessing including tokenization, lemmatization, and stopword removal
- Multiple similarity metrics:
  - Cosine similarity using TF-IDF vectorization
  - Common words analysis
  - Unique words count for each text
- Clean, simple data presentation
- Real-time analysis and interpretation
- User-friendly Streamlit interface

## Requirements

- Python 3.7+
- streamlit==1.24.0
- nltk==3.8.1
- scikit-learn==1.2.2
- numpy==1.23.5
- pandas==1.5.3

## Installation

1. Clone this repository:
```bash
git clone [your-repository-url]
cd text_similarity_app
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided local URL (typically http://localhost:8501)

3. Enter your texts in the two text areas and click "Analyze Similarity"

## How It Works

The application performs the following steps:
1. Text Preprocessing:
   - Converts text to lowercase
   - Removes special characters
   - Tokenizes the text
   - Removes stopwords
   - Applies lemmatization

2. Similarity Analysis:
   - Calculates TF-IDF vectors
   - Computes cosine similarity
   - Analyzes common and unique words

3. Results Display:
   - Shows similarity score with interpretation
   - Displays detailed metrics in a clear table format
   - Provides content duplication warnings when necessary

## Interpretation Guide

- Similarity Score ≥ 80%: High similarity, potential duplicate content
- Similarity Score 50-79%: Moderate similarity, some content overlap
- Similarity Score < 50%: Low similarity, substantially different content
