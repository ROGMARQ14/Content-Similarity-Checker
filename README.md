# Text Similarity Analyzer

A powerful Streamlit-based application for analyzing text similarity, specifically designed for SEO professionals to detect duplicate content issues.

## Features

- Advanced text preprocessing including tokenization, lemmatization, and stopword removal
- Multiple similarity metrics:
  - Cosine similarity using TF-IDF vectorization
  - Common words analysis
  - Unique words count for each text
- Interactive visualization of similarity scores
- Real-time analysis and interpretation
- User-friendly Streamlit interface

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

3. Visualization:
   - Displays similarity score with color-coded interpretation
   - Shows detailed metrics
   - Provides a heatmap visualization

## Interpretation Guide

- Similarity Score ≥ 80%: High similarity, potential duplicate content
- Similarity Score 50-79%: Moderate similarity, some content overlap
- Similarity Score < 50%: Low similarity, substantially different content

## Requirements

- Python 3.7+
- NLTK
- scikit-learn
- Streamlit
- pandas
- matplotlib
- seaborn

## License

This project is licensed under the MIT License - see the LICENSE file for details.
