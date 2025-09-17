import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import requests
from bs4 import BeautifulSoup
import pandas as pd
import joblib
import numpy as np
import os

# Download NLTK data
try:
    nltk.download('stopwords')
except:
    pass

# Initialize text preprocessing tools
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english')) if 'stopwords' in nltk.data.find('corpora') else set()

def preprocess_text(text):
    """
    Preprocess text by removing special characters, lowercasing,
    removing stopwords, and applying stemming.
    
    Args:
        text (str): Input text to preprocess
        
    Returns:
        str: Preprocessed text
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    # Convert to lowercase
    text = text.lower()
    # Tokenize
    tokens = text.split()
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    # Apply stemming
    tokens = [stemmer.stem(word) for word in tokens]
    # Join back to string
    return ' '.join(tokens)

def extract_article_from_url(url):
    """
    Extract article text from a given URL.
    
    Args:
        url (str): URL of the news article
        
    Returns:
        str: Extracted article text or None if extraction fails
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
        
        # Extract title
        title = soup.find('title')
        title_text = title.get_text() if title else ""
        
        # Try to find main content using common selectors
        article_text = ""
        possible_selectors = [
            'article',
            '.article',
            '#article',
            '.content',
            '#content',
            '.post-content',
            '.entry-content',
            '.story-content',
            '[class*="content"]',
            '[id*="content"]'
        ]
        
        for selector in possible_selectors:
            elements = soup.select(selector)
            if elements:
                for element in elements:
                    article_text += element.get_text() + " "
        
        # Fallback: extract all paragraphs
        if not article_text.strip():
            paragraphs = soup.find_all('p')
            for p in paragraphs:
                if len(p.get_text()) > 50:  # Only include substantial paragraphs
                    article_text += p.get_text() + " "
        
        return title_text + " " + article_text
    except Exception as e:
        print(f"Error extracting article from URL: {e}")
        return None

def load_models():
    """
    Load the trained model and vectorizer.
    
    Returns:
        tuple: (model, vectorizer) or (None, None) if loading fails
    """
    try:
        # Get the current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Navigate to the models directory
        models_dir = os.path.join(current_dir, '..', 'models')
        
        model_path = os.path.join(models_dir, 'fake_news_model_tuned.pkl')
        vectorizer_path = os.path.join(models_dir, 'tfidf_vectorizer.pkl')
        
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return model, vectorizer
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None

def predict_news(text, model, vectorizer):
    """
    Predict whether a given text is real or fake news.
    
    Args:
        text (str): Text to analyze
        model: Trained classification model
        vectorizer: Fitted TF-IDF vectorizer
        
    Returns:
        tuple: (prediction, probability) or (None, None) if prediction fails
    """
    if model is None or vectorizer is None:
        return None, None
        
    try:
        processed_text = preprocess_text(text)
        text_vector = vectorizer.transform([processed_text])
        prediction = model.predict(text_vector)
        probability = model.predict_proba(text_vector)
        return prediction[0], probability[0]
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None

def validate_url(url):
    """
    Validate if a URL is likely a news article URL.
    
    Args:
        url (str): URL to validate
        
    Returns:
        bool: True if URL appears valid, False otherwise
    """
    # Basic URL validation
    if not url or not isinstance(url, str):
        return False
    
    # Check if URL has a valid format
    url_pattern = re.compile(
        r'^(https?://)?'  # http:// or https://
        r'(([A-Z0-9-]+\.)+[A-Z]{2,})'  # domain
        r'(:[0-9]+)?'  # port
        r'(/.*)?$',  # path
        re.IGNORECASE
    )
    
    if not re.match(url_pattern, url):
        return False
    
    # Check if URL is from a common news domain
    news_domains = [
        'news', 'article', 'blog', 'post', 
        'reuters', 'bbc', 'cnn', 'nytimes', 'washingtonpost',
        'theguardian', 'wsj', 'bloomberg', 'apnews'
    ]
    
    return any(domain in url.lower() for domain in news_domains)

def get_dataset_stats():
    """
    Get statistics about the training dataset.
    
    Returns:
        dict: Dictionary containing dataset statistics
    """
    return {
        'total_articles': 44898,
        'real_news': 21417,
        'fake_news': 23481,
        'real_topics': {
            'Politics': 35,
            'Business': 25,
            'Technology': 15,
            'Entertainment': 12,
            'Sports': 8,
            'Other': 5
        },
        'fake_topics': {
            'Politics': 45,
            'Health': 20,
            'Entertainment': 15,
            'Technology': 10,
            'Business': 5,
            'Other': 5
        }
    }

def get_model_performance():
    """
    Get performance metrics of the trained model.
    
    Returns:
        dict: Dictionary containing model performance metrics
    """
    return {
        'training_accuracy': 0.996,
        'testing_accuracy': 0.968,
        'precision': 0.971,
        'recall': 0.965,
        'f1_score': 0.968,
        'best_params': {
            'C': 100,
            'penalty': 'l1',
            'solver': 'liblinear'
        }
    }

# Test function to verify the helpers module works correctly
def test_helpers():
    """Test the helper functions"""
    print("Testing helper functions...")
    
    # Test preprocess_text
    test_text = "This is a TEST text with 123 numbers and special characters!"
    processed = preprocess_text(test_text)
    print(f"Preprocess test: '{test_text}' -> '{processed}'")
    
    # Test validate_url
    test_urls = [
        "https://www.bbc.com/news",
        "https://example.com",
        "not-a-url"
    ]
    for url in test_urls:
        is_valid = validate_url(url)
        print(f"URL validation: '{url}' -> {is_valid}")
    
    # Test get_dataset_stats and get_model_performance
    stats = get_dataset_stats()
    perf = get_model_performance()
    print(f"Dataset stats: {stats}")
    print(f"Model performance: {perf}")
    
    print("Helper functions test completed!")

if __name__ == "__main__":
    test_helpers()
