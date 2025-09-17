import os
import re
import joblib
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords if not already present
nltk.download("stopwords", quiet=True)

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()


def preprocess_text(text: str) -> str:
    """
    Clean and preprocess input text for fake news detection.
    
    Steps:
    - Lowercasing
    - Removing special characters/numbers
    - Tokenization
    - Stopword removal
    - Stemming
    
    Args:
        text (str): Raw news article text
    
    Returns:
        str: Preprocessed text
    """
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # keep only letters
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)


def extract_article_from_url(url: str) -> str:
    """
    Extract article text from a news URL using requests + BeautifulSoup.
    
    Args:
        url (str): News article URL
    
    Returns:
        str: Extracted article text (cleaned) or empty string if failed
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Common article containers
        paragraphs = soup.find_all("p")
        text = " ".join([p.get_text() for p in paragraphs])
        return text.strip()
    except Exception as e:
        print(f"Error extracting article: {e}")
        return ""


def load_models():
    """
    Load the trained model and TF-IDF vectorizer.
    
    Returns:
        tuple: (model, vectorizer) or (None, None) if loading fails
    """
    try:
        # Get the current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Navigate to the models directory
        models_dir = os.path.join(current_dir, "..", "models")

        model_path = os.path.join(models_dir, "fake_news_model_tuned.pkl")
        vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")

        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return model, vectorizer
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None
