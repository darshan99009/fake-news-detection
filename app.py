# ... (other imports)

# Import helper functions
from utils.helpers import (
    preprocess_text, 
    extract_article_from_url, 
    load_models, 
    predict_news,
    validate_url,
    get_dataset_stats,
    get_model_performance
)

# Set page configuration
st.set_page_config(
    page_title="Advanced Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Download stopwords
try:
    nltk.download('stopwords')
except:
    st.warning("Could not download NLTK stopwords. Some features might not work properly.")

# Load models with caching and error handling
@st.cache_resource
def load_app_models():
    model, vectorizer = load_models()
    if model is None or vectorizer is None:
        st.error("""
        ❌ Failed to load model files. Please ensure:
        1. The 'models' directory exists
        2. It contains 'fake_news_model_tuned.pkl' and 'tfidf_vectorizer.pkl'
        3. The files are not corrupted
        """)
        # Create placeholder objects to prevent further errors
        class PlaceholderModel:
            def predict(self, X):
                return [0]
            def predict_proba(self, X):
                return [[0.5, 0.5]]
        
        class PlaceholderVectorizer:
            def transform(self, X):
                # Return a dummy matrix
                return np.zeros((1, 1000))
        
        return PlaceholderModel(), PlaceholderVectorizer()
    return model, vectorizer

model, vectorizer = load_app_models()

# ... (rest of your app code)
