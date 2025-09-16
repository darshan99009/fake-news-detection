import os
import joblib

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
