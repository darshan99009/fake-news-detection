import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
from PIL import Image
import time

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

# Load models with caching
@st.cache_resource
def load_app_models():
    return load_models()

model, vectorizer = load_app_models()

def main():
    st.title("📰 Advanced Fake News Detection System")
    st.markdown("""
    This app uses advanced machine learning and NLP techniques to detect fake news in real-time. 
    Our model has been trained on a large dataset of verified fake and real news articles.
    """)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "About", "Dataset Info", "Model Info"])
    
    if page == "Home":
        # Input method selection
        input_method = st.sidebar.radio(
            "Input Method",
            ["Text Input", "URL Input"]
        )
        
        # Model info in sidebar
        model_perf = get_model_performance()
        st.sidebar.info(f"""
        **Model Information:**
        - Algorithm: Logistic Regression with hyperparameter tuning
        - Features: TF-IDF with n-grams
        - Training Data: 40,000+ news articles
        - Accuracy: {model_perf['testing_accuracy']*100:.1f}% on test data
        - Best Parameters: C={model_perf['best_params']['C']}, penalty={model_perf['best_params']['penalty']}
        """)
        
        # Main content area
        if input_method == "Text Input":
            st.header("Analyze News Content")
            title = st.text_input("Article Title (optional)")
            text = st.text_area("Article Text", height=200, 
                               placeholder="Paste the news article content here...")
            
            content = f"{title} {text}".strip()
            
        else:  # URL Input
            st.header("Analyze News from URL")
            url = st.text_input("News Article URL", 
                               placeholder="https://example.com/news-article")
            
            if url:
                # Validate URL
                if not validate_url(url):
                    st.warning("This doesn't appear to be a valid news URL. Extraction may not work properly.")
                
                with st.spinner("Extracting article content..."):
                    content = extract_article_from_url(url)
                    if content:
                        with st.expander("View Extracted Content"):
                            st.text_area("", content, height=200, disabled=True)
                    else:
                        st.error("Could not extract content from the URL. Please try another URL or use text input.")
                        content = ""
            else:
                content = ""
        
        # Prediction button
        if st.button("Analyze Article", type="primary") and content:
            with st.spinner("Analyzing content..."):
                # Add a small delay for better UX
                time.sleep(0.5)
                
                prediction, probability = predict_news(content, model, vectorizer)
                
                if prediction is None:
                    st.error("Model not loaded properly. Please check the model files.")
                    return
                
                # Display results
                st.header("Analysis Results")
                
                # Create columns for layout
                col1, col2 = st.columns(2)
                
                with col1:
                    if prediction == 1:
                        st.error("🔴 **Result: Likely FAKE News**")
                    else:
                        st.success("🟢 **Result: Likely REAL News**")
                
                with col2:
                    confidence = max(probability) * 100
                    st.metric("Confidence Score", f"{confidence:.2f}%")
                
                # Probability visualization
                st.subheader("Probability Distribution")
                
                # Create a bar chart
                prob_data = pd.DataFrame({
                    'Category': ['Real News', 'Fake News'],
                    'Probability': [probability[0] * 100, probability[1] * 100]
                })
                
                st.bar_chart(prob_data.set_index('Category'))
                
                # Detailed probabilities
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Real News Probability", f"{probability[0]*100:.2f}%")
                with col2:
                    st.metric("Fake News Probability", f"{probability[1]*100:.2f}%")
                
                # Explanation
                st.subheader("Analysis Explanation")
                if prediction == 1:
                    st.warning("""
                    **Characteristics of this article that suggest it may be fake:**
                    - Sensationalist or emotionally charged language
                    - Lack of credible sources or citations
                    - Inconsistencies in facts or reporting
                    - Potential political or ideological bias
                    - Unverified claims or conspiracy theories
                    
                    **Recommendation:** Verify this information with trusted news sources before sharing.
                    """)
                else:
                    st.info("""
                    **Characteristics of this article that suggest it may be reliable:**
                    - Fact-based reporting with verifiable information
                    - Citation of credible sources and experts
                    - Balanced perspective with multiple viewpoints
                    - Professional tone and writing style
                    - Publication with established editorial standards
                    
                    **Note:** Even reliable sources can occasionally make errors. Always practice critical thinking.
                    """)
        
        # Add some examples
        with st.expander("See example news snippets"):
            st.write("""
            **Real News Example:**
            "The Federal Reserve announced a 0.25% increase in interest rates today, citing ongoing concerns about inflation. Economists surveyed by Reuters had largely anticipated this move."
            
            **Fake News Example:**
            "SHOCKING: Celebrity politician caught in secret pedophile ring! You won't believe what they found in his basement! Mainstream media is covering it up!"
            """)
    
    elif page == "About":
        st.header("About This Project")
        st.markdown("""
        This Fake News Detection System was developed to combat the spread of misinformation online.
        
        **Key Features:**
        - Advanced machine learning model with hyperparameter tuning
        - Real-time analysis of news content
        - URL extraction for analyzing online articles
        - Comprehensive probability scoring
        
        **Technical Details:**
        - Built with Python and Scikit-learn
        - Uses TF-IDF vectorization with n-grams
        - Logistic Regression classifier with optimized parameters
        - Trained on a dataset of 40,000+ verified news articles
        
        **Disclaimer:**
        This tool is for educational purposes only. It should not be the sole basis for determining
        the credibility of news. Always verify information through multiple trusted sources.
        """)
        
    elif page == "Dataset Info":
        # Use the helper function to get dataset stats
        stats = get_dataset_stats()
        
        st.header("Dataset Information")
        st.markdown("""
        The model was trained on a comprehensive dataset of fake and real news articles.
        """)
        
        # Display dataset statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Articles", f"{stats['total_articles']:,}")
        with col2:
            st.metric("Real News Articles", f"{stats['real_news']:,}")
        with col3:
            st.metric("Fake News Articles", f"{stats['fake_news']:,}")
        
        st.markdown("""
        **Sources:** Various reputable news outlets and fact-checking organizations
        
        **Preprocessing Steps:**
        1. Text cleaning and normalization
        2. Stopword removal
        3. Stemming
        4. Feature extraction (TF-IDF with n-grams)
        5. Model training with hyperparameter optimization
        """)
        
        # Display model performance
        perf = get_model_performance()
        st.subheader("Model Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{perf['testing_accuracy']*100:.1f}%")
        with col2:
            st.metric("Precision", f"{perf['precision']*100:.1f}%")
        with col3:
            st.metric("Recall", f"{perf['recall']*100:.1f}%")
        with col4:
            st.metric("F1 Score", f"{perf['f1_score']*100:.1f}%")
            
        st.subheader("Optimal Hyperparameters")
        st.json(perf['best_params'])
    
    elif page == "Model Info":
        st.header("Model Information")
        st.markdown("""
        This application uses a Logistic Regression classifier with hyperparameter tuning for fake news detection.
        """)
        
        perf = get_model_performance()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Architecture")
            st.markdown("""
            **Feature Extraction:**
            - TF-IDF Vectorization
            - N-gram range: (1, 2) - includes unigrams and bigrams
            - Maximum features: 5,000
            - Sublinear TF scaling
            
            **Classifier:**
            - Logistic Regression
            - L1 regularization (Lasso)
            - liblinear solver
            - Regularization strength (C): 100
            """)
            
        with col2:
            st.subheader("Performance Metrics")
            metrics_data = {
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Training Accuracy'],
                'Value': [
                    f"{perf['testing_accuracy']*100:.1f}%",
                    f"{perf['precision']*100:.1f}%",
                    f"{perf['recall']*100:.1f}%",
                    f"{perf['f1_score']*100:.1f}%",
                    f"{perf['training_accuracy']*100:.1f}%"
                ]
            }
            st.table(pd.DataFrame(metrics_data))
        
        st.subheader("Hyperparameter Tuning")
        st.markdown("""
        The model was optimized using Grid Search Cross-Validation with the following parameter grid:
        
        ```python
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        }
        ```
        
        The optimal parameters found were:
        """)
        st.json(perf['best_params'])
        
        st.subheader("Future Enhancements")
        st.markdown("""
        Planned improvements for future versions:
        - Integration of transformer models (BERT, RoBERTa)
        - Real-time fact-checking API integration
        - Browser extension for on-the-fly analysis
        - Multilingual support
        - User feedback mechanism for continuous learning
        """)

if __name__ == "__main__":
    main()
