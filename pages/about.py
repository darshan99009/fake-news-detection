import streamlit as st

def main():
    st.title("About This Project")
    
    st.header("Project Overview")
    st.write("""
    This Fake News Detection System is an advanced machine learning application designed to identify
    potentially misleading or false information in news articles. The system analyzes text content
    using natural language processing techniques and provides a confidence score regarding the
    article's likely veracity.
    """)
    
    st.header("How It Works")
    st.write("""
    1. **Text Input**: Users can paste news article text directly into the application
    2. **URL Input**: The system can extract content from news article URLs
    3. **Preprocessing**: The text is cleaned, tokenized, and normalized
    4. **Feature Extraction**: TF-IDF vectorization converts text to numerical features
    5. **Classification**: A trained machine learning model predicts the article's authenticity
    6. **Results**: The system provides a confidence score and detailed analysis
    """)
    
    st.header("Technical Details")
    st.write("""
    - **Framework**: Streamlit for the web interface
    - **Machine Learning**: Scikit-learn for model training and inference
    - **NLP Processing**: NLTK for text preprocessing
    - **Web Scraping**: BeautifulSoup for content extraction from URLs
    - **Deployment**: Streamlit Sharing for cloud deployment
    """)
    
    st.header("Model Information")
    st.write("""
    The classification model is a Logistic Regression classifier with optimized hyperparameters:
    - **C**: 10 (Regularization strength)
    - **Penalty**: L2 (Ridge regularization)
    - **Solver**: liblinear
    - **Max Features**: 5000
    - **N-gram Range**: (1, 2) - includes unigrams and bigrams
    """)
    
    st.header("Dataset")
    st.write("""
    The model was trained on the ISOT Fake News Dataset, which contains:
    - 21,417 real news articles from Reuters.com
    - 23,481 fake news articles from various sources verified by Politifact and Wikipedia
    - Articles cover a wide range of topics including politics, entertainment, and technology
    """)
    
    st.header("Limitations")
    st.write("""
    While the system achieves high accuracy, it has some limitations:
    - Performance may vary with very recent news topics not seen during training
    - Satire or opinion pieces might be misclassified
    - Highly sophisticated disinformation campaigns might evade detection
    - The system should be used as a tool, not as the sole arbiter of truth
    """)
    
    st.header("Future Enhancements")
    st.write("""
    Planned improvements for future versions:
    - Integration of transformer models (BERT, RoBERTa) for improved accuracy
    - Real-time fact-checking API integration
    - Browser extension for on-the-fly analysis
    - Multilingual support
    - User feedback mechanism for continuous learning
    """)

if __name__ == "__main__":
    main()
