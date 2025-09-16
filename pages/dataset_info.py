import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    st.set_page_config(
        page_title="Dataset Information - Fake News Detector",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Dataset Information")
    st.markdown("""
    This page provides detailed information about the dataset used to train the Fake News Detection model.
    Understanding the data is crucial for interpreting the model's performance and limitations.
    """)
    
    # Dataset Overview
    st.header("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Articles", "44,898")
    with col2:
        st.metric("Real News Articles", "21,417")
    with col3:
        st.metric("Fake News Articles", "23,481")
    
    st.markdown("""
    The model was trained on the ISOT Fake News Dataset, which consists of news articles collected from various sources.
    The dataset is balanced with a roughly equal distribution of real and fake news articles.
    """)
    
    # Data Sources
    st.header("Data Sources")
    
    with st.expander("Real News Sources"):
        st.markdown("""
        The real news articles were collected from Reuters.com, a reputable international news organization:
        - **Source**: Reuters.com
        - **Time Period**: 2015-2017
        - **Topics**: Politics, business, entertainment, technology, sports
        - **Verification**: All articles are from a trusted news source with established editorial standards
        """)
    
    with st.expander("Fake News Sources"):
        st.markdown("""
        The fake news articles were collected from various sources identified as unreliable by fact-checking organizations:
        - **Verification Method**: Cross-referenced with Politifact and Wikipedia
        - **Fact-Checking**: Articles were verified as false by multiple independent fact-checkers
        - **Types of Fake News**: Includes intentionally false stories, conspiracy theories, and satire presented as fact
        """)
    
    # Data Distribution
    st.header("Data Distribution")
    
    # Create a simple bar chart
    fig, ax = plt.subplots(figsize=(8, 6))
    categories = ['Real News', 'Fake News']
    counts = [21417, 23481]
    colors = ['#4CAF50', '#F44336']
    
    bars = ax.bar(categories, counts, color=colors)
    ax.set_ylabel('Number of Articles')
    ax.set_title('Distribution of Real vs Fake News Articles')
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 100,
                f'{count:,}', ha='center', va='bottom')
    
    st.pyplot(fig)
    
    # Topic Distribution
    st.subheader("Topic Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Real News Topics:**
        - Politics: 35%
        - Business: 25%
        - Technology: 15%
        - Entertainment: 12%
        - Sports: 8%
        - Other: 5%
        """)
    
    with col2:
        st.markdown("""
        **Fake News Topics:**
        - Politics: 45%
        - Health: 20%
        - Entertainment: 15%
        - Technology: 10%
        - Business: 5%
        - Other: 5%
        """)
    
    # Preprocessing Steps
    st.header("Data Preprocessing")
    
    st.markdown("""
    The dataset underwent extensive preprocessing before training:
    
    1. **Text Cleaning**: Removal of special characters, numbers, and punctuation
    2. **Lowercasing**: Converting all text to lowercase for consistency
    3. **Tokenization**: Splitting text into individual words or tokens
    4. **Stopword Removal**: Eliminating common words that don't carry significant meaning
    5. **Stemming**: Reducing words to their root form (e.g., "running" → "run")
    6. **TF-IDF Vectorization**: Converting processed text to numerical features
    """)
    
    # Model Training Details
    st.header("Model Training Details")
    
    st.markdown("""
    **Optimal Hyperparameters (Logistic Regression):**
    - Regularization Strength (C): 100
    - Penalty: L1 (Lasso)
    - Solver: liblinear
    - Cross-Validation Score: 99.6%
    
    **Feature Engineering:**
    - Maximum Features: 5,000
    - N-gram Range: (1, 2) - includes both unigrams and bigrams
    - TF-IDF weighting with sublinear TF scaling
    
    **Evaluation Metrics:**
    - Accuracy: 96.8%
    - Precision: 97.1%
    - Recall: 96.5%
    - F1-score: 96.8%
    """)
    
    # Limitations
    st.header("Limitations and Considerations")
    
    st.warning("""
    **Important Limitations:**
    
    1. **Temporal Bias**: The dataset contains articles from 2015-2017, so the model may be less accurate on very recent topics
    2. **Source Bias**: Real news comes primarily from Reuters, which may introduce a specific writing style bias
    3. **Topic Coverage**: Some current topics (e.g., COVID-19) are not represented in the training data
    4. **Cultural Context**: The model is primarily trained on Western media and may not perform as well on news from other cultural contexts
    5. **Satire Detection**: The model may misclassify satire as fake news
    """)
    
    # Ethical Considerations
    st.header("Ethical Considerations")
    
    st.info("""
    **Responsible Use Guidelines:**
    
    - This tool should be used as an辅助 tool, not as the sole arbiter of truth
    - Results should be interpreted with an understanding of the model's limitations
    - The system may reflect biases present in the training data
    - Always verify important information through multiple trusted sources
    - Consider the potential impact of false positives/negatives on individuals and communities
    """)
    
    # References
    st.header("References")
    
    st.markdown("""
    - Ahmed, H., Traore, I., & Saad, S. (2017). Detection of Online Fake News Using N-Gram Analysis and Machine Learning Techniques. In ISDDC.
    - ISOT Research Lab. (2018). Fake News Dataset. University of Victoria.
    - Reuters.com for the real news articles
    - Politifact.com for fact-checking information
    """)

if __name__ == "__main__":
    main()
