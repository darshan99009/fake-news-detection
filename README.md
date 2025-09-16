# 📰 Advanced Fake News Detection System  

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)]()  
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.0-orange)]()  
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-red)]()  
[![NLP](https://img.shields.io/badge/NLP-Transformers-yellowgreen)]()  

An **AI-powered web application** that detects fake news articles using **machine learning** and **natural language processing**, optimized with hyperparameter tuning for maximum accuracy.  

---
#Try it here 
https://fake-news-detection-darshan.streamlit.app/
---
## ✨ Features
- ⚡ **Real-time Analysis** → Instant verification of news articles  
- 🖊️ **Dual Input Methods** → Text input or URL scraping  
- 🤖 **Advanced ML Model** → Logistic Regression with tuned hyperparameters  
- 📊 **Detailed Analytics** → Confidence scores & probability distributions  
- 🎨 **Clean UI** → Responsive Streamlit interface  
- 🌐 **Web Scraping** → Automatic content extraction from URLs  

---

## 🚀 Try It Out
The app is deployed on **Streamlit Sharing**:  
👉 [Live Demo](https://your-username-fake-news-detector.streamlit.app/)  

---

## 🏗️ Model Architecture  

### 🔧 Optimal Hyperparameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| **C**     | 100   | Inverse of regularization strength |
| **Penalty** | L1 | Lasso regularization |
| **Solver** | liblinear | Optimization algorithm |
| **Best CV Score** | 0.996 | 5-fold CV accuracy |

### 📈 Performance Metrics
- Training Accuracy: **99.6%**  
- Testing Accuracy: **96.8%**  
- Precision: **97.1%**  
- Recall: **96.5%**  
- F1-score: **96.8%**  

---

## 📊 Dataset
- Source: **ISOT Fake News Dataset**  
- Total Articles: **44,898**  
  - Real: 21,417 (Reuters.com)  
  - Fake: 23,481 (PolitiFact, Wikipedia)  
- Topics: Politics, World News, Entertainment, Technology  

---

## 🛠 Installation  

--Clone the repository:

$ git clone https://github.com/darshan99009/fake-news-detection.git
$ cd fake-news-detector
Install dependencies:


$ pip install -r requirements.txt
Download NLTK stopwords:


import nltk
nltk.download('stopwords')
Run the Streamlit app:
$ streamlit run app.py

📁 Project Structure
```fake-news-detector/
├── app.py                   # Main Streamlit app
├── requirements.txt         # Python dependencies
├── setup.sh                 # Env setup script
├── models/
│   ├── fake_news_model_tuned.pkl    # Trained ML model
│   └── tfidf_vectorizer.pkl         # Vectorizer
├── pages/
│   ├── about.py             # About project
│   └── dataset_info.py      # Dataset details
├── .streamlit/
│   └── config.toml          # Streamlit config
├── utils
│   └── helpers.py 
└── README.md                # Documentation
```
💻 Usage
Paste article text or enter a news URL

Click Analyze Article

View authenticity prediction with confidence scores

🔬 Technical Details
Preprocessing Pipeline
Text cleaning, lowercasing, tokenization

Stopword removal (NLTK)

Porter stemming

TF-IDF vectorization with n-grams (1,2)

Feature Engineering
Max features: 5000

Regularization: L1 penalty

Model Selection
Compared multiple ML models

Logistic Regression (L1) gave the best balance of accuracy + interpretability

🌟 Future Enhancements
🤗 Integration of Transformers (BERT, RoBERTa)

🔍 Real-time fact-checking API

🧩 Browser extension for instant analysis

🌍 Multilingual support

📥 User feedback loop for continuous learning

⚠️ Disclaimer
This tool is for educational purposes only. Always verify information from trusted sources before drawing conclusions.

👨‍💻 Author
Darshan Gowda S


🙏 Acknowledgments
ISOT Research Lab for dataset

Scikit-learn & Streamlit teams

Open-source community 💜

