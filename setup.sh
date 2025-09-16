#!/bin/bash

# Create necessary directories
mkdir -p models
mkdir -p .streamlit

# Download NLTK data
python -c "import nltk; nltk.download('stopwords')"

echo "Setup complete!"
