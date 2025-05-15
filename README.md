# EmoDec - Emotion Detection in Text

This project demonstrates emotion detection on text using two approaches:

1. **Classical Machine Learning**  
   A Naive Bayes classifier trained on a small sample dataset with TF-IDF features to classify emotions like joy, sadness, anger, surprise, and fear.

2. **Pre-trained Transformer Model**  
   Utilizes the Hugging Face transformer model [`j-hartmann/emotion-english-distilroberta-base`] for emotion detection without needing to train from scratch.

---

## What I Did

- Created a small dataset of sentences labeled with emotions.
- Preprocessed text data (lowercasing).
- Vectorized text using TF-IDF.
- Trained and evaluated a Multinomial Naive Bayes classifier.
- Integrated Hugging Face's pre-trained emotion detection pipeline for more advanced, out-of-the-box predictions.
- Combined both methods in a Python script (`main.py`).

---

## How to Use

1. **Clone the repo**

   ```bash
   git clone https://github.com/sundramxkumar/EmoDec.git
   cd EmoDec
2. **Create and activate a Python virtual environment (optional but recommended)**
    python -m venv .venv
     .venv\Scripts\activate    # Windows
    source .venv/bin/activate # Linux/macOS
