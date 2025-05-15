import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from transformers import pipeline
import pandas as pd

# Download necessary NLTK data
nltk.download('stopwords')

# Sample dataset
data = {
    'text': [
        "I am so happy today!",
        "I am feeling very sad.",
        "What a beautiful day!",
        "I am angry with you!",
        "I feel so surprised and amazed.",
        "I hate waiting in lines.",
        "This is awesome!",
        "I don’t know what to do, I feel lost.",
        "I want to kill myself.",
        "I hate you and will never talk you again.",
        "I am so depressed and now all i do is cry all day long",
    ],
    'emotion': ['joy', 'sadness', 'joy', 'anger', 'surprise', 'anger', 'joy', 'fear', 'sadness', 'anger', 'sadness']
}

df = pd.DataFrame(data)

# Preprocessing
def preprocess_text(text):
    return text.lower()

df['text'] = df['text'].apply(preprocess_text)

# Features and labels
X = df['text']
y = df['emotion']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train model
classifier = MultinomialNB()
classifier.fit(X_train_tfidf, y_train)

# Predict
y_pred = classifier.predict(X_test_tfidf)

# Evaluation
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Pre-trained Hugging Face Emotion Detector
emotion_pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

sample_text = "I think my father will be happy if i will score higher in my annual exams and that makes me feel so relieved"
emotion = emotion_pipeline(sample_text)
print(f"Detected Emotion: {emotion[0]['label']}, with score: {emotion[0]['score']:.2f}")
