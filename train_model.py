#!/usr/bin/env python3
"""
Advanced Model Training Script for Fake News Detection
This script allows you to train and improve the fake news detection model
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class FakeNewsModelTrainer:
    def __init__(self):
        self.ps = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = None
        self.classifier = None
        self.model_name = "fake_news_detector"
        
    def preprocess_text(self, text):
        """Advanced text preprocessing for fake news detection"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove special characters and digits but keep some punctuation
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize
        words = text.split()
        
        # Remove stopwords and apply stemming
        words = [self.ps.stem(word) for word in words if word not in self.stop_words and len(word) > 2]
        
        return ' '.join(words)
    
    def create_training_data(self):
        """Create comprehensive training data for fake news detection"""
        
        # Fake news examples (more comprehensive)
        fake_news_data = [
            # Sensational claims
            "BREAKING: Scientists discover that aliens are living among us and controlling our thoughts",
            "SHOCKING: Time travel machine invented by 15-year-old in garage",
            "INCREDIBLE: Government admits to hiding evidence of Bigfoot in secret facility",
            "AMAZING: New study shows that drinking coffee makes you immortal",
            "UNBELIEVABLE: Dragons discovered in remote mountain range",
            "STUNNING: Scientists prove that the Earth is actually flat and hollow",
            "MIRACULOUS: New technology allows humans to breathe underwater without equipment",
            "FANTASTIC: Unicorns found in Amazon rainforest",
            "WONDERFUL: Study shows that eating chocolate prevents all diseases",
            "EXTRAORDINARY: Government admits to weather control technology",
            
            # Clickbait headlines
            "You won't believe what happened next!",
            "This one weird trick will change your life forever",
            "Doctors hate this simple method",
            "The secret they don't want you to know",
            "This will shock you to your core",
            "What happens next will amaze you",
            "The truth about [topic] that will blow your mind",
            "This simple hack will solve all your problems",
            "Scientists are baffled by this discovery",
            "The government is hiding this from you",
            
            # Conspiracy theories
            "The moon landing was completely faked",
            "Chemtrails are poisoning our atmosphere",
            "Vaccines contain microchips for tracking",
            "The Earth is hollow and inhabited inside",
            "Dinosaurs never existed - it's all a hoax",
            "Time travelers are among us",
            "Reptilian aliens control world governments",
            "The pyramids were built by aliens",
            "All world leaders are clones",
            "The sun is actually cold and artificial"
        ]
        
        # Real news examples (more comprehensive)
        real_news_data = [
            # Scientific studies
            "New study shows benefits of regular exercise on mental health",
            "Scientists discover new species of deep-sea creatures",
            "Research indicates climate change impact on global agriculture",
            "New technology improves solar panel efficiency by 25%",
            "Study finds correlation between diet and heart disease",
            "Scientists develop new method for plastic recycling",
            "Research shows benefits of meditation on stress reduction",
            "New study on renewable energy sources published",
            "Scientists discover potential treatment for Alzheimer's disease",
            "Research indicates benefits of green spaces in urban areas",
            
            # Technology news
            "Apple releases new iPhone with improved camera system",
            "Tesla announces new electric vehicle model",
            "Google updates search algorithm for better results",
            "Microsoft releases Windows 11 update",
            "Amazon expands drone delivery service",
            "Facebook announces new privacy features",
            "Netflix adds new original content",
            "Spotify introduces new playlist algorithm",
            "Uber launches new safety features",
            "Airbnb updates booking policies",
            
            # Business and economy
            "Federal Reserve announces interest rate decision",
            "Stock market reaches new record high",
            "Company reports quarterly earnings results",
            "New employment data shows job growth",
            "Inflation rate remains stable this month",
            "Housing market shows signs of recovery",
            "Oil prices fluctuate due to market conditions",
            "Retail sales increase during holiday season",
            "Manufacturing sector shows growth",
            "Service industry employment rises"
        ]
        
        # Create labels (0 for real, 1 for fake)
        labels = [1] * len(fake_news_data) + [0] * len(real_news_data)
        
        # Combine data
        all_texts = fake_news_data + real_news_data
        
        return all_texts, labels
    
    def train_model(self, algorithm='naive_bayes'):
        """Train the fake news detection model"""
        
        print("🔄 Creating training data...")
        texts, labels = self.create_training_data()
        
        print("🔄 Preprocessing text...")
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        print("🔄 Creating TF-IDF features...")
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.95
        )
        
        X = self.vectorizer.fit_transform(processed_texts)
        y = np.array(labels)
        
        print(f"🔄 Training {algorithm} classifier...")
        
        if algorithm == 'naive_bayes':
            self.classifier = MultinomialNB(alpha=0.1)
        elif algorithm == 'random_forest':
            self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        elif algorithm == 'svm':
            self.classifier = SVC(kernel='linear', probability=True, random_state=42)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Split data for evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train the model
        self.classifier.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Model trained successfully!")
        print(f"📊 Accuracy: {accuracy:.2%}")
        print(f"📊 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))
        
        # Cross-validation
        cv_scores = cross_val_score(self.classifier, X, y, cv=5)
        print(f"📊 Cross-validation scores: {cv_scores}")
        print(f"📊 Average CV accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std() * 2:.2%})")
        
        return accuracy
    
    def save_model(self, filename=None):
        """Save the trained model and vectorizer"""
        if filename is None:
            filename = f"{self.model_name}.joblib"
        
        model_data = {
            'vectorizer': self.vectorizer,
            'classifier': self.classifier,
            'preprocessor': self.preprocess_text
        }
        
        joblib.dump(model_data, filename)
        print(f"💾 Model saved to {filename}")
    
    def load_model(self, filename):
        """Load a trained model"""
        if not os.path.exists(filename):
            print(f"❌ Model file {filename} not found")
            return False
        
        model_data = joblib.load(filename)
        self.vectorizer = model_data['vectorizer']
        self.classifier = model_data['classifier']
        print(f"✅ Model loaded from {filename}")
        return True
    
    def predict(self, text):
        """Make a prediction on new text"""
        if self.classifier is None or self.vectorizer is None:
            print("❌ Model not trained or loaded")
            return None
        
        processed_text = self.preprocess_text(text)
        text_vector = self.vectorizer.transform([processed_text])
        
        prediction = self.classifier.predict(text_vector)[0]
        probability = self.classifier.predict_proba(text_vector)[0]
        
        return {
            'is_fake': bool(prediction),
            'confidence': {
                'fake': round(probability[1] * 100, 2),
                'real': round(probability[0] * 100, 2)
            },
            'processed_text': processed_text
        }

def main():
    """Main function to demonstrate model training"""
    
    print("🚀 Fake News Detection Model Trainer")
    print("=" * 50)
    
    trainer = FakeNewsModelTrainer()
    
    # Train different models
    algorithms = ['naive_bayes', 'random_forest', 'svm']
    
    best_accuracy = 0
    best_algorithm = None
    
    for algorithm in algorithms:
        print(f"\n🔄 Training {algorithm.upper()} model...")
        print("-" * 30)
        
        try:
            accuracy = trainer.train_model(algorithm)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_algorithm = algorithm
                
        except Exception as e:
            print(f"❌ Error training {algorithm}: {e}")
    
    print(f"\n🏆 Best performing algorithm: {best_algorithm} (Accuracy: {best_accuracy:.2%})")
    
    # Retrain the best model
    print(f"\n🔄 Retraining best model ({best_algorithm})...")
    trainer.train_model(best_algorithm)
    
    # Save the model
    trainer.save_model()
    
    # Test the model
    print("\n🧪 Testing the model...")
    test_texts = [
        "Scientists discover that aliens are living among us",
        "New study shows benefits of regular exercise",
        "BREAKING: Time travel machine invented",
        "Research indicates climate change impact"
    ]
    
    for text in test_texts:
        result = trainer.predict(text)
        if result:
            prediction = "FAKE" if result['is_fake'] else "REAL"
            print(f"Text: {text[:50]}...")
            print(f"Prediction: {prediction}")
            print(f"Confidence: Fake {result['confidence']['fake']}%, Real {result['confidence']['real']}%")
            print("-" * 40)

if __name__ == "__main__":
    main()
