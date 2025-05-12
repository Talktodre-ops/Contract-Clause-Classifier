"""
Contract Clause Classifier
Trains a TF-IDF + Logistic Regression classifier on contract clauses
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import pickle
import os
import argparse

def train_classifier(data_file, output_dir="models", test_size=0.3, random_state=42):
    """Train a contract clause classifier and save the model"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from {data_file}...")
    df = pd.read_csv(data_file)
    print(f"Loaded {len(df)} samples across {df['category'].nunique()} categories")
    
    # Prepare data for classification
    X = df['text']
    y = df['category']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print(f"Split data into {len(X_train)} training and {len(X_test)} testing samples")
    
    # Convert text to TF-IDF features
    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train classifier
    print("Training classifier...")
    classifier = LogisticRegression(max_iter=1000, random_state=random_state)
    classifier.fit(X_train_tfidf, y_train)
    
    # Evaluate the model
    print("Evaluating model...")
    y_pred = classifier.predict(X_test_tfidf)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(class_report)
    
    # Save results
    categories = sorted(df['category'].unique())
    results = {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'categories': categories
    }
    
    # Save model and vectorizer
    print(f"Saving model to {output_dir}...")
    with open(os.path.join(output_dir, 'vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
    
    with open(os.path.join(output_dir, 'classifier.pkl'), 'wb') as f:
        pickle.dump(classifier, f)
    
    with open(os.path.join(output_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    # Generate confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=categories, yticklabels=categories)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    
    print(f"Model and results saved to {output_dir}")
    return classifier, vectorizer, results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a contract clause classifier")
    parser.add_argument("--data", default="contract_clauses.csv", help="Input CSV file")
    parser.add_argument("--output", default="models", help="Output directory for model files")
    parser.add_argument("--test-size", type=float, default=0.3, help="Test set size (proportion)")
    
    args = parser.parse_args()
    
    train_classifier(args.data, args.output, args.test_size)