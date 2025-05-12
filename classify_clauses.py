"""
Contract Clause Classifier - Inference
Classifies contract clauses using a trained model
"""

import pickle
import argparse
import pandas as pd
import os

def load_model(model_dir="models"):
    """Load the trained model and vectorizer"""
    with open(os.path.join(model_dir, 'vectorizer.pkl'), 'rb') as f:
        vectorizer = pickle.load(f)
    
    with open(os.path.join(model_dir, 'classifier.pkl'), 'rb') as f:
        classifier = pickle.load(f)
    
    with open(os.path.join(model_dir, 'results.pkl'), 'rb') as f:
        results = pickle.load(f)
    
    return vectorizer, classifier, results

def classify_clause(text, vectorizer, classifier):
    """Classify a contract clause"""
    text_tfidf = vectorizer.transform([text])
    prediction = classifier.predict(text_tfidf)[0]
    probabilities = classifier.predict_proba(text_tfidf)[0]
    prob_dict = {cat: prob for cat, prob in zip(classifier.classes_, probabilities)}
    
    return {
        "prediction": prediction,
        "confidence": prob_dict[prediction],
        "all_probabilities": prob_dict
    }

def classify_file(input_file, output_file=None, model_dir="models"):
    """Classify clauses from a CSV or text file"""
    vectorizer, classifier, results = load_model(model_dir)
    
    # Determine file type and load accordingly
    if input_file.endswith('.csv'):
        df = pd.read_csv(input_file)
        texts = df['text'].tolist()
    else:
        # Assume text file with one clause per line
        with open(input_file, 'r') as f:
            texts = [line.strip() for line in f if line.strip()]
    
    # Classify each text
    classifications = []
    for text in texts:
        result = classify_clause(text, vectorizer, classifier)
        classifications.append({
            "text": text,
            "prediction": result["prediction"]
            # Removed confidence from output
        })
    
    # Save results
    output_df = pd.DataFrame(classifications)
    if output_file:
        output_df.to_csv(output_file, index=False)
        print(f"Classifications saved to {output_file}")
    
    return output_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify contract clauses")
    parser.add_argument("--input", required=True, help="Input file (CSV or text)")
    parser.add_argument("--output", help="Output CSV file")
    parser.add_argument("--model-dir", default="models", help="Directory containing model files")
    
    args = parser.parse_args()
    
    classify_file(args.input, args.output, args.model_dir)
