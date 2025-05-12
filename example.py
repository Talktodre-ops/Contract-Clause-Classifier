"""
Example usage of the contract clause classifier
"""

from classify_clauses import load_model, classify_clause

def main():
    # Load the model
    print("Loading model...")
    try:
        vectorizer, classifier, results = load_model("models")
        print(f"Model loaded successfully. Accuracy: {results['accuracy']:.4f}")
    except FileNotFoundError:
        print("Error: Model files not found. Please train the model first.")
        print("Run: python train_classifier.py --data data/contract_clauses.csv --output models")
        return
    
    # Test with some examples
    test_clauses = [
        "Tenant shall be responsible for payment of all electricity bills.",
        "Upon termination of the lease, the security deposit will be returned within 30 days.",
        "Tenant shall have the option to renew this lease for an additional term."
    ]
    
    print("\nClassifying example clauses:")
    for clause in test_clauses:
        result = classify_clause(clause, vectorizer, classifier)
        print(f"\nText: {clause}")
        print(f"Prediction: {result['prediction']}")

    # Interactive mode
    print("\n" + "-"*50)
    print("Interactive mode: Enter a contract clause to classify (or 'quit' to exit)")
    while True:
        user_input = input("\nEnter clause: ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
            
        if not user_input.strip():
            continue
            
        result = classify_clause(user_input, vectorizer, classifier)
        print(f"Prediction: {result['prediction']}")
        

if __name__ == "__main__":
    main()
