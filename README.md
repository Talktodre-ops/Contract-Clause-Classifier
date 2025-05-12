# Contract Clause Classifier

Prototype NLP system that classifies commercial lease clauses into predefined categories using lightweight ML models and synthetic data generation.

A simple and fast text classification system using TF-IDF and Logistic Regression. Built for local execution and reproducibility, this tool supports synthetic data generation using Mistral LLM via Ollama.

## Features

- Generate synthetic contract clauses using Mistral LLM locally (Ollama)
- Train lightweight classifiers (TF-IDF + Logistic Regression)
- Evaluate performance with confusion matrix and metrics
- Inference time <150ms per clause
- Fully local, no OpenAI or external APIs



# Contract Clause Classifier

A simple text classification system for categorizing contract clauses using TF-IDF and Logistic Regression.

## Features

- Generate synthetic contract clauses using Mistral LLM via Ollama
- Train a TF-IDF + Logistic Regression classifier
- Evaluate model performance with confusion matrix and accuracy metrics
- Classify new contract clauses

## Requirements

- Python 3.11.8
- Ollama with mistral:instruct model installed
- Required Python packages: pandas, numpy, scikit-learn, matplotlib, seaborn

## Usage

### 1. Generate synthetic data

```bash
python generate_contract_data.py --samples 10 --output data/contract_clauses.csv
```

Options:
- `--categories`: List of categories (default: "Electricity", "Security Deposit", "Renewal Rights")
- `--samples`: Number of samples per category (default: 5)
- `--model`: Ollama model to use (default: mistral:instruct)
- `--output`: Output CSV file (default: contract_clauses.csv)

### 2. Train the classifier

```bash
python train_classifier.py --data data/contract_clauses.csv --output models
```

Options:
- `--data`: Input CSV file (default: contract_clauses.csv)
- `--output`: Output directory for model files (default: models)
- `--test-size`: Test set size proportion (default: 0.3)

### 3. Classify new clauses

```bash
python classify_clauses.py --input new_clauses.csv --output classifications.csv
```

Options:
- `--input`: Input file (CSV or text)
- `--output`: Output CSV file
- `--model-dir`: Directory containing model files (default: models)

## Model result 

C:\Users\VICTUS\Documents\GITHUB\MockUp>python train_classifier.py
Loading data from contract_clauses.csv...
Loaded 15 samples across 3 categories
Split data into 10 training and 5 testing samples
Vectorizing text...
Training classifier...
Evaluating model...
Accuracy: 1.0000

Classification Report:
                  precision    recall  f1-score   support

     Electricity       1.00      1.00      1.00         1
  Renewal Rights       1.00      1.00      1.00         2
Security Deposit       1.00      1.00      1.00         2

        accuracy                           1.00         5
       macro avg       1.00      1.00      1.00         5
    weighted avg       1.00      1.00      1.00         5

Saving model to models...
Model and results saved to models

## Python Inference Example

```python
from classify_clauses import load_model, classify_clause

vectorizer, classifier, results = load_model("models")

clause = "Tenant shall be responsible for payment of all electricity bills."
result = classify_clause(clause, vectorizer, classifier)
print(f"Prediction: {result['prediction']}")
```
Prediction: Electricity:The pridiction is in the  classifications.csv


## License

MIT