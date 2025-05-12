"""
Contract Clause Data Generator
Generates synthetic contract clauses using Ollama/Mistral and saves to CSV
"""

import subprocess
import pandas as pd
import time
import argparse
import os

def generate_example(category, model="mistral:instruct", timeout=60):
    """Generate a single example for a category"""
    prompt = f"Write one example of a contract clause about {category}. Keep it to 1-2 sentences."
    
    try:
        # Use UTF-8 encoding explicitly
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True,
            text=True,
            encoding='utf-8',  # Explicitly set encoding
            errors='replace',  # Replace invalid characters
            timeout=timeout
        )
        return result.stdout.strip()
    except subprocess.SubprocessError as e:
        print(f"  Error with subprocess: {e}")
        return None
    except UnicodeError as e:
        print(f"  Unicode error: {e}")
        return None
    except Exception as e:
        print(f"  Error generating example: {e}")
        return None

def generate_dataset(categories, samples_per_category=5, model="mistral:instruct", output_file="contract_clauses.csv"):
    """Generate a dataset of contract clauses and save to CSV"""
    data = []
    
    for category in categories:
        print(f"Generating samples for category: {category}...")
        
        for i in range(samples_per_category):
            print(f"  Sample {i+1}/{samples_per_category}...")
            try:
                example = generate_example(category, model)
                
                if example:
                    print(f"  Got: {example[:50]}...")
                    data.append({
                        "text": example,
                        "category": category,
                        "source": "generated",
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    })
                else:
                    print("  Failed to generate example, skipping...")
                
                # Add a small delay between requests
                time.sleep(1)
            except KeyboardInterrupt:
                print("\nInterrupted by user. Saving current data...")
                break
            except Exception as e:
                print(f"  Unexpected error: {e}")
    
    # Create DataFrame and save to CSV
    if data:
        df = pd.DataFrame(data)
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"Generated {len(df)} samples across {len(set(item['category'] for item in data))} categories")
        print(f"Data saved to {output_file}")
        return df
    else:
        print("No data generated.")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic contract clauses")
    parser.add_argument("--categories", nargs="+", default=["Electricity", "Security Deposit", "Renewal Rights"],
                        help="Categories to generate clauses for")
    parser.add_argument("--samples", type=int, default=5, help="Number of samples per category")
    parser.add_argument("--model", default="mistral:instruct", help="Ollama model to use")
    parser.add_argument("--output", default="contract_clauses.csv", help="Output CSV file")
    
    args = parser.parse_args()
    
    try:
        generate_dataset(args.categories, args.samples, args.model, args.output)
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
