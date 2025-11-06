import pandas as pd
import torch
from transformers import pipeline
from tqdm import tqdm
import re
import string
from sklearn.model_selection import train_test_split

def clean_text(text):
    """
    A simple text cleaning function to remove noise from reviews.
    """
    text = str(text)
    text = text.lower()
    # Remove "Trip Verified" prefixes
    text = re.sub(r'✅\s*trip verified\s*\|', '', text)
    text = re.sub(r'trip verified\s*\|', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove extra whitespace
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def analyze_airline_sentiment_on_test_set():
    """
    Complete script with preprocessing and train/test split.
    """
    
    # --- Step 1: Load Data ---
    try:
        df = pd.read_csv("Dataset/Full_Indian_Domestic_Airline.csv")
    except FileNotFoundError:
        print("Error: 'Full_Indian_Domestic_Airline.csv' not found.")
        return

    df = df[['AirLine_Name', 'Rating - 10', 'Review']].copy()
    
    # --- Step 2: Data Preprocessing and Cleaning ---
    
    print("Starting data preprocessing...")
    df.dropna(subset=['Review'], inplace=True)
    initial_rows = len(df)
    df.drop_duplicates(subset=['Review'], inplace=True)
    print(f"Removed {initial_rows - len(df)} duplicate reviews.")

    tqdm.pandas(desc="Cleaning review text")
    df['cleaned_review'] = df['Review'].progress_apply(clean_text)
    df = df[df['cleaned_review'].str.strip().astype(bool)]
    print(f"Preprocessing complete. {len(df)} unique, non-empty reviews remaining.")

    # --- Step 3: Train/Test Split ---
    
    print("Splitting data into training and test sets (80/20 split)...")
    
    # Data is split here into train_df and test_df
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    print(f"Total samples: {len(df)}")
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")

    # --- Step 4: Define Aspects and Initialize Model ---
    
    aspects = [
        "seat comfort", 
        "crew service", 
        "food and beverage", 
        "check-in and boarding", 
        "value for money",
        "in-flight entertainment"
    ]
    sentiment_labels = ["positive", "negative", "neutral"]
    
    # --- Hardware Detection Logic ---
    device_to_use = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing model on device: {device_to_use}")
    
    # device=-1 is the legacy way to force CPU, 
    # but passing device="cpu" is the modern, clear way.
    classifier = pipeline(
        "zero-shot-classification", 
        model="facebook/bart-large-mnli",
        device=device_to_use
    )

    # --- Step 5: Process Reviews (Test Set Only) ---
    
    results = []
    print(f"Starting sentiment analysis process on {len(test_df)} test set reviews...")
    
    for index, row in tqdm(test_df.iterrows(), total=test_df.shape[0], desc="Analyzing Test Set"):
        
        review_to_analyze = row['cleaned_review']
        
        result_row = {
            "airline": row['AirLine_Name'],
            "original_rating": row['Rating - 10'],
            "original_review": row['Review'],
            "cleaned_review": review_to_analyze
        }
        
        for aspect in aspects:
            template = f"The sentiment about {aspect} is {{}}."
            try:
                output = classifier(
                    review_to_analyze,
                    candidate_labels=sentiment_labels,
                    hypothesis_template=template,
                    multi_label=False
                )
                
                best_sentiment = output['labels'][0]
                confidence_score = output['scores'][0]
                
                result_row[f"{aspect}_sentiment"] = best_sentiment
                result_row[f"{aspect}_score"] = round(confidence_score, 4)
                
            except Exception as e:
                # Handle any unexpected errors during classification
                print(f"\nError processing row {index} for aspect '{aspect}': {e}")
                result_row[f"{aspect}_sentiment"] = "Error"
                result_row[f"{aspect}_score"] = 0.0

        results.append(result_row)

    # --- Step 6: Save Results ---
    
    print("\nAnalysis complete. Saving test set results to CSV...")
    
    results_df = pd.DataFrame(results)
    
    base_cols = ["airline", "original_rating", "original_review", "cleaned_review"]
    aspect_cols = []
    for aspect in aspects:
        aspect_cols.append(f"{aspect}_sentiment")
        aspect_cols.append(f"{aspect}_score")
        
    final_cols = base_cols + aspect_cols
    results_df = results_df[final_cols]
    
    output_filename = "Output/absa_airline_results_test_set.csv"
    results_df.to_csv(output_filename, index=False)
    
    print(f"Successfully saved test set results to {output_filename}")


if __name__ == "__main__":
    analyze_airline_sentiment_on_test_set()
