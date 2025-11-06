import pandas as pd
import spacy

# Load the spaCy English model.
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("Spacy model 'en_core_web_sm' not found.")
    print("Please run 'python -m spacy download en_core_web_sm' to install it.")
    exit()

# Load the cleaned dataset
file_path = 'Dataset/Cleaned_Airline_Reviews.csv'
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
    exit()


# --- Aspect Extraction Function ---
def extract_aspects(text : str):
    # Process the text with spaCy
    doc = nlp(text)
    
    # We will ignore pronouns like "I", "you", "we" as they are not aspects
    descriptive_noun_chunks = [
        chunk.text.lower() for chunk in doc.noun_chunks
        if chunk.root.pos_ != 'PRON'
    ]
    
    return descriptive_noun_chunks

# Apply the function to the 'Review' column to extract aspects
print("Starting aspect extraction")
df['Extracted_Aspects'] = df['Review'].apply(extract_aspects)

# Save the results to a new CSV file
output_file_path = 'Dataset/Aspect_Extraction_Results.csv'
df.to_csv(output_file_path, index=False)

print(f"\nSuccessfully extracted aspects and saved the results to: {output_file_path}")
print("Preview:")
# Displaying relevant columns for clarity
print(df[['AirLine_Name', 'Review', 'Extracted_Aspects']].head())
