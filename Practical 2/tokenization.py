import pandas as pd
import re
import spacy

# Load spaCy English model.
# May need to download it first : python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")
stopwords = nlp.Defaults.stop_words

# Load the Dataset
file_path = "Dataset/Cleaned_Airline_Reviews.csv"
df = pd.read_csv(file_path)

# Sentence Segmentation, Tokenization, and Stopword Removal
def process_text(text : str):
  sentences = re.split(r'(?<=[.!?]) +', text)
  tokens = [word for sentence in sentences for word in re.findall(r'\b\w+\b', sentence.lower())]

  useful_tokens = [token for token in tokens if token not in stopwords]

  return useful_tokens


df["Tokenized_Review"] = df["Review"].apply(process_text)

# Save the updated DataFrame to a new CSV
df.to_csv("Dataset/Tokenized_Dataset.csv", index=False)

print("Sucessfully Tokenized Dataset")
print("Preview:")
print(df.head())