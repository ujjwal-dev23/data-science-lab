import pandas as pd
import string

df = pd.read_csv('Dataset/Raw_Airline_Review_Combined.csv')

print("--- Initial DataFrame Info ---")
df.info()

print("\n--- First 5 Rows ---")
print(df.head())

# Check for and count duplicate rows
print(f"\nNumber of duplicate rows found: {df.duplicated().sum()}")

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Remove rows with any missing values
df.dropna(inplace=True)

print("Duplicates and rows with missing values have been removed.")


# Perform Text Cleaning

def clean_text_row(text : str) -> str:
  text = text.lower()
  # Remove prefix
  text = text.removeprefix("✅ trip verified |").removeprefix('trip verified |').removeprefix("❎ not verified |").removeprefix("not verified |").removeprefix("✅ verified review").strip()
  # Remove punctuation
  text = text.translate(str.maketrans('', '', string.punctuation + "“”"))
  return text

# Apply text cleaning to 'Review' and 'Title' columns
df["Review"] = df["Review"].apply(clean_text_row)
df["Title"] = df["Title"].apply(clean_text_row)

print("\n--- First 5 Rows After Text Cleaning ---")
print(df[['Title', 'Review']].head())

# Date Standardization and Conversion
df["Date"] = df["Date"].astype(str).str.strip()
df["Date"] = df["Date"].str.replace(r'(\d+)(st|nd|rd|th)', r'\1', regex=True)
df['Date'] = pd.to_datetime(df['Date'], format='%d %B %Y', errors='coerce')

# Convert 'Rating - 10' column to a numeric type
df['Rating - 10'] = pd.to_numeric(df['Rating - 10'])
# Convert 'Reccomend' column to binary (1 for 'yes', 0 for 'no')
df['Recommend'] = df['Recommend'].str.strip().str.lower().map({'yes': 1, 'no': 0})

print("\n--- Final DataFrame Info After Formatting ---")
df.info()

# Save the cleaned dataframe to a new CSV file
df.to_csv('Dataset/Cleaned_Airline_Reviews.csv', index=False)