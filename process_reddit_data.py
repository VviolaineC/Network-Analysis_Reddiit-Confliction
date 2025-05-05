import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Define column names
columns = [
    'SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT', 'POST_ID', 'TIMESTAMP', 
    'POST_LABEL', 'POST_PROPERTIES'
]

# Load data
def load_data(file_path):
    df = pd.read_csv(file_path, sep='\t', names=columns, low_memory=False)
    return df

# Process POST_PROPERTIES column
def process_post_properties(df):
    # Split POST_PROPERTIES into separate columns
    properties = df['POST_PROPERTIES'].str.split(',', expand=True)
    properties = properties.apply(pd.to_numeric, errors='coerce')
    
    # Create meaningful column names for each property
    property_names = [
        'num_chars', 'num_chars_no_space', 'frac_alpha', 'frac_digits',
        'frac_uppercase', 'frac_whitespace', 'frac_special', 'num_words',
        'num_unique_words', 'num_long_words', 'avg_word_length',
        'num_stopwords', 'frac_stopwords', 'num_sentences',
        'num_long_sentences', 'avg_chars_per_sentence',
        'avg_words_per_sentence', 'readability_index',
        'sentiment_positive', 'sentiment_negative', 'sentiment_compound'
    ]
    
    # Add LIWC feature names
    liwc_features = [
        'LIWC_Funct', 'LIWC_Pronoun', 'LIWC_Ppron', 'LIWC_I', 'LIWC_We',
        'LIWC_You', 'LIWC_SheHe', 'LIWC_They', 'LIWC_Ipron', 'LIWC_Article',
        'LIWC_Verbs', 'LIWC_AuxVb', 'LIWC_Past', 'LIWC_Present', 'LIWC_Future',
        'LIWC_Adverbs', 'LIWC_Prep', 'LIWC_Conj', 'LIWC_Negate', 'LIWC_Quant',
        'LIWC_Numbers', 'LIWC_Swear', 'LIWC_Social', 'LIWC_Family',
        'LIWC_Friends', 'LIWC_Humans', 'LIWC_Affect', 'LIWC_Posemo',
        'LIWC_Negemo', 'LIWC_Anx', 'LIWC_Anger', 'LIWC_Sad', 'LIWC_CogMech',
        'LIWC_Insight', 'LIWC_Cause', 'LIWC_Discrep', 'LIWC_Tentat',
        'LIWC_Certain', 'LIWC_Inhib', 'LIWC_Incl', 'LIWC_Excl', 'LIWC_Percept',
        'LIWC_See', 'LIWC_Hear', 'LIWC_Feel', 'LIWC_Bio', 'LIWC_Body',
        'LIWC_Health', 'LIWC_Sexual', 'LIWC_Ingest', 'LIWC_Relativ',
        'LIWC_Motion', 'LIWC_Space', 'LIWC_Time', 'LIWC_Work', 'LIWC_Achiev',
        'LIWC_Leisure', 'LIWC_Home', 'LIWC_Money', 'LIWC_Relig', 'LIWC_Death',
        'LIWC_Assent', 'LIWC_Dissent', 'LIWC_Nonflu', 'LIWC_Filler'
    ]
    
    property_names.extend(liwc_features)
    
    # Ensure we have enough column names
    if len(property_names) != properties.shape[1]:
        print(f"Warning: Number of property names ({len(property_names)}) does not match number of columns ({properties.shape[1]})")
    
    # Rename columns
    properties.columns = property_names[:properties.shape[1]]
    
    # Merge back with original dataframe
    df = pd.concat([df.drop('POST_PROPERTIES', axis=1), properties], axis=1)
    
    return df

# Basic data analysis
def analyze_data(df, source_name):
    print(f"\nAnalyzing {source_name} dataset:")
    print(f"Total rows: {len(df)}")
    print(f"Number of subreddits: {df['SOURCE_SUBREDDIT'].nunique()}")
    print(f"Number of target subreddits: {df['TARGET_SUBREDDIT'].nunique()}")
    
    # Clean label data
    df['POST_LABEL'] = pd.to_numeric(df['POST_LABEL'], errors='coerce')
    df = df.dropna(subset=['POST_LABEL'])
    df['POST_LABEL'] = df['POST_LABEL'].astype(int)
    
    print("\nLabel distribution:")
    print(df['POST_LABEL'].value_counts())
    
    # Time range
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], errors='coerce')
    df = df.dropna(subset=['TIMESTAMP'])
    print(f"\nTime range: {df['TIMESTAMP'].min()} to {df['TIMESTAMP'].max()}")

def main():
    # Load data
    body_df = load_data("soc-redditHyperlinks-body.tsv")
    title_df = load_data("soc-redditHyperlinks-title.tsv")
    
    # Process data
    body_df = process_post_properties(body_df)
    title_df = process_post_properties(title_df)
    
    # Add data source identifier
    body_df['SOURCE_TYPE'] = 'body'
    title_df['SOURCE_TYPE'] = 'title'
    
    # Merge datasets
    combined_df = pd.concat([body_df, title_df], ignore_index=True)
    
    # Analyze data
    analyze_data(body_df, "Body")
    analyze_data(title_df, "Title")
    analyze_data(combined_df, "Combined")
    
    # Save processed data
    combined_df.to_csv('processed_reddit_data.csv', index=False)
    print("\nProcessed data saved to 'processed_reddit_data.csv'")

if __name__ == "__main__":
    main() 
