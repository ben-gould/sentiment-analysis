import pandas as pd
from sklearn.model_selection import train_test_split
import re
import os
import kagglehub


def download_phrasebank(kaggle_string = "ankurzing/sentiment-analysis-for-financial-news"):
    path = kagglehub.dataset_download(kaggle_string)
    files = os.listdir(path)
    csv_file = files[1]
    csv_path = os.path.join(path, csv_file)

    return csv_path

def clean_text(text):
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def load_and_split_data(csv_path):
    df = pd.read_csv(csv_path, 
                        names=['sentiment', 'sentence'],
                        encoding='latin-1')
    df['text_clean'] = df['sentence'].apply(clean_text)

    train_val, test = train_test_split(
        df, 
        test_size=0.15, 
        stratify=df['sentiment'],
        random_state=42
    )
    
    # Second split: train and validation
    train, val = train_test_split(
        train_val,
        test_size=.15/.85, 
        stratify=train_val['sentiment'],
        random_state=42
    )

    return train, val, test

if __name__ == "__main__":
    csv_path = download_phrasebank()
    train, val, test = load_and_split_data(csv_path)
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")