import kagglehub
import pandas as pd 
import os

def get_kaggle_data(kaggle_string = "ankurzing/sentiment-analysis-for-financial-news"):
    path = kagglehub.dataset_download(kaggle_string)
    files = os.listdir(path)
    csv_file = files[1]
    csv_path = os.path.join(path, csv_file)

    df = pd.read_csv(csv_path, 
                        names=['sentiment', 'sentence'],
                        encoding='latin-1')
    print("dataframe arranged successfully")
    
    return df
