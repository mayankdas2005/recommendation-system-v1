import pandas as pd
import numpy as np
import glob
import os
from tqdm import tqdm

def transform_chunk(df):
    """
    Applies consistent transformations to a single dataframe.
    """
    #Price Log Transformation
    if 'price' in df.columns:
        # We fill NaNs with the median price of the dataset (calculated during EDA)
        df['price'] = df['price'].fillna(df['price'].median())
        df['log_price'] = np.log1p(df['price'])

    # Category Cleaning
    if 'main_category' in df.columns:
        df['main_category'] = df['main_category'].fillna('Unknown')
        
    # Handle Rating Number
    if 'rating_number' in df.columns:
        df['rating_number'] = df['rating_number'].fillna(0)

    return df

def process_all_chunks(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    files = glob.glob(f"{input_dir}/*.parquet")
    
    for file in tqdm(files, desc="Transforming chunks"):
        df = pd.read_parquet(file)
        df_transformed = transform_chunk(df)
        
        file_name = os.path.basename(file)
        df_transformed.to_parquet(os.path.join(output_dir, file_name))

if __name__ == "__main__":
    process_all_chunks('data/processed/electronics_chunks', 'data/processed/transformed_chunks')