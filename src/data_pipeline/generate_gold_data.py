import pandas as pd
import glob
import os
from tqdm import tqdm

def create_gold_dataset(review_dir, meta_dir, output_dir, min_ratings=5):
    os.makedirs(output_dir, exist_ok=True)
    
    # Load metadata in RAM (since only approx 200MB)
    print("Building Metadata Lookup...")
    meta_files = glob.glob(f"{meta_dir}/*.parquet")
    meta_df = pd.concat([pd.read_parquet(f) for f in meta_files])
    
    # Keep only unique ASINS and remove duplicates
    meta_df = meta_df.drop_duplicates(subset=['parent_asin']).set_index('parent_asin')

    # Process review chunks
    review_files = glob.glob(f"{review_dir}/*.parquet")
    
    for i, file in enumerate(tqdm(review_files, desc="Processing Gold Chunks")):
        df = pd.read_parquet(file)
        
        # Convert from milliseconds to Datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Keep only users who have given more than 5 ratings in total
        user_counts = df['user_id'].value_counts()
        power_users = user_counts[user_counts >= min_ratings].index
        df = df[df['user_id'].isin(power_users)]
        
        # Attach item features (title, category, price) to the review
        gold_chunk = df.join(meta_df, on='parent_asin', how='inner')


        # Save the finalized chunk
        output_file = os.path.join(output_dir, f"gold_chunk_{i}.parquet")
        gold_chunk.to_parquet(output_file, index=False)

if __name__ == "__main__":
    create_gold_dataset(
        'data/processed/review_chunks', 
        'data/processed/transformed_chunks', 
        'data/processed/gold_training_set'
    )