import pandas as pd
import json
import gzip
import os
from tqdm import tqdm

def load_amazon_in_chunks(file_path, output_dir, chunk_size=100000):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Opening {file_path}...")
    
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        chunk = []
        chunk_count = 0
        
        for i, line in enumerate(tqdm(f, desc="Processing Lines")):
            try:
                chunk.append(json.loads(line))
            except json.JSONDecodeError:
                continue

            if len(chunk) >= chunk_size:
                process_and_save(chunk, chunk_count, output_dir)
                chunk = []
                chunk_count += 1

        if chunk:
            process_and_save(chunk, chunk_count, output_dir)

def process_and_save(chunk, count, output_dir):
    df = pd.DataFrame(chunk)
    
    cols_to_keep = ['parent_asin', 'main_category', 'title', 'average_rating', 'rating_number', 'price']
    df = df[[c for c in cols_to_keep if c in df.columns]]

    # Convert price to numeric, turning errors (like '—') into NaN
    if 'price' in df.columns:
        df['price'] = pd.to_numeric(df['price'], errors='coerce')

    output_file = os.path.join(output_dir, f"chunk_{count}.parquet")
    df.to_parquet(output_file, engine='pyarrow', index=False)

if __name__ == "__main__":
    INPUT_FILE = 'data/raw/meta_Electronics.jsonl.gz'
    OUTPUT_FOLDER = 'data/processed/electronics_chunks/'
    load_amazon_in_chunks(INPUT_FILE, OUTPUT_FOLDER)