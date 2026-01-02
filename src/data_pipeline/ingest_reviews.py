import pandas as pd 
import os
import json
import gzip
from tqdm import tqdm


def ingest_reviews(file_path, save_path, chunk_size=200000):
    os.makedirs(save_path, exist_ok=True)

    cols_to_keep = ['user_id', 'parent_asin', 'rating', 'timestamp']

    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        chunk = []
        chunk_count = 0

        for i, line in enumerate(tqdm(f, desc="Processing Reviews")):
            raw_data = json.loads(line)
            filtered_data = {k: raw_data[k] for k in cols_to_keep if k in raw_data}
            chunk.append(filtered_data)

            if len(chunk) >= chunk_size:
                df = pd.DataFrame(chunk)
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

                output_file = os.path.join(save_path, f"review_chunk_{chunk_count}.parquet")
                df.to_parquet(output_file, index=False)
                
                chunk = []
                chunk_count += 1
        if chunk:
            pd.DataFrame(chunk).to_parquet(os.path.join(save_path, f"review_chunk_{chunk_count}.parquet"))
                
if __name__ == "__main__":
    ingest_reviews('data/raw/Electronics.jsonl.gz', 'data/processed/review_chunks')