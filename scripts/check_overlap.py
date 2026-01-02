import pandas as pd
import glob
from tqdm import tqdm

def get_all_metadata_asins(meta_dir):
    meta_asins = set()
    files = glob.glob(f"{meta_dir}/*.parquet")
    
    for file in tqdm(files, desc="Extracting Metadata ASINs"):
        df = pd.read_parquet(file, columns=['parent_asin'])
        meta_asins.update(df['parent_asin'].unique())
        
    return meta_asins


def check_review_overlap(review_dir, metadata_set):
    files = glob.glob(f"{review_dir}/*.parquet")
    total_reviews = 0
    matched_reviews = 0
    
    for file in tqdm(files, desc="Checking Review Overlap"):
        df = pd.read_parquet(file, columns=['parent_asin'])
        total_reviews += len(df)
        
        # Check how many ASINs in this chunk are in our metadata set
        is_matched = df['parent_asin'].isin(metadata_set)
        matched_reviews += is_matched.sum()
        
    overlap_pct = (matched_reviews / total_reviews) * 100
    print(f"\nFinal Results:")
    print(f"Total Reviews Scanned: {total_reviews}")
    print(f"Reviews with Metadata: {matched_reviews}")
    print(f"Overlap Percentage: {overlap_pct:.2f}%")



# Run this first
metadata_asin_set = get_all_metadata_asins('data/processed/transformed_chunks')
print(f"Total unique products in Metadata: {len(metadata_asin_set)}")


# Run the check
check_review_overlap('data/processed/review_chunks', metadata_asin_set)