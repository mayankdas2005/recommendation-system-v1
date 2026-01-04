import pandas as pd
import glob
import os

class MetadataResolver:
    def __init__(self, metadata_dir):
        print(f"Searching for metadata in: {os.path.abspath(metadata_dir)}")
        
        # Look for parquet files
        files = glob.glob(os.path.join(metadata_dir, "*.parquet"))
        
        if not files:
            raise FileNotFoundError(f"No .parquet files found in {metadata_dir}. Check your path!")
            
        print(f"Found {len(files)} chunks. Loading...")
        
        dfs = []
        for f in files:
            try:
                # Load only necessary columns to save RAM
                df = pd.read_parquet(f, columns=['parent_asin', 'title', 'main_category', 'average_rating'])
                dfs.append(df)
            except Exception as e:
                print(f"Warning: Could not read {f}: {e}")

        if not dfs:
            raise ValueError("Files found, but none could be loaded into DataFrames.")

        self.meta_df = pd.concat(dfs).drop_duplicates(subset=['parent_asin']).set_index('parent_asin')
        print(f"Resolver ready with {len(self.meta_df)} unique items.")

    def get_details(self, asin):
        try:
            item = self.meta_df.loc[asin]
            return {
                "title": item['title'],
                "category": item['main_category'],
                "rating": item['average_rating']
            }
        except KeyError:
            return None


resolver = MetadataResolver("data/processed/transformed_chunks")
print(resolver.get_details('B00005LENO'))