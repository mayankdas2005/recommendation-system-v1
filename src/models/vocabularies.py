import pandas as pd
import numpy as np 
import glob


def build_vocab(gold_data_path):
    user_ids = set()
    item_ids = set()
    categories = set()

    files = glob.glob(f"{gold_data_path}/*.parquet")

    for file in files:
        df = pd.read_parquet(file)
        user_ids.update(df['user_id'].unique().astype('str'))
        item_ids.update(df['parent_asin'].unique().astype('str'))
        categories.update(df['main_category'].unique().astype('str'))

    return list(user_ids), list(item_ids), list(categories)

INPUT_PATH = "C:/Users/Dell/Desktop/recommendation-engine/data/processed/gold_training_set"

user_ids, item_ids, categories = build_vocab(INPUT_PATH)

print(f"Number of users: {len(user_ids)}")
print(f"Number of items: {len(item_ids)}")
print(f"Number of categories: {len(categories)}")

