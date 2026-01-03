import pandas as pd
import glob
import tensorflow as tf
from src.models.recommender_v1 import AmazonModel
from src.models.vocabularies import build_vocab

def get_dataset_from_chunks(chunk_dir):
    """
    Generator that reads Parquet chunks and converts them to TF Tensors.
    """
    files = glob.glob(f"{chunk_dir}/*.parquet")
    for file in files:
        df = pd.read_parquet(file)
        inputs = {
            "user_id": df["user_id"].values.astype(str),
            "parent_asin": df["parent_asin"].values.astype(str)
        }
        yield tf.data.Dataset.from_tensor_slices(inputs)

def run_training():
    print("Building vocabularies...")
    user_ids, item_ids, _ = build_vocabs('data/processed/gold_training_set')

    model = AmazonModel(user_ids, item_ids)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

    for epoch in range(3): 
        print(f"\n--- Epoch {epoch+1} ---")
        
        chunk_gen = get_dataset_from_chunks('data/processed/gold_training_set')
        
        for i, chunk_ds in enumerate(chunk_gen):
            cached_ds = chunk_ds.batch(4096).cache()
            
            print(f"Training on Gold Chunk {i}...")
            model.fit(cached_ds, epochs=1, verbose=1)
    model.save_weights('src/models/weights/two_tower_v1')
    print("Model weights saved successfully!")

if __name__ == "__main__":
    run_training()