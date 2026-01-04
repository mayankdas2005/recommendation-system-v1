import tensorflow as tf
import pandas as pd # Added missing import
import glob # Added missing import
from src.models.recommender_v1 import AmazonModel
from src.models.vocabularies import build_vocab # Corrected function name
from src.data_pipeline.resolver import MetadataResolver

# 1. Initialize Tools
print("Initializing Recommendation Engine...")
user_ids, item_ids, _ = build_vocab('data/processed/gold_training_set')
resolver = MetadataResolver('data/processed/transformed_chunks')

# 2. Load the Brain
model = AmazonModel(user_ids, item_ids)
# Force build weights by passing dummy data
_ = model({"user_id": tf.constant([user_ids[0]]), "parent_asin": tf.constant([item_ids[0]])})
model.load_weights('src/models/weights/two_tower_v1.weights.h5')

# 3. History Filter Logic
class HistoryProvider:
    def __init__(self, training_dir):
        print("Indexing user history for filtering (This may take a minute)...")
        files = glob.glob(f"{training_dir}/*.parquet")
        dfs = [pd.read_parquet(f, columns=['user_id', 'parent_asin']) for f in files]
        full_df = pd.concat(dfs)
        self.history = full_df.groupby('user_id')['parent_asin'].apply(set).to_dict()
        print(f"✅ History indexed for {len(self.history)} users.")

    def get_seen(self, user_id):
        return self.history.get(user_id, set())

history_provider = HistoryProvider('data/processed/gold_training_set')

# 4. Recommendation Engine
def get_recommendations(user_id, k=5):
    seen_items = history_provider.get_seen(user_id)
    
    # Generate User Embedding
    user_embedding = model.user_model(tf.constant([str(user_id)]))
    
    # Calculate Similarity Scores (Dot Product)
    # Note: For production with 1M+ items, you'd use a Vector DB like Scann
    item_embeddings = model.item_model(tf.constant(item_ids))
    scores = tf.matmul(user_embedding, item_embeddings, transpose_b=True)
    
    # Get top 50 candidates to allow for filtering
    _, indices = tf.math.top_k(scores, k=50)
    
    final_asins = []
    for idx in indices.numpy()[0]:
        asin = item_ids[idx]
        if asin not in seen_items:
            final_asins.append(asin)
        if len(final_asins) == k:
            break
            
    # DISPLAY RESULTS
    print(f"\n🚀 Recommendations for User: {user_id}")
    print("="*60)
    for asin in final_asins:
        details = resolver.get_details(asin)
        if details:
            print(f"⭐ {details['title']}")
            print(f"   Category: {details['category']} | Rating: {details['rating']}")
            print("-" * 60)

if __name__ == "__main__":
    test_user_id = "AFKZENTNBQ7A7V7UXW5JJI6UGRYQ" 
    get_recommendations(test_user_id)