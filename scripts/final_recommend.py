import tensorflow as tf
from src.models.recommender_v1 import AmazonModel
from src.models.vocabularies import build_vocab
from src.data_pipeline.resolver import MetadataResolver


print("Initializing Recommendation Engine...")
user_ids, item_ids, _ = build_vocab('data/processed/gold_training_set')
resolver = MetadataResolver('data/processed/transformed_chunks')

model = AmazonModel(user_ids, item_ids)

_ = model({"user_id": tf.constant([user_ids[0]]), "parent_asin": tf.constant([item_ids[0]])})
model.load_weights('src/models/weights/two_tower_v1.weights.h5')

def get_recommendations(user_id, k=5):
    user_embedding = model.user_model(tf.constant([str(user_id)]))
    
    item_embeddings = model.item_model(tf.constant(item_ids))
    scores = tf.matmul(user_embedding, item_embeddings, transpose_b=True)


    values, indices = tf.math.top_k(scores, k=k)
    
    print(f"\n Recommendations for User: {user_id}")
    print("="*50)
    for idx in indices.numpy()[0]:
        asin = item_ids[idx]
        details = resolver.get_details(asin)
        if details:
            print(f"{details['title']}")
            print(f"   Category: {details['category']} | Rating: {details['rating']}")
            print("-" * 50)

if __name__ == "__main__":
    # Use one of the power user found during EDA
    test_user_id = "AFKZENTNBQ7A7V7UXW5JJI6UGRYQ" 
    get_recommendations(test_user_id)