import tensorflow as tf
from src.models.recommender_v1 import AmazonModel

# 1. Create dummy vocabularies (just to initialize the layers)
dummy_users = ["user_1", "user_2"]
dummy_items = ["item_1", "item_2"]

# 2. Initialize the model
print("Initializing model...")
model = AmazonModel(dummy_users, dummy_items)

# 3. THE FIX: Force the model to "Build" its weights
# We give it one fake interaction to create the internal matrices
dummy_input = {
    "user_id": tf.constant(["user_1"], dtype=tf.string),
    "parent_asin": tf.constant(["item_1"], dtype=tf.string)
}
_ = model(dummy_input) 

# 4. Try to save
try:
    model.save_weights('src/models/weights/test_save.weights.h5')
    print("✅ SUCCESS! The save logic is working.")
except Exception as e:
    print(f"❌ STILL FAILING: {e}")