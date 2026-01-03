import tensorflow as tf
import tensorflow_recommenders as tfrs
import keras

class AmazonModel(tfrs.Model):
    def __init__(self, user_ids, item_ids):
        super().__init__()
        
        # 1. THE USER TOWER
        self.user_model = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=user_ids, mask_token=None),
            tf.keras.layers.Embedding(len(user_ids) + 1, 32)
        ])
        
        # 2. THE ITEM TOWER
        self.item_model = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=item_ids, mask_token=None),
            tf.keras.layers.Embedding(len(item_ids) + 1, 32)
        ])
        
        # 3. THE TASK (Modified for Keras 3 Compatibility)
        # We remove the metric for now to get training started
        self.task = tfrs.tasks.Retrieval()

    def compute_loss(self, features, training=False):
        # We need to handle the input mapping
        user_embeddings = self.user_model(features["user_id"])
        item_embeddings = self.item_model(features["parent_asin"])
        
        # Retrieval task computes the loss (Categorical Cross-Entropy by default)
        return self.task(user_embeddings, item_embeddings)