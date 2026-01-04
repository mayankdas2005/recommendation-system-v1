import tensorflow as tf
import tensorflow_recommenders as tfrs
import keras

class AmazonModel(tfrs.Model):
    def __init__(self, user_ids, item_ids):
        super().__init__()
        
        # User tower
        self.user_model = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=user_ids, mask_token=None),
            tf.keras.layers.Embedding(len(user_ids) + 1, 32)
        ])
        
        # Item towr
        self.item_model = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=item_ids, mask_token=None),
            tf.keras.layers.Embedding(len(item_ids) + 1, 32)
        ])
        
        # Keras 3 compatibility fix (long story short, it's required)
        self.task = tfrs.tasks.Retrieval()
    
    def call(self, features):
        # This tells Keras how data flows through the model
        return (
            self.user_model(features["user_id"]),
            self.item_model(features["parent_asin"]),
        )

    def compute_loss(self, features, training=False):
        # We need to handle the input mapping
        user_embeddings = self.user_model(features["user_id"])
        item_embeddings = self.item_model(features["parent_asin"])
        
        # Retrieval task computes the loss (categorical cross-entropy by default)
        return self.task(user_embeddings, item_embeddings)