import tensorflow as tf
import tensorflow_recommenders as tfrs

class AmazonModel(tfrs.Model):
    def __init__(self, user_ids, item_ids):
        super().__init__()
        
        self.user_model = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=user_ids, mask_token=None),
            tf.keras.layers.Embedding(len(user_ids) + 1, 32) 
        ])
        
        
        self.item_model = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=item_ids, mask_token=None),
            tf.keras.layers.Embedding(len(item_ids) + 1, 32)
        ])
        
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=tf.data.Dataset.from_tensor_slices(item_ids).batch(128).map(self.item_model)
            )
        )

    def compute_loss(self, features, training=False):
        user_embeddings = self.user_model(features["user_id"])
        item_embeddings = self.item_model(features["parent_asin"])
        
        return self.task(user_embeddings, item_embeddings)