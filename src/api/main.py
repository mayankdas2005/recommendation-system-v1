from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import tensorflow as tf
import uvicorn


from scripts.final_recommend import (
    MODEL, 
    ITEM_IDS, 
    resolver, 
    history_provider
)
from src.models.llm_explainer import generate_personalized_email

app = FastAPI(
    title="Amazon Recommendation API",
    description="Production-grade Two-Tower Recommender with Gemini AI Personalization",
    version="1.0.0"
)


class Recommendation(BaseModel):
    asin: str
    title: str
    category: str
    rating: float

class RecResponse(BaseModel):
    user_id: str
    recommendations: List[Recommendation]
    ai_pitch: str



@app.get("/")
def health_check():
    """Verify the API and Model state."""
    return {
        "status": "online",
        "model_loaded": MODEL is not None,
        "items_in_index": len(ITEM_IDS)
    }

@app.get("/recommend/{user_id}", response_model=RecResponse)
async def get_rec(user_id: str, k: int = 5):
    try:
        
        seen_items = history_provider.get_seen(user_id)
        
        user_embedding = MODEL.user_model(tf.constant([str(user_id)]))
        item_embeddings = MODEL.item_model(tf.constant(ITEM_IDS))
        
        scores = tf.matmul(user_embedding, item_embeddings, transpose_b=True)
        
        _, indices = tf.math.top_k(scores, k=50)
        
        final_recs = []
        for idx in indices.numpy()[0]:
            asin = ITEM_IDS[idx]
            if asin not in seen_items:
                details = resolver.get_details(asin)
                if details:
                    final_recs.append({
                        "asin": asin,
                        "title": details['title'],
                        "category": details['category'],
                        "rating": float(details['rating'])
                    })
            if len(final_recs) == k:
                break
        
        ai_pitch = generate_personalized_email(user_id, final_recs)
        
        return {
            "user_id": user_id,
            "recommendations": final_recs,
            "ai_pitch": ai_pitch
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)