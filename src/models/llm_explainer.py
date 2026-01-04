import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()


genai.configure(api_key=os.getenv("API_KEY"))

def generate_personalized_email(user_id, recs_details):
    model = genai.GenerativeModel('gemini-2.5-flash')

    items_list = ""
    for i, item in enumerate(recs_details[:3]):
        items_list += f"{i+1}. {item['title']} (Category: {item['category']})\n"
    prompt = f"""
    You are a helpful Amazon personal shopper. 
    A user with ID {user_id} has a history of buying high-end electronics and photography gear.
    
    We are recommending these 3 items to them:
    {items_list}
    
    Write a 3-sentence, professional but exciting email snippet.
    - Sentence 1: Acknowledge their interest in pro-level tech.
    - Sentence 2: Explain how these specific recommendations (like the docking station or HDMI cables) solve a 'workflow' problem.
    - Sentence 3: End with a call to action to check out their personalized deals.
    
    Don't use placeholders like [Product Name], use the actual titles provided.
    """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Could not generate pitch: {e}"