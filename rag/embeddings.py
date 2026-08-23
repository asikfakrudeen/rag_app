from google import genai
from dotenv import load_dotenv
import os

# Load environment variables from the .env file (like GOOGLE_API_KEY)
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize the Gemini GenAI client
client = genai.Client(api_key=API_KEY)


def create_embedding(text):
    """
    Sends text to Google Gemini's embedding model to convert the text 
    into a high-dimensional vector (an array of numbers representing meaning).
    
    Example:
        Input: create_embedding("This is a legal contract.")
        Output: [0.0123, -0.0456, 0.0789, ...] # (A long array of floats)
    """

    # Hit the Gemini API to get the embedding vector for the single text chunk
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )

    # Return the raw numerical array from the response object
    return result.embeddings[0].values
