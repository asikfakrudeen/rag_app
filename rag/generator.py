from google import genai
from dotenv import load_dotenv
import os

# Load environment variables (such as GOOGLE_API_KEY)
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize the Gemini client to query the LLM
client = genai.Client(api_key=API_KEY)


def generate_answer(question, results):
    """
    Generate an answer using retrieved context (from ChromaDB) and the Gemini model.
    
    Example:
        Input: 
            question = "How many days for termination?"
            results = {
                "documents": [["Notice period is 30 days."]],
                "metadatas": [[{"page": 1, "source": "contract.pdf"}]]
            }
            generate_answer(question, results)
            
        Output: "The notice period for termination is 30 days."
    """

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    context_parts = []

    # Iterate through the retrieved chunks and format them nicely with their metadata
    for document, metadata in zip(documents, metadatas):

        context_parts.append(
            f"""
SOURCE: {metadata['source']}
PAGE: {metadata['page']}

{document}
"""
        )

    # Stitch all the chunks together into a single block of context text
    context = "\n\n".join(context_parts)

    # Create the strict prompt template directing Gemini on how to behave
    prompt = f"""
You are a legal contract document assistant.

Answer the user's question ONLY using the
provided amendment documents.

Rules:

1. Do not use outside knowledge.
2. Do not invent contract terms.
3. Do not make assumptions.
4. If the answer cannot be found in the
   provided context, say:

"I could not find this information in the
provided amendment documents."

5. Give a concise answer.
6. Mention the relevant source and page.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}
"""

    # Send the structured prompt containing the context to the Gemini model
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    # Return the clean text response from Gemini
    return response.text
