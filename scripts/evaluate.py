import os
import json
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACE_FILE = os.path.join(BASE_DIR, "traces.jsonl")

def evaluate_traces():
    """Runs a full automated score evaluation using FREE Google Gemini Tier!"""
    if "GOOGLE_API_KEY" not in os.environ:
        print("WARNING: GOOGLE_API_KEY environment variable is not set. The evaluation will fail.")
        
    if not os.path.exists(TRACE_FILE):
        print(f"File not found: {TRACE_FILE}. Please use the app to generate some data first.")
        return

    data = []
    with open(TRACE_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            trace = json.loads(line)
            data.append({
                "question": trace.get("query", ""),
                "answer": trace.get("answer", ""),
                "contexts": [c.get("text", "") for c in trace.get("fetched_chunks", [])],
                "ground_truth": trace.get("answer", "")
            })

    if not data:
        print("No traces available to evaluate!")
        return

    dataset = Dataset.from_pandas(pd.DataFrame(data))
    print(f"Evaluating {len(dataset)} examples via Gemini Assistant...")
    
    # Init Gemini specific to Ragas using the free tier!
    gemini_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    
    # Override Ragas Defaults:
    results = evaluate(
        dataset, 
        metrics=metrics,
        llm=gemini_llm,
        embeddings=gemini_embeddings
    )
    
    print("\n--- DELTA SCORE REPORT ---")
    print("Aggregate Scores:")
    for m, score in results.items():
        print(f" - {m}: {score:.4f}")
        
    # Write to a file for before/after comparison
    df = results.to_pandas()
    df.to_csv(os.path.join(BASE_DIR, "ragas_eval_results.csv"), index=False)
    print("Detailed scores saved to ragas_eval_results.csv")

if __name__ == "__main__":
    evaluate_traces()
