import os
import json
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACE_FILE = os.path.join(BASE_DIR, "traces.jsonl")
CSV_FILE = os.path.join(BASE_DIR, "annotations.csv")

def evaluate_traces():
    """Runs a full automated score evaluation on the pipeline before/after."""
    if "OPENAI_API_KEY" not in os.environ:
        print("WARNING: OPENAI_API_KEY environment variable is not set. Ragas will fail if executed.")
        
    if not os.path.exists(TRACE_FILE):
        print(f"File not found: {TRACE_FILE}. Please use the app to generate some data.")
        return

    # Load traces to build the dataset
    data = []
    with open(TRACE_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            trace = json.loads(line)
            data.append({
                "question": trace.get("query", ""),
                "answer": trace.get("answer", ""),
                "contexts": [c.get("text", "") for c in trace.get("fetched_chunks", [])],
                # Mock ground truth for metrics that demand it. 
                # Realistically ground truth should be mapped from annotations manually.
                "ground_truth": trace.get("answer", "")
            })

    if not data:
        print("No traces available to evaluate!")
        return

    dataset = Dataset.from_pandas(pd.DataFrame(data))
    
    print(f"Evaluating {len(dataset)} examples...")
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    
    # Run the evaluation
    results = evaluate(dataset, metrics=metrics)
    
    print("\n--- DELTA SCORE RAPORT ---")
    print("Aggregate Scores:")
    for m, score in results.items():
        print(f" - {m}: {score:.4f}")
        
    # Write to a file for before/after comparison
    df = results.to_pandas()
    df.to_csv(os.path.join(BASE_DIR, "ragas_eval_results.csv"), index=False)
    print("Detailed scores saved to ragas_eval_results.csv")

if __name__ == "__main__":
    evaluate_traces()
