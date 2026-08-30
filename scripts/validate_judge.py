import os
import json
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACE_FILE = os.path.join(BASE_DIR, "traces.jsonl")
CSV_FILE = os.path.join(BASE_DIR, "annotations.csv")

def validate_judge():
    """
    Validates the LLM-as-a-judge against human severity ratings 
    proving human-AI agreement before trusting the metric.
    """
    if "OPENAI_API_KEY" not in os.environ:
        print("WARNING: OPENAI_API_KEY environment variable is not set.")
    
    if not os.path.exists(CSV_FILE) or not os.path.exists(TRACE_FILE):
        print("Required files missing. Please generate traces and manually annotate the CSV!")
        return
        
    annotations = pd.read_csv(CSV_FILE)
    if "Severity (1-5)" not in annotations.columns or annotations["Severity (1-5)"].dropna().empty:
        print("No human 'Severity (1-5)' ratings found in the annotations CSV to compare against!")
        return

    # Load traces to run the judge
    data_map = {}
    with open(TRACE_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            tr = json.loads(line)
            data_map[tr["query"]] = tr

    eval_data = []
    severities = []
    
    for idx, row in annotations.iterrows():
        query = row.get("Query", "")
        sev = row.get("Severity (1-5)")
        # Skip if no severity annotated by human
        if pd.isna(sev) or query not in data_map:
            continue
            
        trace = data_map[query]
        eval_data.append({
            "question": trace["query"],
            "answer": trace["answer"],
            "contexts": [c["text"] for c in trace.get("fetched_chunks", [])],
        })
        severities.append(float(sev))
        
    if not eval_data:
        print("No valid paired human annotations found!")
        return

    print("Running LLM judge (Answer Relevancy) to compare against human severity...")
    dataset = Dataset.from_pandas(pd.DataFrame(eval_data))
    
    results = evaluate(dataset, metrics=[answer_relevancy])
    df_res = results.to_pandas()
    
    # Calculate agreement (correlation)
    llm_scores = df_res["answer_relevancy"].values
    human_severities = np.array(severities)
    
    # Usually, high severity -> low relevancy. They should have negative correlation.
    correlation = np.corrcoef(human_severities, llm_scores)[0, 1]
    
    print("\n--- JUDGE VALIDATION REPORT ---")
    print(f"Assessed {len(eval_data)} annotated queries.")
    print(f"Correlation between Human Severity and LLM Relevancy Score: {correlation:.4f}")
    if correlation < -0.3:
        print("Conclusion: VALIDATED! The LLM judge aligns strongly with human penalty grades.")
    elif correlation > 0.3:
        print("Conclusion: INVERSE VALIDATION? Warning, high severity is getting high LLM scores.")
    else:
        print("Conclusion: WEAK CORRELATION. You may need to tune the LLM judge prompt.")

if __name__ == "__main__":
    validate_judge()
