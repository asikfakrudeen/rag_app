import os
import json
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy
import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACE_FILE = os.path.join(BASE_DIR, "traces.jsonl")
CSV_FILE = os.path.join(BASE_DIR, "annotations.csv")

def validate_judge():
    """
    Validates the FREE Gemini LLM-as-a-judge against human severity ratings 
    proving human-AI agreement before trusting the metric.
    """
    if "GOOGLE_API_KEY" not in os.environ:
        print("WARNING: GOOGLE_API_KEY environment variable is not set.")
    
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

    print("Running Gemini LLM judge (Answer Relevancy) to compare against human severity...")
    dataset = Dataset.from_pandas(pd.DataFrame(eval_data))
    
    # Configure Free Gemini Judge
    gemini_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    results = evaluate(
        dataset, 
        metrics=[answer_relevancy],
        llm=gemini_llm,
        embeddings=gemini_embeddings
    )
    
    df_res = results.to_pandas()
    
    llm_scores = df_res["answer_relevancy"].values
    human_severities = np.array(severities)
    
    correlation = np.corrcoef(human_severities, llm_scores)[0, 1]
    
    print("\n--- JUDGE VALIDATION REPORT ---")
    print(f"Assessed {len(eval_data)} annotated queries.")
    print(f"Correlation between Human Severity and Gemini Relevancy Score: {correlation:.4f}")
    if correlation < -0.3:
        print("Conclusion: VALIDATED! The Free Gemini judge aligns strongly with human penalty grades.")
    elif correlation > 0.3:
        print("Conclusion: INVERSE VALIDATION? Warning, high severity is getting high Gemini scores.")
    else:
        print("Conclusion: WEAK CORRELATION. Gemini may struggle to judge these exact answers properly.")

if __name__ == "__main__":
    validate_judge()
