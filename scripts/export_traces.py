import json
import csv
import os

TRACE_FILE = "traces.jsonl"
CSV_FILE = "annotations.csv"

def export_traces_to_csv():
    # Helper to jump to the parent dir if executed directly from inside scripts/
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    trace_path = os.path.join(base_dir, TRACE_FILE)
    csv_path = os.path.join(base_dir, CSV_FILE)

    if not os.path.exists(trace_path):
        print(f"No {TRACE_FILE} found running locally. Ask some questions in the app first!")
        return

    traces = []
    with open(trace_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                traces.append(json.loads(line))
                
    # Sort traces by timestamp (newest first)
    traces.sort(key=lambda x: x["timestamp"], reverse=True)
                
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Timestamp", "Query", "Final Answer", 
            "Fetched Chunks Count", 
            "Failure Note", 
            "Problem Category", 
            "Severity (1-5)"
        ])
        
        for trace in traces:
            writer.writerow([
                trace["timestamp"],
                trace["query"],
                trace["answer"],
                len(trace.get("fetched_chunks", [])),
                "", # Blank for Failure Note
                "", # Blank for Problem Category
                ""  # Blank for Severity
            ])
            
    print(f"Exported {len(traces)} traces to {CSV_FILE}.")
    print(f"Please open {csv_path} in Excel/Sheets, fill out the last 3 columns for each error, and save it.")

if __name__ == "__main__":
    export_traces_to_csv()
