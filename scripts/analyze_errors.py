import csv
import os
from collections import Counter

CSV_FILE = "annotations.csv"

def analyze_errors():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, CSV_FILE)

    if not os.path.exists(csv_path):
        print(f"No {CSV_FILE} found. Please run export_traces.py and annotate it first!")
        return

    categories = Counter()
    total_analyzed = 0
    total_failures = 0

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_analyzed += 1
            # Note: Checking for permutations of column names just in case user edited them
            category = row.get("Problem Category", "").strip()
            
            if category:
                categories[category] += 1
                total_failures += 1
                
    print(f"\n--- ERROR ANALYSIS REPORT ---")
    print(f"Total Traces Analyzed: {total_analyzed}")
    print(f"Total Annotated Failures: {total_failures}\n")
    
    if total_failures == 0:
        print("No failures were categorized. Try categorizing some in the CSV!")
        return
        
    print("Ranking of Problem Types:")
    ranked = categories.most_common()
    for rank, (cat, count) in enumerate(ranked, 1):
        print(f" {rank}. {cat} ({count} occurrences)")
        
    print(f"\nTarget to fix next: '{ranked[0][0]}' is causing the most issues and should be prioritized.")
    print("-----------------------------\n")

if __name__ == "__main__":
    analyze_errors()
