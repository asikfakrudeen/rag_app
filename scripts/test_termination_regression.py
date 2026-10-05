"""
Week 11 — Regression Test

This test verifies the fix for the support drill.
It calls the live API and ensures it does not reproduce the '60 days' error.

To run this, the FastAPI backend must be running.
"""

import urllib.request
import json

URL = "http://localhost:8000/ask"

def run_regression_test():
    print("Running Regression Test: Termination Notice Period")
    
    req_body = json.dumps({"question": "What is the termination notice period?"}).encode("utf-8")
    req = urllib.request.Request(URL, data=req_body, headers={'Content-Type': 'application/json'})
    
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            answer = data.get("answer", "")
            trace_id = data.get("trace_id", "MISSING")
            
            print(f"Received Answer: {answer}")
            print(f"Trace ID: {trace_id}")
            
            if "60 days" in answer.lower():
                print("\n❌ FAIL: The model still hallucinated/found the bad 60 days clause!")
                exit(1)
            else:
                print("\n✅ PASS: The model did not output the known bad response.")
                
    except Exception as e:
        print(f"Error calling API (is it running?): {e}")

if __name__ == "__main__":
    run_regression_test()
