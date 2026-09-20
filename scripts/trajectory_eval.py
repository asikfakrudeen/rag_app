"""
Week 8: Trajectory Evaluator for Legal Contract RAG Agent
========================================================
Reads traces.jsonl and evaluates each agent run on PATH quality,
not just final answer correctness. Detects the outcome-vs-trajectory gap.

Usage:
    python scripts/trajectory_eval.py

Output:
    - Prints a report to the terminal
    - Saves per-trace scores to trajectory_report.csv
"""

import json
import csv
import os
import re
from collections import Counter

# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACE_FILE = os.path.join(BASE_DIR, "traces.jsonl")
REPORT_FILE = os.path.join(BASE_DIR, "trajectory_report.csv")

# Known injection / red-flag patterns in observations (used later too)
INJECTION_PATTERNS = [
    r"ignore (previous|all|prior) instruction",
    r"disregard (previous|all|prior)",
    r"new instruction",
    r"act as ",
    r"forget (everything|all)",
    r"you are now",
]

# Keywords that signal the agent hallucinated (claimed something with no observation support)
HEDGE_PHRASES = [
    "according to the contract",
    "the contract states",
    "the agreement specifies",
    "section",
    "clause",
    "the document says",
]


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def parse_agent_steps(memory: str):
    """
    Parse the ReAct memory string into a list of steps.
    Returns a dict with:
        - thoughts    : list of Thought strings
        - actions     : list of Action strings
        - action_inputs: list of Action Input strings
        - observations: list of Observation strings
        - final_answer: str or None
        - format_errors: int (count of format error observations)
    """
    steps = {
        "thoughts": re.findall(r"Thought:\s*(.*?)(?=\n|$)", memory),
        "actions": re.findall(r"Action:\s*(.*?)(?=\n|$)", memory),
        "action_inputs": re.findall(r"Action Input:\s*(.*?)(?=\n|$)", memory),
        "observations": re.findall(r"Observation:\s*(.*?)(?=\nThought:|\nFinal Answer:|$)", memory, re.DOTALL),
        "final_answer": None,
        "format_errors": 0,
    }

    fa_match = re.search(r"Final Answer:\s*(.*?)$", memory, re.DOTALL)
    if fa_match:
        steps["final_answer"] = fa_match.group(1).strip()

    steps["format_errors"] = sum(
        1 for obs in steps["observations"]
        if "format error" in obs.lower()
    )

    return steps


def score_path(steps: dict, final_answer: str) -> dict:
    """
    Score the agent's trajectory on several dimensions.
    Returns a result dict.
    """
    result = {}

    n_actions = len(steps["actions"])
    n_format_errors = steps["format_errors"]
    observations = steps["observations"]
    final_ans = steps["final_answer"] or final_answer or ""

    # --- 1. Tool was called at all before Final Answer ---
    result["called_tool"] = n_actions > 0

    # --- 2. Loop detection: repeated identical action inputs ---
    inputs = [s.strip().lower() for s in steps["action_inputs"]]
    loop_detected = len(inputs) != len(set(inputs)) and len(inputs) > 1
    result["loop_detected"] = loop_detected

    # --- 3. Format error rate ---
    total_steps = max(n_actions + 1, 1)  # avoid div by zero
    result["format_error_rate"] = round(n_format_errors / total_steps, 3)
    result["format_errors"] = n_format_errors

    # --- 4. Gave up quietly (hit budget without Final Answer) ---
    result["gave_up"] = steps["final_answer"] is None

    # --- 5. Step efficiency score (1.0 = optimal, drops with wasted steps) ---
    wasted = n_format_errors + (1 if loop_detected else 0)
    efficiency = max(0.0, 1.0 - (wasted / max(total_steps, 1)))
    result["step_efficiency"] = round(efficiency, 3)

    # --- 6. Hallucination signal: final answer uses hedge phrases
    #        but none of those claims appear verbatim in any observation ---
    found_support = False
    has_hedges = any(hp in final_ans.lower() for hp in HEDGE_PHRASES)
    if has_hedges and observations:
        obs_combined = " ".join(observations).lower()
        # Check for at least some overlap between answer words and observations
        answer_words = set(re.findall(r"\b\w{5,}\b", final_ans.lower()))
        obs_words = set(re.findall(r"\b\w{5,}\b", obs_combined))
        overlap = answer_words & obs_words
        found_support = len(overlap) > 3
    result["possible_hallucination"] = has_hedges and not found_support

    # --- 7. Outcome vs Trajectory gap ---
    #        "Outcome OK" = agent gave a non-trivial final answer
    #        "Path OK"    = called tool, no loops, no format errors, efficient
    outcome_ok = bool(final_ans) and "agent failed" not in final_ans.lower()
    path_ok = result["called_tool"] and not loop_detected and n_format_errors == 0
    result["outcome_ok"] = outcome_ok
    result["path_ok"] = path_ok
    result["gap_detected"] = outcome_ok and not path_ok  # THE KEY METRIC

    # --- 8. Overall path score (0.0 – 1.0) ---
    score = result["step_efficiency"]
    if not result["called_tool"]:
        score -= 0.3
    if result["loop_detected"]:
        score -= 0.3
    if result["possible_hallucination"]:
        score -= 0.2
    result["path_score"] = round(max(0.0, score), 3)

    return result


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def run_trajectory_eval():
    if not os.path.exists(TRACE_FILE):
        print(f"[ERROR] {TRACE_FILE} not found. Ask some questions in the app first!")
        return

    traces = []
    with open(TRACE_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                traces.append(json.loads(line))

    if not traces:
        print("[INFO] traces.jsonl is empty.")
        return

    print(f"\n=== TRAJECTORY EVALUATION REPORT ===")
    print(f"Total traces loaded: {len(traces)}\n")

    rows = []
    gap_count = 0
    loop_count = 0
    hallucination_count = 0
    gave_up_count = 0
    total_format_errors = 0

    for i, trace in enumerate(traces):
        query = trace.get("query", "")
        answer = trace.get("answer", "")
        chunks = trace.get("fetched_chunks", [])

        # Pull the agent memory from the stored chunks (agent traces store memory there)
        memory_text = ""
        for chunk in chunks:
            if chunk.get("source") == "Agent Memory":
                memory_text = chunk.get("text", "")
                break

        if memory_text:
            steps = parse_agent_steps(memory_text)
        else:
            # Non-agent trace (direct RAG call) — minimal evaluation
            steps = {
                "thoughts": [],
                "actions": ["search_contract"],  # implicitly called
                "action_inputs": [query],
                "observations": [c.get("text", "") for c in chunks],
                "final_answer": answer,
                "format_errors": 0,
            }

        result = score_path(steps, answer)
        total_format_errors += result["format_errors"]

        if result["gap_detected"]:
            gap_count += 1
        if result["loop_detected"]:
            loop_count += 1
        if result["possible_hallucination"]:
            hallucination_count += 1
        if result["gave_up"]:
            gave_up_count += 1

        import datetime
        ts = trace.get("timestamp", 0)
        ts_str = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if ts else ""

        row = {
            "trace_index": i + 1,
            "timestamp": ts_str,
            "query": query[:120],
            "outcome_ok": result["outcome_ok"],
            "path_ok": result["path_ok"],
            "gap_detected": result["gap_detected"],
            "path_score": result["path_score"],
            "step_efficiency": result["step_efficiency"],
            "format_errors": result["format_errors"],
            "format_error_rate": result["format_error_rate"],
            "loop_detected": result["loop_detected"],
            "gave_up": result["gave_up"],
            "possible_hallucination": result["possible_hallucination"],
            "final_answer_snippet": answer[:150],
        }
        rows.append(row)

        # Print flagged traces
        flag = []
        if result["gap_detected"]:
            flag.append("[GAP] outcome-ok, path-bad")
        if result["loop_detected"]:
            flag.append("[LOOP]")
        if result["gave_up"]:
            flag.append("[GAVE UP]")
        if result["possible_hallucination"]:
            flag.append("[POSSIBLE HALLUCINATION]")
        if result["format_errors"] > 0:
            flag.append(f"[{result['format_errors']} FORMAT ERRORS]")

        status = " | ".join(flag) if flag else "[OK]"
        print(f"[{i+1}] path_score={result['path_score']:.2f} | {status}")
        if flag:
            print(f"     Query: {query[:80]}")
            print(f"     Answer: {answer[:80]}")

    # --- Save CSV ---
    fieldnames = list(rows[0].keys()) if rows else []
    with open(REPORT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # --- Summary ---
    n = len(traces)
    avg_path = round(sum(r["path_score"] for r in rows) / n, 3) if n else 0
    print(f"\n--- SUMMARY ---")
    print(f"  Total traces:           {n}")
    print(f"  Outcome-vs-Path gaps:   {gap_count} ({100*gap_count//n if n else 0}%)")
    print(f"  Loops detected:         {loop_count}")
    print(f"  Gave up (budget hit):   {gave_up_count}")
    print(f"  Possible hallucinations:{hallucination_count}")
    print(f"  Total format errors:    {total_format_errors}")
    print(f"  Avg path score:         {avg_path}")
    print(f"\n  Saved: {REPORT_FILE}")
    print(f"  (Open trajectory_report.csv to see per-trace details)")
    print("-----------------------------------\n")

    return {
        "gap_count": gap_count,
        "loop_count": loop_count,
        "gave_up_count": gave_up_count,
        "hallucination_count": hallucination_count,
        "total_format_errors": total_format_errors,
        "avg_path_score": avg_path,
        "n_traces": n,
    }


if __name__ == "__main__":
    run_trajectory_eval()
