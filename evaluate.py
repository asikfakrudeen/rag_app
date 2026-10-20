"""
evaluate.py — Capstone End-to-End Evaluation Suite
====================================================
Runs all three evaluation stages and prints a single consolidated report.

Usage:
    python evaluate.py
    python evaluate.py --hybrid --rerank      (enable both retrieval upgrades)
"""

import argparse
import subprocess
import sys
import os

from rag.vector_store import get_collection, get_all_documents
from rag.hybrid_retriever import hybrid_retrieve, init_bm25

# ---------------------------------------------------------------------------
# Evaluation dataset (ground-truth retrieval pairs)
# ---------------------------------------------------------------------------
EVAL_DATA = [
    {
        "query": "What is the penalty for violating ERR-4032?",
        "expected_source": "Vendor_Agreement.pdf",
    },
    {
        "query": "Are there any provisions about mutual termination in section 4(b)?",
        "expected_source": "Master_Service_Agreement.pdf",
    },
    {
        "query": "What is the notice period required for contract termination?",
        "expected_source": "Master_Service_Agreement.pdf",
    },
    {
        "query": "What are the indemnification obligations under this agreement?",
        "expected_source": "Vendor_Agreement.pdf",
    },
]


# ---------------------------------------------------------------------------
# Retrieval Metrics
# ---------------------------------------------------------------------------

def mrr_at_k(expected_source: str, retrieved_metas: list) -> float:
    for i, meta in enumerate(retrieved_metas[0]):
        if meta.get("source") == expected_source:
            return 1.0 / (i + 1)
    return 0.0


def hit_rate_at_k(expected_source: str, retrieved_metas: list) -> float:
    for meta in retrieved_metas[0]:
        if meta.get("source") == expected_source:
            return 1.0
    return 0.0


def run_retrieval_eval(use_hybrid: bool = True, use_rerank: bool = True) -> dict:
    try:
        collection = get_collection("legal_contracts")
        all_ids, all_docs, all_metas = get_all_documents(collection)

        if not all_ids:
            return {
                "status": "skipped",
                "reason": "No documents indexed. Upload PDFs and build the index first.",
            }

        bm25_index = init_bm25(all_docs)
        total_mrr = total_hit = 0.0

        for item in EVAL_DATA:
            results = hybrid_retrieve(
                collection=collection,
                bm25_index=bm25_index,
                all_ids=all_ids,
                all_documents=all_docs,
                all_metadatas=all_metas,
                query=item["query"],
                top_k=5,
                use_hybrid=use_hybrid,
                use_rerank=use_rerank,
            )
            total_mrr += mrr_at_k(item["expected_source"], results["metadatas"])
            total_hit += hit_rate_at_k(item["expected_source"], results["metadatas"])

        n = len(EVAL_DATA)
        return {
            "status": "ok",
            "mrr_at_5": round(total_mrr / n, 4),
            "hit_rate_at_5": round(total_hit / n, 4),
            "queries_evaluated": n,
        }

    except Exception as exc:
        return {"status": "error", "reason": str(exc)}


# ---------------------------------------------------------------------------
# Regression Test
# ---------------------------------------------------------------------------

def run_regression_test() -> dict:
    regression_script = os.path.join("scripts", "test_termination_regression.py")
    if not os.path.isfile(regression_script):
        return {"status": "skipped", "reason": "scripts/test_termination_regression.py not found"}

    python = os.path.join(".venv", "Scripts", "python.exe")
    if not os.path.isfile(python):
        python = sys.executable

    result = subprocess.run(
        [python, regression_script],
        capture_output=True,
        text=True,
        timeout=60,
    )
    passed = result.returncode == 0
    return {
        "status": "passed" if passed else "FAILED",
        "output": (result.stdout + result.stderr).strip(),
    }


# ---------------------------------------------------------------------------
# Report Printer
# ---------------------------------------------------------------------------

def _section(title: str):
    print(f"\n{'─' * 58}")
    print(f"  {title}")
    print(f"{'─' * 58}")


def print_report(args):
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║      ⚖️  Legal Contract Assistant — Eval Report          ║")
    print("║               Week 12 Capstone · Track F                ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # ── Stage 1: Retrieval ─────────────────────────────────────────
    _section("Stage 1 · Retrieval Quality")

    configs = [
        ("Baseline (semantic only)",        False, False),
        ("Hybrid BM25 + semantic",           True,  False),
        ("Hybrid + cross-encoder reranking", True,  True),
    ]

    for label, hybrid, rerank in configs:
        r = run_retrieval_eval(use_hybrid=hybrid, use_rerank=rerank)
        if r["status"] == "ok":
            print(f"  {label}")
            print(f"    MRR@5 = {r['mrr_at_5']:.4f}   HitRate@5 = {r['hit_rate_at_5']:.4f}")
        elif r["status"] == "skipped":
            print(f"  ⚠️  Skipped — {r['reason']}")
            break
        else:
            print(f"  ❌  Error — {r['reason']}")

    # ── Stage 2: Regression Test ───────────────────────────────────
    _section("Stage 2 · Regression Test (Termination Clause)")

    reg = run_regression_test()
    icon = "✅" if reg["status"] == "passed" else ("⚠️" if reg["status"] == "skipped" else "❌")
    print(f"  {icon}  Status: {reg['status']}")
    if reg.get("output"):
        for line in reg["output"].splitlines()[:10]:
            print(f"     {line}")

    # ── Stage 3: Summary ───────────────────────────────────────────
    _section("Stage 3 · How to Read These Numbers")
    print("  • MRR@5   — Mean Reciprocal Rank: 1.0 = correct doc always ranked #1")
    print("  • HitRate — Did the correct doc appear anywhere in top-5? 1.0 = always")
    print("  • Regression PASSED = termination-clause hallucination is not regressed")
    print()
    print("  For multi-agent quality scores run:")
    print("    .venv\\Scripts\\python.exe scripts/race_multi_vs_single.py")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capstone eval suite")
    parser.add_argument("--hybrid", action="store_true", help="Enable BM25 hybrid retrieval")
    parser.add_argument("--rerank", action="store_true", help="Enable cross-encoder reranking")
    args = parser.parse_args()
    print_report(args)
