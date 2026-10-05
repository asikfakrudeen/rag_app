r"""
Week 10 — Race: Multi-Agent Contract Squad vs Single ReAct Agent

Runs both systems on the same benchmark questions and reports:
    Quality   — LLM-as-judge score 1-5
    Latency   — wall-clock seconds
    Tokens    — input + output tokens consumed
    Cost      — estimated USD (Gemini 2.0 Flash pricing)

Usage:
    .\.venv\Scripts\python.exe scripts/race_multi_vs_single.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import json
from google import genai
from dotenv import load_dotenv

load_dotenv()

from rag.vector_store import get_collection, get_all_documents
from rag.hybrid_retriever import init_bm25
from rag.agent_loop import run_agent_loop
from rag.multi_agent.orchestrator import OrchestratorAgent

_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
_MODEL  = "gemini-2.0-flash"

# ------------------------------------------------------------------
# Gemini 2.0 Flash pricing
# ------------------------------------------------------------------
_INPUT_COST_PER_TOKEN  = 0.075 / 1_000_000
_OUTPUT_COST_PER_TOKEN = 0.30  / 1_000_000

# ------------------------------------------------------------------
# Benchmark questions (Track F — Legal Contracts)
# ------------------------------------------------------------------
BENCHMARK_QUESTIONS = [
    "What is the notice period required for contract termination?",
    "Who owns the intellectual property created during the engagement?",
    "What are the payment terms and late payment penalties?",
    "Under what conditions can either party claim indemnification?",
    "What is the governing law and jurisdiction for dispute resolution?",
]


# ------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------

def setup_app_state():
    print("[SETUP] Loading index...")
    collection = get_collection("legal_contracts")
    if collection.count() == 0:
        print("ERROR: Collection is empty. Please index a document via app.py first.")
        return None
    all_ids, all_docs, all_metas = get_all_documents(collection)
    bm25_index = init_bm25(all_docs)
    return {
        "indexed":    True,
        "all_ids":    all_ids,
        "all_docs":   all_docs,
        "all_metas":  all_metas,
        "bm25_index": bm25_index,
    }


# ------------------------------------------------------------------
# Token-aware single-agent wrapper
# ------------------------------------------------------------------

def run_single_agent_with_metrics(question: str, app_state: dict) -> dict:
    """Wraps run_agent_loop and adds token/cost estimates via usage_metadata."""
    t0 = time.time()
    result = run_agent_loop(question, app_state, max_iterations=5)
    latency = time.time() - t0

    # Estimate tokens from memory length (no usage_metadata in agent_loop).
    # We do one extra judge call later that has real usage; here we use a
    # conservative character-based estimate for the agent turns.
    memory_chars  = len(result.get("memory", ""))
    # ~4 chars per token is a common rough estimate
    est_total_tok  = memory_chars // 4
    est_input_tok  = int(est_total_tok * 0.75)
    est_output_tok = est_total_tok - est_input_tok
    est_cost = (
        est_input_tok  * _INPUT_COST_PER_TOKEN +
        est_output_tok * _OUTPUT_COST_PER_TOKEN
    )

    return {
        "answer":              result.get("answer", ""),
        "latency_s":           round(latency, 3),
        "input_tokens":        est_input_tok,
        "output_tokens":       est_output_tok,
        "total_tokens":        est_total_tok,
        "estimated_cost_usd":  round(est_cost, 8),
    }


# ------------------------------------------------------------------
# LLM-as-judge quality scoring
# ------------------------------------------------------------------

def judge_quality(question: str, answer: str) -> tuple[int, str]:
    """
    Ask Gemini to score an answer 1-5.
    Returns (score, reasoning).
    """
    prompt = f"""You are an expert legal document reviewer acting as a judge.

Score the following answer to a legal contract question on a scale of 1 to 5:
1 = Completely wrong or refuses to answer
2 = Partially relevant but mostly incorrect
3 = Partially correct with gaps
4 = Mostly correct, minor gaps
5 = Accurate, complete, and well-cited

Question: {question}

Answer: {answer}

Respond with ONLY a JSON object in this exact format:
{{"score": <integer 1-5>, "reasoning": "<one sentence>"}}
"""
    try:
        response = _client.models.generate_content(model=_MODEL, contents=prompt)
        text = response.text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        parsed = json.loads(text.strip())
        return int(parsed["score"]), parsed.get("reasoning", "")
    except Exception as e:
        return 0, f"Judge failed: {e}"


# ------------------------------------------------------------------
# Main race
# ------------------------------------------------------------------

def run_race(app_state: dict):
    squad = OrchestratorAgent()

    rows_single = []
    rows_multi  = []

    print("\n" + "="*70)
    print("  WEEK 10 RACE — Multi-Agent Contract Squad vs Single ReAct Agent")
    print("="*70)

    for i, question in enumerate(BENCHMARK_QUESTIONS, 1):
        print(f"\n[Q{i}] {question}")
        print("-" * 60)

        # ── Single agent ──────────────────────────────────────────
        print("  → Running Single ReAct Agent …")
        try:
            single = run_single_agent_with_metrics(question, app_state)
        except Exception as e:
            single = {"answer": f"ERROR: {e}", "latency_s": 0,
                      "total_tokens": 0, "estimated_cost_usd": 0}

        # ── Multi-agent squad ─────────────────────────────────────
        print("  → Running Multi-Agent Squad …")
        try:
            multi = squad.run(question, app_state)
        except Exception as e:
            multi = {"answer": f"ERROR: {e}", "latency_s": 0,
                     "total_tokens": 0, "estimated_cost_usd": 0}

        # ── Quality scores ────────────────────────────────────────
        print("  → Judging quality …")
        single_score, single_reasoning = judge_quality(question, single["answer"])
        multi_score,  multi_reasoning  = judge_quality(question, multi["answer"])

        print(f"  Single → quality={single_score}/5  latency={single['latency_s']}s  tokens={single['total_tokens']}  cost=${single['estimated_cost_usd']:.6f}")
        print(f"  Multi  → quality={multi_score}/5   latency={multi['latency_s']}s  tokens={multi['total_tokens']}  cost=${multi['estimated_cost_usd']:.6f}")

        rows_single.append({
            "question": question,
            "quality":  single_score,
            "latency":  single["latency_s"],
            "tokens":   single["total_tokens"],
            "cost":     single["estimated_cost_usd"],
            "answer":   single["answer"],
            "reasoning": single_reasoning,
        })
        rows_multi.append({
            "question": question,
            "quality":  multi_score,
            "latency":  multi["latency_s"],
            "tokens":   multi["total_tokens"],
            "cost":     multi["estimated_cost_usd"],
            "answer":   multi["answer"],
            "reasoning": multi_reasoning,
            "agents_called": multi.get("agents_called", []),
        })

    # ── Final summary ─────────────────────────────────────────────
    generate_report(rows_single, rows_multi)


def generate_report(rows_single, rows_multi):
    def avg(rows, key):
        vals = [r[key] for r in rows if isinstance(r[key], (int, float))]
        return round(sum(vals) / len(vals), 4) if vals else 0

    avg_q_s = avg(rows_single, "quality")
    avg_q_m = avg(rows_multi,  "quality")
    avg_l_s = avg(rows_single, "latency")
    avg_l_m = avg(rows_multi,  "latency")
    avg_t_s = avg(rows_single, "tokens")
    avg_t_m = avg(rows_multi,  "tokens")
    avg_c_s = avg(rows_single, "cost")
    avg_c_m = avg(rows_multi,  "cost")

    # Verdict
    wins_multi = sum(1 for s, m in zip(rows_single, rows_multi) if m["quality"] >= s["quality"])
    winner = "Multi-Agent Squad" if avg_q_m >= avg_q_s else "Single ReAct Agent"
    verdict_reason = (
        f"The {winner} wins on average quality ({avg_q_m:.2f} vs {avg_q_s:.2f}). "
        f"The squad used {avg_t_m:.0f} tokens on average vs {avg_t_s:.0f} for the single agent "
        f"and cost ${avg_c_m:.6f} vs ${avg_c_s:.6f} per question."
    )
    when_multi_wins = (
        "Multi-agent is worth the extra token cost when questions require "
        "distinct reasoning steps (e.g., clause extraction + risk analysis + synthesis), "
        "or when you want specialist-grade accuracy per sub-task. "
        "For simple single-turn lookups, the single ReAct agent is cheaper and faster."
    )

    # Terminal print
    print("\n" + "="*70)
    print("  RESULTS SUMMARY")
    print("="*70)
    print(f"  {'Metric':<22} {'Single Agent':>15} {'Multi-Agent':>15}")
    print(f"  {'-'*22} {'-'*15} {'-'*15}")
    print(f"  {'Avg Quality (1-5)':<22} {avg_q_s:>15.2f} {avg_q_m:>15.2f}")
    print(f"  {'Avg Latency (s)':<22} {avg_l_s:>15.3f} {avg_l_m:>15.3f}")
    print(f"  {'Avg Tokens':<22} {avg_t_s:>15.0f} {avg_t_m:>15.0f}")
    print(f"  {'Avg Cost (USD)':<22} ${avg_c_s:>14.6f} ${avg_c_m:>14.6f}")
    print(f"\n  🏆 VERDICT: {verdict_reason}")
    print(f"\n  💡 When to use multi-agent: {when_multi_wins}")

    # Write markdown report
    report_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "race_report.md"
    )
    lines = [
        "# Week 10 — Race Report: Multi-Agent vs Single ReAct Agent",
        "",
        "## Summary Table",
        "",
        "| Metric | Single Agent | Multi-Agent Squad |",
        "|---|---|---|",
        f"| Avg Quality (1–5) | {avg_q_s:.2f} | {avg_q_m:.2f} |",
        f"| Avg Latency (s) | {avg_l_s:.3f} | {avg_l_m:.3f} |",
        f"| Avg Tokens | {avg_t_s:.0f} | {avg_t_m:.0f} |",
        f"| Avg Cost (USD) | ${avg_c_s:.6f} | ${avg_c_m:.6f} |",
        "",
        f"## 🏆 Verdict",
        "",
        verdict_reason,
        "",
        f"## 💡 When Multi-Agent Is Worth It",
        "",
        when_multi_wins,
        "",
        "---",
        "",
        "## Per-Question Detail",
        "",
    ]

    for i, (s, m) in enumerate(zip(rows_single, rows_multi), 1):
        lines += [
            f"### Q{i}: {s['question']}",
            "",
            f"**Single Agent** — Quality: {s['quality']}/5 | Latency: {s['latency']}s | Tokens: {s['tokens']} | Cost: ${s['cost']:.6f}",
            f"> {s['answer'][:300]}{'…' if len(s['answer']) > 300 else ''}",
            f"*Judge: {s['reasoning']}*",
            "",
            f"**Multi-Agent Squad** — Quality: {m['quality']}/5 | Latency: {m['latency']}s | Tokens: {m['tokens']} | Cost: ${m['cost']:.6f}",
            f"> {m['answer'][:300]}{'…' if len(m['answer']) > 300 else ''}",
            f"*Judge: {m['reasoning']}*",
            "",
        ]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n  📄 Full report saved to: {report_path}")


if __name__ == "__main__":
    app_state = setup_app_state()
    if app_state:
        run_race(app_state)
