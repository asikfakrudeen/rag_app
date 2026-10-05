"""
Week 10 — Orchestrator for the Legal Contracts multi-agent squad.

Architecture:
    OrchestratorAgent
    ├── ClauseRetrieverAgent   (pulls verbatim clause from contract context)
    ├── RiskAnalystAgent       (flags obligations / risks in the clause)
    └── SummaryDrafterAgent    (drafts the final answer from clause + risks)

The orchestrator uses the same MCP-backed contract retrieval as the single
agent (via the app_state dict), then delegates each reasoning step to a
specialist instead of asking one model to do everything.

Return value mirrors run_agent_loop() for easy side-by-side comparison.
"""

from __future__ import annotations

import time
from typing import Any

from rag.hybrid_retriever import hybrid_retrieve
from rag.vector_store import get_collection
from rag.multi_agent.specialists import (
    AgentResult,
    ClauseRetrieverAgent,
    RiskAnalystAgent,
    SummaryDrafterAgent,
)

# ------------------------------------------------------------------
# Cost constants (mirror specialists.py for combined totals)
# ------------------------------------------------------------------
_INPUT_COST_PER_TOKEN  = 0.075 / 1_000_000
_OUTPUT_COST_PER_TOKEN = 0.30  / 1_000_000


def _format_context(results: dict) -> str:
    """Turn hybrid_search results into a context string for specialists."""
    docs      = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    parts = []
    for doc, meta in zip(docs, metadatas):
        parts.append(
            f"SOURCE: {meta.get('source', 'unknown')} | PAGE: {meta.get('page', '?')}\n{doc}"
        )
    return "\n\n---\n\n".join(parts) or "No contract context retrieved."


class OrchestratorAgent:
    """
    Manager agent: retrieves context then delegates to three specialists
    in sequence and assembles the final answer.
    """

    def run(self, question: str, app_state: dict[str, Any]) -> dict:
        """
        Returns a dict compatible with run_agent_loop() plus extra metrics:
            answer, memory, tools,
            latency_s, input_tokens, output_tokens, total_tokens,
            estimated_cost_usd, agents_called
        """
        if not app_state.get("indexed"):
            raise ValueError("Please build the index first.")

        wall_start = time.time()
        step_log   = []
        agents_called: list[str] = []

        # ── Step 0: Retrieve contract context ─────────────────────────
        step_log.append(f"[Orchestrator] Retrieving context for: {question!r}")
        collection = get_collection("legal_contracts")
        results = hybrid_retrieve(
            collection,
            app_state["bm25_index"],
            app_state["all_ids"],
            app_state["all_docs"],
            app_state["all_metas"],
            question,
            top_k=5,
        )
        context = _format_context(results)
        step_log.append(f"[Orchestrator] Retrieved {len(results.get('documents', [[]])[0])} chunks.")

        # ── Step 1: Clause Retriever ───────────────────────────────────
        step_log.append("[Orchestrator] → ClauseRetrieverAgent")
        clause_result: AgentResult = ClauseRetrieverAgent().run(question, context)
        agents_called.append(clause_result.agent_name)
        step_log.append(f"[ClauseRetriever] Output: {clause_result.output[:200]}…")

        # ── Step 2: Risk Analyst ───────────────────────────────────────
        step_log.append("[Orchestrator] → RiskAnalystAgent")
        risk_result: AgentResult = RiskAnalystAgent().run(question, clause_result.output)
        agents_called.append(risk_result.agent_name)
        step_log.append(f"[RiskAnalyst] Output: {risk_result.output[:200]}…")

        # ── Step 3: Summary Drafter ────────────────────────────────────
        step_log.append("[Orchestrator] → SummaryDrafterAgent")
        summary_result: AgentResult = SummaryDrafterAgent().run(
            question, clause_result.output, risk_result.output
        )
        agents_called.append(summary_result.agent_name)
        step_log.append(f"[SummaryDrafter] Final answer: {summary_result.output[:200]}…")

        # ── Aggregate metrics ──────────────────────────────────────────
        wall_latency   = time.time() - wall_start
        total_input    = clause_result.input_tokens  + risk_result.input_tokens  + summary_result.input_tokens
        total_output   = clause_result.output_tokens + risk_result.output_tokens + summary_result.output_tokens
        total_tokens   = total_input + total_output
        total_cost     = (
            total_input  * _INPUT_COST_PER_TOKEN +
            total_output * _OUTPUT_COST_PER_TOKEN
        )

        memory = "\n".join(step_log)

        return {
            # Compatible with single-agent interface
            "answer":  summary_result.output,
            "memory":  memory,
            "tools":   agents_called,
            "evidence": results.get("documents", [[]])[0],

            # Extra metrics for the race
            "latency_s":           round(wall_latency, 3),
            "input_tokens":        total_input,
            "output_tokens":       total_output,
            "total_tokens":        total_tokens,
            "estimated_cost_usd":  round(total_cost, 8),
            "agents_called":       agents_called,
        }
