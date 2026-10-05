"""
Week 10 — Specialist sub-agents for the Legal Contracts multi-agent squad.

Each specialist is a focused single LLM call with a narrowed system prompt.
Token usage is captured from response.usage_metadata so the race script can
report input_tokens, output_tokens, and an estimated cost.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

from google import genai
from dotenv import load_dotenv

load_dotenv()
_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
_MODEL = "gemini-2.0-flash"

# ------------------------------------------------------------------
# Gemini 2.0 Flash pricing (as of 2025)
# https://ai.google.dev/pricing
# ------------------------------------------------------------------
_INPUT_COST_PER_TOKEN  = 0.075 / 1_000_000   # $0.075 per 1M input tokens
_OUTPUT_COST_PER_TOKEN = 0.30  / 1_000_000   # $0.30  per 1M output tokens


@dataclass
class AgentResult:
    agent_name: str
    output: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def estimated_cost_usd(self) -> float:
        return (
            self.input_tokens  * _INPUT_COST_PER_TOKEN +
            self.output_tokens * _OUTPUT_COST_PER_TOKEN
        )


def _call_llm(system_prompt: str, user_message: str, agent_name: str) -> AgentResult:
    """Single Gemini call, captures token usage."""
    t0 = time.time()
    response = _client.models.generate_content(
        model=_MODEL,
        contents=f"{system_prompt}\n\n---\n\n{user_message}",
    )
    latency = time.time() - t0

    usage = getattr(response, "usage_metadata", None)
    input_tok  = getattr(usage, "prompt_token_count",     0) or 0
    output_tok = getattr(usage, "candidates_token_count", 0) or 0

    return AgentResult(
        agent_name=agent_name,
        output=response.text.strip(),
        input_tokens=input_tok,
        output_tokens=output_tok,
        latency_s=latency,
    )


# ------------------------------------------------------------------
# Specialist 1 — Clause Retriever
# ------------------------------------------------------------------

_CLAUSE_RETRIEVER_SYSTEM = """\
You are a Clause Retriever specialist in a legal AI team.

Your ONLY job is to extract the verbatim contract clause(s) most relevant to
the question. Do NOT answer the question — just quote the clause text exactly
as it appears and note any section headings or article numbers.

If you cannot find a relevant clause, say: "No relevant clause found."
"""


class ClauseRetrieverAgent:
    """Pulls the exact clause text from the retrieved context."""

    NAME = "ClauseRetriever"

    def run(self, question: str, context: str) -> AgentResult:
        user_msg = (
            f"CONTRACT CONTEXT:\n{context}\n\n"
            f"QUESTION: {question}\n\n"
            "Extract the relevant clause(s) verbatim."
        )
        return _call_llm(_CLAUSE_RETRIEVER_SYSTEM, user_msg, self.NAME)


# ------------------------------------------------------------------
# Specialist 2 — Risk Analyst
# ------------------------------------------------------------------

_RISK_ANALYST_SYSTEM = """\
You are a Risk Analyst specialist in a legal AI team.

You receive a contract clause and a question. Your job is to identify any
risks, obligations, or red-flags in the clause that are relevant to the
question. Focus on: liability, indemnity, IP ownership, termination triggers,
and payment obligations.

Be concise — bullet points preferred.
If there are no notable risks, say: "No significant risks identified."
"""


class RiskAnalystAgent:
    """Flags risks and obligations in the retrieved clause."""

    NAME = "RiskAnalyst"

    def run(self, question: str, clause_text: str) -> AgentResult:
        user_msg = (
            f"CLAUSE TEXT:\n{clause_text}\n\n"
            f"QUESTION: {question}\n\n"
            "Identify risks and obligations."
        )
        return _call_llm(_RISK_ANALYST_SYSTEM, user_msg, self.NAME)


# ------------------------------------------------------------------
# Specialist 3 — Summary Drafter
# ------------------------------------------------------------------

_SUMMARY_DRAFTER_SYSTEM = """\
You are a Summary Drafter specialist in a legal AI team.

You receive:
1. The original user question
2. The raw clause text extracted from the contract
3. A risk analysis of that clause

Your job is to write a clear, plain-English final answer that directly
addresses the question. Cite the clause source if a section is mentioned.
Do not invent any contract terms not present in the provided inputs.
"""


class SummaryDrafterAgent:
    """Synthesises the final answer from clause + risk analysis."""

    NAME = "SummaryDrafter"

    def run(
        self,
        question: str,
        clause_text: str,
        risk_analysis: str,
    ) -> AgentResult:
        user_msg = (
            f"ORIGINAL QUESTION: {question}\n\n"
            f"EXTRACTED CLAUSE:\n{clause_text}\n\n"
            f"RISK ANALYSIS:\n{risk_analysis}\n\n"
            "Write the final plain-English answer."
        )
        return _call_llm(_SUMMARY_DRAFTER_SYSTEM, user_msg, self.NAME)
