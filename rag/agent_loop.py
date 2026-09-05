import os
import re
from google import genai
from rag.vector_store import get_collection
from rag.hybrid_retriever import hybrid_retrieve
from rag.tracer import log_trace
from rag.agent_memory import retrieve_long_term_memory, save_long_term_memory
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def run_agent_loop(question: str, app_state: dict, max_iterations: int = 5):
    """
    Custom Pure Python ReAct Agent Loop.
    Limits infinite loops using max_iterations budget.
    """
    if not app_state.get("indexed"):
        raise ValueError("Please build the index first.")

    long_term_context = retrieve_long_term_memory(question)

    system_prompt = f"""You are a Legal Assistant Agent. You answer questions strictly based on contracts.
You have access to the following tool:
- search_contract(query): Searches the legal contracts for the given query.

PAST LONG-TERM INTERACTIONS:
{long_term_context if long_term_context else "No prior conversations."}

You must follow this exact format:
Thought: ... (think about what you need to do)
Action: ... (the action to take, exactly one of [search_contract])
Action Input: ... (the exact search string to look up)
Observation: ... (the result of the action, provided by the system)
... (this loop can repeat)
Thought: I know the answer
Final Answer: the final answer to the original input question

Do not invent facts. If the answer is not in the contract, admit you do not know.
"""

    memory = f"Original Question: {question}\n"
    
    for i in range(max_iterations):
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=f"{system_prompt}\n---\nPAST INTERACTION LOG:\n{memory}\nPlease output your next Thought/Action (or Final Answer) now."
        )
        
        reply = response.text.strip()
        memory += f"{reply}\n"
        
        if "Final Answer:" in reply:
            # We found the final answer
            answer_splits = reply.split("Final Answer:")
            final_ans = answer_splits[-1].strip() if len(answer_splits) > 1 else reply
            
            # Save interaction to long-term memory
            save_long_term_memory(question, final_ans)
            
            # Simple trace bypass for agents - no concrete chunk distances provided directly
            log_trace(question, [{"source": "Agent Memory", "page": 0, "distance": 0.0, "text": memory}], final_ans)
            return {"answer": final_ans, "evidence": [], "memory": memory}
            
        action_match = re.search(r"Action:\s*(.*?)\n", reply + "\n")
        input_match = re.search(r"Action Input:\s*(.*?)\n", reply + "\n")
        
        if action_match and input_match:
            action = action_match.group(1).strip()
            action_input = input_match.group(1).strip()
            
            if "search" in action.lower():
                collection = get_collection("legal_contracts")
                results = hybrid_retrieve(
                    collection=collection,
                    bm25_index=app_state["bm25_index"],
                    all_ids=app_state["all_ids"],
                    all_documents=app_state["all_docs"],
                    all_metadatas=app_state["all_metas"],
                    query=action_input,
                    top_k=3,
                    use_hybrid=True,
                    use_rerank=True
                )
                docs = results["documents"][0]
                if docs:
                    obs = "\n".join(docs)
                else:
                    obs = "No matching contract terms found."
                    
                memory += f"Observation: {obs}\n"
            else:
                memory += f"Observation: Tool '{action}' does not exist.\n"
        else:
            memory += "Observation: Format error. Ensure you provide 'Thought:', 'Action:', and 'Action Input:'. If you are done, provide 'Final Answer:'.\n"

    return {"answer": "Agent failed due to execution budget constraint.", "evidence": [], "memory": memory}
