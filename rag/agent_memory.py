import os
from mem0 import Memory
from dotenv import load_dotenv

load_dotenv()

_memory_instance = None
fallback_memory = []

def get_agent_memory():
    global _memory_instance
    if _memory_instance is None:
        try:
            config = {
                "llm": {
                    "provider": "gemini",
                    "config": {
                        "api_key": os.getenv("GOOGLE_API_KEY")
                    }
                }
            }
            # Initialize mem0 Memory client
            _memory_instance = Memory.from_config(config_dict=config)
        except Exception as e:
            print(f"Warning: mem0 initialization failed: {e}. Falling back.")
            _memory_instance = "FALLBACK"
    return _memory_instance

def save_long_term_memory(question: str, answer: str):
    """
    Saves an interaction to the agent's long term memory.
    """
    m = get_agent_memory()
    mem_string = f"User asked: {question}. We answered: {answer}"
    if m == "FALLBACK":
        fallback_memory.append(mem_string)
    else:
        try:
            m.add(mem_string, user_id="agent_user")
        except:
            fallback_memory.append(mem_string)

def retrieve_long_term_memory(query: str):
    """
    Retrieves relevant past interactions from memory.
    """
    m = get_agent_memory()
    if m == "FALLBACK":
        return "\n".join(fallback_memory[-3:])
    else:
        try:
            results = m.search(query, user_id="agent_user")
            if results:
                # results is typically a list of dicts with a 'text' or 'memory' key
                texts = [r.get('text', r.get('memory', str(r))) for r in results]
                return "\n".join(texts)
            return ""
        except:
            return "\n".join(fallback_memory[-3:])
