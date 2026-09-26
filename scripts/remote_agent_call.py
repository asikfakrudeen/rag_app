"""
A second agent. It is not the legal assistant.

It connects to the contract repository over HTTP, discovers tools, and calls
one. The repository must already be running with MCP_AUTH_TOKEN set:

    $env:MCP_AUTH_TOKEN = "a-long-random-token"
    python -m mcp_server.contract_repository --http --port 8765

Then, in another terminal, with the same token:

    python scripts/remote_agent_call.py
    python scripts/remote_agent_call.py --tool search_contract --query "termination notice"
"""

import argparse
import asyncio
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from rag.mcp_host import call_http_tool


async def call_remote(url: str, token: str | None, tool_name: str, arguments: dict) -> str:
    names, text = await call_http_tool(url, token, tool_name, arguments)
    print(f"Discovered tools: {', '.join(names)}")
    if tool_name not in names:
        raise SystemExit(f"{tool_name} is not on this server. Discovered: {', '.join(names)}")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Call the contract repository as another agent")
    parser.add_argument("--url", default=os.getenv("MCP_REMOTE_URL", "http://127.0.0.1:8765/mcp"))
    parser.add_argument("--tool", default="list_contracts")
    parser.add_argument("--query", default="")
    args = parser.parse_args()

    token = os.getenv("MCP_AUTH_TOKEN", "").strip() or None
    arguments = {"query": args.query} if args.tool == "search_contract" else {}
    if args.tool == "search_contract" and not args.query:
        arguments = {"query": "termination notice"}

    try:
        text = asyncio.run(call_remote(args.url, token, args.tool, arguments))
    except Exception as exc:
        print(f"Call failed: {exc}")
        sys.exit(1)
    print(text)


if __name__ == "__main__":
    main()
