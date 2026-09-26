"""
Print the three MCP messages that make the protocol concrete.

    python scripts/show_mcp_handshake.py

The host speaks JSON-RPC. The contract repository answers. No model is called.
"""

import asyncio
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from mcp import ClientSession
from mcp.client.stdio import stdio_client

from rag.mcp_host import StdioServerSpec


def _dump(label: str, payload) -> None:
    print(f"\n--- {label} ---")
    if hasattr(payload, "model_dump_json"):
        print(payload.model_dump_json(indent=2))
    else:
        print(payload)


async def main() -> None:
    spec = StdioServerSpec(module="mcp_server.contract_repository")
    async with stdio_client(spec.parameters()) as (read, write):
        async with ClientSession(read, write) as session:
            initialized = await session.initialize()
            _dump("JSON-RPC initialize (handshake)", initialized)

            tools = await session.list_tools()
            _dump("JSON-RPC tools/list (discovery)", tools)

            listed = await session.call_tool("list_contracts", {})
            _dump("JSON-RPC tools/call list_contracts", listed)

    print(
        "\nThe server info above names contract-repository. "
        "That process returned tool results. It did not run the legal assistant."
    )


if __name__ == "__main__":
    asyncio.run(main())
