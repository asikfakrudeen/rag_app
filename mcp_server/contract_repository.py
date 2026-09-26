"""
Contract repository MCP server (Track F).

This process offers tools, one resource, and one prompt. It does not call a
model. The legal assistant runs in the host (rag/agent_loop.py) and discovers
these tools over MCP.

The SDK class is MCPServer. Course notes still say FastMCP; that is the same
idea after the Python SDK rename in mcp 2.

Run for the local agent (stdio, spawned by the host):

    python -m mcp_server.contract_repository

Run so another person's agent can call it (HTTP, bearer token required):

    $env:MCP_AUTH_TOKEN = "a-long-random-token"
    python -m mcp_server.contract_repository --http --port 8765

Add another tool by writing another @server.tool() function below.
The agent does not need a code change; it reads the tool list at startup.
"""

from __future__ import annotations

import argparse
import os
import sys

import uvicorn
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations

from rag.vector_store import get_all_documents, get_collection, retrieve

READ_ONLY = ToolAnnotations(
    read_only_hint=True,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=False,
)

server = MCPServer(
    "contract-repository",
    instructions=(
        "Read-only tools over the indexed legal contracts. "
        "No model runs in this server."
    ),
    version="1.0.0",
    log_level="WARNING",
)

def _collection():
    try:
        return get_collection("legal_contracts")
    except Exception as exc:
        raise ToolError(
            "Contract index is not available. Build the index in the app, then retry. "
            f"Detail: {exc}"
        ) from exc


def _catalog_lines() -> str:
    collection = _collection()
    _ids, docs, metas = get_all_documents(collection)
    if not docs:
        raise ToolError("The repository is empty. Upload a contract and build the index.")

    counts: dict[str, int] = {}
    for meta in metas or []:
        source = (meta or {}).get("source", "unknown")
        counts[source] = counts.get(source, 0) + 1

    lines = [f"- {name} ({count} chunks)" for name, count in sorted(counts.items())]
    return "Contracts in the repository:\n" + "\n".join(lines)


def _format_hits(documents: list, metadatas: list) -> str:
    if not documents:
        return "No matching contract terms found."
    chunks = []
    for doc, meta in zip(documents, metadatas):
        meta = meta or {}
        source = meta.get("source", "unknown")
        page = meta.get("page", "?")
        chunks.append(f"[{source} p.{page}]\n{doc}")
    return "\n\n".join(chunks)


@server.tool(annotations=READ_ONLY)
def search_contract(query: str) -> str:
    """Search indexed legal contracts and return the closest matching clauses."""
    query = (query or "").strip()
    if not query:
        raise ToolError("Query is empty. Send the clause topic or question to look up.")

    collection = _collection()
    try:
        results = retrieve(collection, query, top_k=3)
    except Exception as exc:
        raise ToolError(
            "Search failed. Retry with a shorter query. "
            f"Detail: {exc}"
        ) from exc

    found_docs = (results.get("documents") or [[]])[0]
    found_metas = (results.get("metadatas") or [[]])[0]
    return _format_hits(found_docs, found_metas)


@server.tool(annotations=READ_ONLY)
def list_contracts() -> str:
    """List the contract files stored in the repository and how many chunks each has."""
    return _catalog_lines()


@server.resource(
    "contract://catalog",
    name="contract_catalog",
    description="Read-only catalog of contracts in the repository.",
    mime_type="text/plain",
)
def contract_catalog() -> str:
    """Resource: the same catalog list_contracts returns, for clients that read resources."""
    try:
        return list_contracts()
    except ToolError as exc:
        return str(exc)


@server.prompt(description="Ask what the indexed contracts say about a topic.")
def ask_about_clause(topic: str) -> str:
    """Prompt another agent can fetch before it asks its own model."""
    topic = (topic or "").strip() or "the main obligations"
    return (
        f"Using only the contract repository, what does the agreement say about {topic}? "
        "Quote the clause and name the source file."
    )


class _BearerGate:
    """Reject HTTP calls that do not present the shared bearer token.

    Stdio is a local process on this machine, so it does not use this gate.
    HTTP is how someone else's agent connects, so it does.
    """

    def __init__(self, app, token: str):
        self.app = app
        self._expected = f"Bearer {token}".encode()

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            headers = dict(scope.get("headers") or [])
            if headers.get(b"authorization") != self._expected:
                body = b'{"error":"unauthorized","hint":"Send Authorization: Bearer <MCP_AUTH_TOKEN>"}'
                await send(
                    {
                        "type": "http.response.start",
                        "status": 401,
                        "headers": [
                            (b"content-type", b"application/json"),
                            (b"content-length", str(len(body)).encode()),
                        ],
                    }
                )
                await send({"type": "http.response.body", "body": body})
                return
        await self.app(scope, receive, send)


def serve_http(host: str, port: int) -> None:
    token = os.getenv("MCP_AUTH_TOKEN", "").strip()
    if not token:
        print(
            "MCP_AUTH_TOKEN is required for HTTP. Refusing to start an open server.",
            file=sys.stderr,
        )
        sys.exit(1)

    app = _BearerGate(server.streamable_http_app(host=host), token)
    print(
        f"Contract repository MCP at http://{host}:{port}/mcp (bearer token required).",
        file=sys.stderr,
    )
    uvicorn.run(app, host=host, port=port, log_level="warning")


def main() -> None:
    parser = argparse.ArgumentParser(description="Legal contract repository MCP server")
    parser.add_argument(
        "--http",
        action="store_true",
        help="Serve over streamable HTTP instead of stdio",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    if args.http:
        serve_http(args.host, args.port)
    else:
        server.run(transport="stdio")


if __name__ == "__main__":
    main()
