"""
MCP host for the legal assistant.

The model runs in this process. MCP servers are separate processes that only
offer tools. This module discovers those tools, reviews them, and calls the
ones that pass.

A new tool on a server shows up here on the next connection. Agent code does
not keep a list of tool names.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import threading
from dataclasses import dataclass, field

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_DANGEROUS_NAME = re.compile(
    r"(shell|exec|eval|subprocess|delete|drop|write_file|send_mail|exfil|credential)",
    re.IGNORECASE,
)
_INJECTION_IN_DESCRIPTION = re.compile(
    r"(ignore (previous|all|prior) instruction|"
    r"disregard (previous|all|prior)|"
    r"forget (everything|all)|"
    r"you are now|"
    r"execute any command|"
    r"act as )",
    re.IGNORECASE,
)
_EMPTY_INPUT = {"", "none", "n/a", "null", "-", "na"}


@dataclass
class ToolOffer:
    """A tool as advertised by a server, before we decide to trust it."""

    name: str
    description: str
    input_schema: dict
    server_name: str = ""
    read_only: bool | None = None
    destructive: bool | None = None


@dataclass
class DiscoveredTool:
    name: str
    description: str
    input_schema: dict
    server_name: str
    trusted: bool
    review_reason: str


@dataclass
class StdioServerSpec:
    """A local MCP server this host will spawn."""

    module: str
    cwd: str = PROJECT_ROOT

    def parameters(self) -> StdioServerParameters:
        return StdioServerParameters(
            command=sys.executable,
            args=["-m", self.module],
            cwd=self.cwd,
            env=os.environ.copy(),
        )


@dataclass
class HttpServerSpec:
    """Someone else's MCP server, reached over HTTP."""

    url: str
    token: str | None = None
    name: str = "remote"


def review_tool(offer: ToolOffer) -> tuple[bool, str]:
    """Decide whether a discovered tool may be shown to the model.

    This runs before any call. A second server can be attached without
    editing the agent; tools that look dangerous or instructional are dropped.
    """
    if offer.destructive:
        return False, "server marked this tool as destructive"
    if _DANGEROUS_NAME.search(offer.name or ""):
        return False, "tool name looks like a dangerous capability"
    if _INJECTION_IN_DESCRIPTION.search(offer.description or ""):
        return False, "tool description contains instructions aimed at the model"
    if offer.read_only is False and offer.destructive is not False:
        return False, "tool is not marked read-only"
    return True, "name and description accepted"


def arguments_for(tool: DiscoveredTool, action_input: str) -> dict:
    """Map the agent's Action Input line onto the tool's JSON schema.

    Works for a no-argument tool, a single string argument, or a JSON object,
    so a new tool does not need a new branch in the agent.
    """
    schema = tool.input_schema or {}
    props = schema.get("properties") or {}
    required = schema.get("required") or []
    text = (action_input or "").strip().strip('"').strip("'")

    if not props:
        return {}

    if text.startswith("{") and text.endswith("}"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            return parsed

    if len(props) == 1:
        key = next(iter(props))
        if text.lower() in _EMPTY_INPUT and key not in required:
            return {}
        return {key: text}

    if text.lower() in _EMPTY_INPUT:
        return {}
    if required:
        return {required[0]: text}
    return {next(iter(props)): text}


async def call_http_tool(url: str, token: str | None, tool_name: str, arguments: dict) -> tuple[list[str], str]:
    """What another person's agent does: connect, discover, call. No local model."""
    import httpx2

    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    async with httpx2.AsyncClient(headers=headers) as http:
        async with streamable_http_client(url, http_client=http) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                listed = await session.list_tools()
                names = [tool.name for tool in listed.tools]
                result = await session.call_tool(tool_name, arguments)
                return names, _result_text(result)


def _result_text(result) -> str:
    parts = []
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    body = "\n".join(parts).strip() or "(empty tool result)"
    if getattr(result, "is_error", False):
        return f"Tool error (recoverable): {body}"
    return body


def _schema_dict(schema) -> dict:
    if schema is None:
        return {}
    if isinstance(schema, dict):
        return schema
    if hasattr(schema, "model_dump"):
        return schema.model_dump(exclude_none=True)
    return {}


def _offer_from_tool(tool, server_name: str) -> ToolOffer:
    annotations = getattr(tool, "annotations", None)
    return ToolOffer(
        name=tool.name,
        description=tool.description or "",
        input_schema=_schema_dict(getattr(tool, "input_schema", None)),
        server_name=server_name,
        read_only=getattr(annotations, "read_only_hint", None) if annotations else None,
        destructive=getattr(annotations, "destructive_hint", None) if annotations else None,
    )


def default_servers() -> list:
    """Servers the agent connects to. Override with MCP_REMOTE_URL to bolt one on."""
    servers: list = [StdioServerSpec(module="mcp_server.contract_repository")]
    remote = os.getenv("MCP_REMOTE_URL", "").strip()
    if remote:
        servers.append(
            HttpServerSpec(
                url=remote,
                token=os.getenv("MCP_AUTH_TOKEN", "").strip() or None,
            )
        )
    return servers


class MCPHost:
    """Keeps MCP sessions open on a background thread and calls discovered tools."""

    def __init__(self, servers: list | None = None):
        self.servers = servers if servers is not None else default_servers()
        self.tools: list[DiscoveredTool] = []
        self._sessions: dict[str, ClientSession] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._closed: asyncio.Event | None = None
        self._error: BaseException | None = None

    def open(self) -> "MCPHost":
        self._thread = threading.Thread(target=self._thread_main, name="mcp-host", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=120):
            self.close()
            raise TimeoutError("MCP server did not finish the handshake within 120 seconds.")
        if self._error:
            self.close()
            raise RuntimeError(f"MCP connection failed: {self._error}") from self._error
        return self

    def close(self) -> None:
        loop = self._loop
        closed = self._closed
        if loop is not None and closed is not None and loop.is_running():
            loop.call_soon_threadsafe(closed.set)
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=10)

    def trusted_tools(self) -> list[DiscoveredTool]:
        return [tool for tool in self.tools if tool.trusted]

    def call_tool(self, name: str, action_input: str) -> str:
        trusted = {tool.name: tool for tool in self.trusted_tools()}
        tool = trusted.get(name)
        if tool is None:
            available = ", ".join(sorted(trusted)) or "(none)"
            return (
                f"Tool '{name}' is not available. "
                f"Discovered tools: {available}."
            )
        if self._loop is None:
            raise RuntimeError("MCP host is not open.")
        arguments = arguments_for(tool, action_input)
        future = asyncio.run_coroutine_threadsafe(self._call(name, arguments), self._loop)
        try:
            return future.result(timeout=120)
        except Exception as exc:
            return f"Tool error (recoverable): {name} failed before a result came back. Detail: {exc}"

    def _thread_main(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except Exception as exc:
            self._error = exc
            self._ready.set()
        finally:
            self._loop.close()

    async def _serve(self) -> None:
        from contextlib import AsyncExitStack

        import httpx2

        self._closed = asyncio.Event()
        try:
            async with AsyncExitStack() as stack:
                for spec in self.servers:
                    if isinstance(spec, StdioServerSpec):
                        read, write = await stack.enter_async_context(stdio_client(spec.parameters()))
                        server_name = spec.module.rsplit(".", 1)[-1]
                    elif isinstance(spec, HttpServerSpec):
                        headers = {}
                        if spec.token:
                            headers["Authorization"] = f"Bearer {spec.token}"
                        http = await stack.enter_async_context(httpx2.AsyncClient(headers=headers))
                        read, write = await stack.enter_async_context(
                            streamable_http_client(spec.url, http_client=http)
                        )
                        server_name = spec.name
                    else:
                        raise TypeError(f"Unknown MCP server spec: {spec!r}")

                    session = await stack.enter_async_context(ClientSession(read, write))
                    await session.initialize()
                    listed = await session.list_tools()
                    for tool in listed.tools:
                        offer = _offer_from_tool(tool, server_name)
                        trusted, reason = review_tool(offer)
                        discovered = DiscoveredTool(
                            name=offer.name,
                            description=offer.description,
                            input_schema=offer.input_schema,
                            server_name=server_name,
                            trusted=trusted,
                            review_reason=reason,
                        )
                        if offer.name in self._sessions:
                            discovered.trusted = False
                            discovered.review_reason = "duplicate tool name from another server"
                        else:
                            self._sessions[offer.name] = session
                        self.tools.append(discovered)
                self._ready.set()
                await self._closed.wait()
        except Exception as exc:
            self._error = exc
            self._ready.set()
            raise

    async def _call(self, name: str, arguments: dict) -> str:
        session = self._sessions[name]
        result = await session.call_tool(name, arguments)
        return _result_text(result)


@dataclass
class StaticToolBackend:
    """In-memory tools for tests that should not spawn a server."""

    tools: list[DiscoveredTool] = field(default_factory=list)
    observation: str = ""

    def trusted_tools(self) -> list[DiscoveredTool]:
        return [tool for tool in self.tools if tool.trusted]

    def call_tool(self, name: str, action_input: str) -> str:
        known = {tool.name for tool in self.trusted_tools()}
        if name not in known:
            available = ", ".join(sorted(known)) or "(none)"
            return f"Tool '{name}' is not available. Discovered tools: {available}."
        return self.observation

    def close(self) -> None:
        return None
