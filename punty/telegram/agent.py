"""Claude agent with tool use for PuntyAI server management via Telegram."""

import logging
from typing import Optional

from anthropic import AsyncAnthropic

from punty.telegram.tools import (
    tool_bash,
    tool_edit_file,
    tool_list_files,
    tool_query_db,
    tool_read_file,
    tool_update_track_condition,
    tool_write_file,
)

logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-5-20250929"
MAX_TURNS = 25
MAX_TOKENS = 4096

SYSTEM_PROMPT = """\
You are PuntyAI's server assistant, talking to Rochey (the owner) via Telegram.
You have tools to manage the PuntyAI server at /opt/puntyai.

Keep responses concise — Rochey is reading on his phone.

## Project
Python 3.11 FastAPI app. SQLAlchemy 2.0 async + SQLite. Uvicorn behind Caddy.
AI horse racing tips generator with scrapers, content generation, and delivery (Twitter/Facebook).

## Key Paths
- /opt/puntyai/ — project root
- /opt/puntyai/data/punty.db — SQLite database
- /opt/puntyai/punty/ — source code
- /opt/puntyai/prompts/ — AI prompt templates
- /opt/puntyai/venv/ — Python virtualenv

## Key Files
- punty/main.py — App entry, lifespan
- punty/ai/generator.py — Content generation
- punty/results/monitor.py — Background results polling
- punty/scrapers/ — Racing data scrapers (racing_com.py, tab_scraper.py)
- punty/scheduler/ — APScheduler jobs (manager.py, automation.py)
- punty/models/ — DB models (meeting.py, content.py, pick.py, settings.py)
- punty/delivery/ — Twitter, Facebook posting
- punty/web/routes.py — Page routes
- punty/api/ — API endpoints

## DB Tables
meetings, races, runners, content, picks, app_settings, race_assessments, \
live_updates, settings_audit, analysis_weights, pattern_insights

## Common Tasks
- Check logs: journalctl -u puntyai --no-pager -n 50
- Restart: systemctl restart puntyai
- Deploy: cd /opt/puntyai && git fetch origin master && git checkout -f origin/master \
&& chmod 666 prompts/*.md && systemctl restart puntyai
- Run tests: cd /opt/puntyai && source venv/bin/activate && pytest tests/unit/ -v
- DB query: Use the query_db tool with SELECT statements

## Pick Types
- selection: Individual horse pick (win/place/each_way/saver_win)
- exotic: Trifecta/Exacta/Quinella/First4
- sequence: Quaddie/Big6 (skinny/balanced/wide variants)
- big3: Top 3 picks across meeting
- big3_multi: Multi bet on Big 3

## Content Types
- early_mail: Pre-race tips
- meeting_wrapup: Post-race summary
- race_result: Individual race result\
"""

TOOL_DEFINITIONS = [
    {
        "name": "bash",
        "description": (
            "Run a bash command on the PuntyAI server. Use for checking logs, "
            "restarting services, git operations, running scripts. "
            "Working directory is /opt/puntyai."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default 30, max 120)",
                },
            },
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": (
            "Read a file on the server. Paths can be absolute or relative to /opt/puntyai."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path",
                },
                "max_lines": {
                    "type": "integer",
                    "description": "Max lines to read (default 500)",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file. Creates parent directories if needed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path",
                },
                "content": {
                    "type": "string",
                    "description": "File content",
                },
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": (
            "Find and replace text in a file. Replaces the first occurrence of old_text "
            "with new_text."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path",
                },
                "old_text": {
                    "type": "string",
                    "description": "Text to find (exact match)",
                },
                "new_text": {
                    "type": "string",
                    "description": "Replacement text",
                },
            },
            "required": ["path", "old_text", "new_text"],
        },
    },
    {
        "name": "list_files",
        "description": "List files in a directory with optional glob pattern.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path (default: /opt/puntyai)",
                },
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern (default: *)",
                },
            },
        },
    },
    {
        "name": "update_track_condition",
        "description": (
            "Update the track condition for a meeting and all its races. "
            "Use when Rochey says the track condition is wrong and needs correcting. "
            "Example conditions: 'Good 4', 'Soft 5', 'Heavy 8', 'Firm 1'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "venue": {
                    "type": "string",
                    "description": "Track/venue name, e.g. 'Sale', 'Flemington'",
                },
                "date": {
                    "type": "string",
                    "description": "Meeting date in YYYY-MM-DD format",
                },
                "condition": {
                    "type": "string",
                    "description": "New track condition, e.g. 'Good 4', 'Soft 5'",
                },
            },
            "required": ["venue", "date", "condition"],
        },
    },
    {
        "name": "query_db",
        "description": (
            "Run a read-only SQL query against the PuntyAI SQLite database. "
            "Only SELECT/WITH statements allowed. "
            "Tables: meetings, races, runners, content, picks, app_settings, "
            "race_assessments, live_updates, pattern_insights."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "SQL SELECT query",
                },
            },
            "required": ["sql"],
        },
    },
]

# Map tool names to implementations
TOOL_HANDLERS = {
    "bash": lambda inp: tool_bash(inp["command"], inp.get("timeout", 30)),
    "read_file": lambda inp: tool_read_file(inp["path"], inp.get("max_lines", 500)),
    "write_file": lambda inp: tool_write_file(inp["path"], inp["content"]),
    "edit_file": lambda inp: tool_edit_file(inp["path"], inp["old_text"], inp["new_text"]),
    "list_files": lambda inp: tool_list_files(inp.get("path", "/opt/puntyai"), inp.get("pattern", "*")),
    "update_track_condition": lambda inp: tool_update_track_condition(inp["venue"], inp["date"], inp["condition"]),
    "query_db": lambda inp: tool_query_db(inp["sql"]),
}


class ClaudeAgent:
    """Claude-powered agent that can manage the PuntyAI server."""

    def __init__(self, api_key: str):
        self.client = AsyncAnthropic(api_key=api_key)
        self.conversations: dict[int, list] = {}

    async def chat(self, chat_id: int, user_message) -> str:
        """Process a user message through the agentic tool-use loop.

        user_message can be a string or a list of content blocks (for images).
        Returns the final text response from Claude.
        """
        if chat_id not in self.conversations:
            self.conversations[chat_id] = []

        messages = self.conversations[chat_id]
        messages.append({"role": "user", "content": user_message})

        for turn in range(MAX_TURNS):
            try:
                response = await self.client.messages.create(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    system=SYSTEM_PROMPT,
                    tools=TOOL_DEFINITIONS,
                    messages=messages,
                )
            except Exception as e:
                logger.error(f"Anthropic API error: {e}")
                # Remove the user message we just added so conversation stays clean
                if messages and messages[-1]["role"] == "user":
                    messages.pop()
                return f"Claude API error: {e}"

            # Add assistant response to history
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                # Extract text from response
                text_parts = []
                for block in response.content:
                    if hasattr(block, "text"):
                        text_parts.append(block.text)
                return "\n".join(text_parts) if text_parts else "(no response)"

            # Process tool calls
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = await self._run_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
                    logger.info(f"Tool {block.name}: {len(result)} chars result")

            messages.append({"role": "user", "content": tool_results})

        return "(max tool turns reached — please try a simpler request)"

    def clear_history(self, chat_id: int):
        """Clear conversation history for a chat."""
        self.conversations.pop(chat_id, None)

    async def _run_tool(self, name: str, tool_input: dict) -> str:
        """Dispatch a tool call to the appropriate implementation."""
        handler = TOOL_HANDLERS.get(name)
        if not handler:
            return f"Unknown tool: {name}"

        try:
            return await handler(tool_input)
        except Exception as e:
            logger.error(f"Tool {name} error: {e}")
            return f"Tool error: {e}"

    async def close(self):
        """Close the Anthropic client."""
        await self.client.close()
