"""Tests for Telegram bot, Claude agent, and server tools."""

import asyncio
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


# ── Message Chunking Tests ───────────────────────────────────────────────────


class TestChunkMessage:
    """Test TelegramBot._chunk_message() splitting logic."""

    def test_short_message_single_chunk(self):
        from punty.telegram.bot import TelegramBot

        result = TelegramBot._chunk_message("Hello world")
        assert result == ["Hello world"]

    def test_empty_message(self):
        from punty.telegram.bot import TelegramBot

        result = TelegramBot._chunk_message("")
        assert result == ["(empty response)"]

    def test_exact_limit_no_split(self):
        from punty.telegram.bot import TelegramBot

        text = "x" * 4096
        result = TelegramBot._chunk_message(text)
        assert len(result) == 1
        assert result[0] == text

    def test_splits_on_paragraph_boundary(self):
        from punty.telegram.bot import TelegramBot

        para1 = "A" * 2000
        para2 = "B" * 2000
        para3 = "C" * 2000
        text = f"{para1}\n\n{para2}\n\n{para3}"

        result = TelegramBot._chunk_message(text)
        assert len(result) >= 2
        for chunk in result:
            assert len(chunk) <= 4096

    def test_splits_on_newline_if_no_paragraph(self):
        from punty.telegram.bot import TelegramBot

        lines = ["Line " + str(i) for i in range(1000)]
        text = "\n".join(lines)

        result = TelegramBot._chunk_message(text)
        assert len(result) >= 2
        for chunk in result:
            assert len(chunk) <= 4096

    def test_hard_split_no_newlines(self):
        from punty.telegram.bot import TelegramBot

        text = "x" * 10000
        result = TelegramBot._chunk_message(text)
        assert len(result) >= 3
        for chunk in result:
            assert len(chunk) <= 4096


# ── Authorization Tests ──────────────────────────────────────────────────────


class TestAuthorization:
    """Test TelegramBot._check_authorized()."""

    def test_correct_owner_passes(self):
        from punty.telegram.bot import TelegramBot

        bot = TelegramBot(None)
        bot._owner_id = 12345

        update = MagicMock()
        update.effective_user.id = 12345

        assert bot._check_authorized(update) is True

    def test_wrong_user_rejected(self):
        from punty.telegram.bot import TelegramBot

        bot = TelegramBot(None)
        bot._owner_id = 12345

        update = MagicMock()
        update.effective_user.id = 99999

        assert bot._check_authorized(update) is False

    def test_no_user_rejected(self):
        from punty.telegram.bot import TelegramBot

        bot = TelegramBot(None)
        bot._owner_id = 12345

        update = MagicMock()
        update.effective_user = None

        assert bot._check_authorized(update) is False


# ── Tool Dispatch Tests ──────────────────────────────────────────────────────


class TestToolDispatch:
    """Test ClaudeAgent._run_tool() dispatching."""

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self):
        from punty.telegram.agent import ClaudeAgent

        agent = ClaudeAgent(api_key="test-key")
        result = await agent._run_tool("nonexistent_tool", {})
        assert "Unknown tool" in result

    @pytest.mark.asyncio
    async def test_bash_dispatches(self):
        from punty.telegram.agent import ClaudeAgent

        agent = ClaudeAgent(api_key="test-key")

        async def mock_bash(cmd, timeout=30):
            return f"ran: {cmd}"

        with patch("punty.telegram.agent.tool_bash", mock_bash):
            result = await agent._run_tool("bash", {"command": "echo hello"})
            assert result == "ran: echo hello"

    @pytest.mark.asyncio
    async def test_read_file_dispatches(self):
        from punty.telegram.agent import ClaudeAgent

        agent = ClaudeAgent(api_key="test-key")

        async def mock_read(path, max_lines=500):
            return f"contents of {path}"

        with patch("punty.telegram.agent.tool_read_file", mock_read):
            result = await agent._run_tool("read_file", {"path": "/tmp/test.py"})
            assert result == "contents of /tmp/test.py"

    @pytest.mark.asyncio
    async def test_query_db_dispatches(self):
        from punty.telegram.agent import ClaudeAgent

        agent = ClaudeAgent(api_key="test-key")

        async def mock_query(sql):
            return f"results for: {sql}"

        with patch("punty.telegram.agent.tool_query_db", mock_query):
            result = await agent._run_tool("query_db", {"sql": "SELECT 1"})
            assert result == "results for: SELECT 1"


# ── Tool Implementation Tests ────────────────────────────────────────────────


class TestToolBash:
    """Test tool_bash() execution."""

    @pytest.mark.asyncio
    async def test_simple_command(self):
        from punty.telegram.tools import tool_bash

        # Override cwd for test environment
        with patch("punty.telegram.tools.PROJECT_ROOT", "."):
            result = await tool_bash("echo hello")
        assert "hello" in result

    @pytest.mark.asyncio
    async def test_timeout_returns_message(self):
        from punty.telegram.tools import tool_bash

        with patch("punty.telegram.tools.PROJECT_ROOT", "."):
            result = await tool_bash("python3 -c \"import time; time.sleep(10)\"", timeout=1)
        assert "timed out" in result.lower()

    @pytest.mark.asyncio
    async def test_failed_command_includes_error(self):
        from punty.telegram.tools import tool_bash

        with patch("punty.telegram.tools.PROJECT_ROOT", "."):
            result = await tool_bash("python3 -c \"import sys; sys.exit(1)\"")
        assert "exit code" in result.lower()

    @pytest.mark.asyncio
    async def test_output_truncation(self):
        from punty.telegram.tools import tool_bash

        with patch("punty.telegram.tools.PROJECT_ROOT", "."):
            result = await tool_bash("python3 -c \"print('x' * 20000)\"")
        assert len(result) <= 11000  # MAX_OUTPUT_LENGTH + truncation notice


class TestToolReadFile:
    """Test tool_read_file()."""

    @pytest.mark.asyncio
    async def test_reads_existing_file(self, tmp_path):
        from punty.telegram.tools import tool_read_file

        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")

        result = await tool_read_file(str(test_file))
        assert "hello world" in result

    @pytest.mark.asyncio
    async def test_missing_file_returns_error(self):
        from punty.telegram.tools import tool_read_file

        result = await tool_read_file("/tmp/nonexistent_file_xyz_123.txt")
        assert "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_line_limit(self, tmp_path):
        from punty.telegram.tools import tool_read_file

        test_file = tmp_path / "big.txt"
        test_file.write_text("\n".join(f"line {i}" for i in range(1000)))

        result = await tool_read_file(str(test_file), max_lines=10)
        assert "line 0" in result
        assert "1000 lines total" in result


class TestToolEditFile:
    """Test tool_edit_file()."""

    @pytest.mark.asyncio
    async def test_successful_edit(self, tmp_path):
        from punty.telegram.tools import tool_edit_file

        test_file = tmp_path / "test.py"
        test_file.write_text("def hello():\n    return 'world'\n")

        result = await tool_edit_file(str(test_file), "world", "universe")
        assert "Replaced" in result
        assert "universe" in test_file.read_text()

    @pytest.mark.asyncio
    async def test_text_not_found(self, tmp_path):
        from punty.telegram.tools import tool_edit_file

        test_file = tmp_path / "test.py"
        test_file.write_text("hello world")

        result = await tool_edit_file(str(test_file), "nonexistent", "replacement")
        assert "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_file_not_found(self):
        from punty.telegram.tools import tool_edit_file

        result = await tool_edit_file("/tmp/nonexistent_xyz.py", "old", "new")
        assert "not found" in result.lower()


class TestToolQueryDb:
    """Test tool_query_db() write protection."""

    @pytest.mark.asyncio
    async def test_blocks_insert(self):
        from punty.telegram.tools import tool_query_db

        result = await tool_query_db("INSERT INTO meetings VALUES ('test')")
        assert "Only SELECT" in result

    @pytest.mark.asyncio
    async def test_blocks_delete(self):
        from punty.telegram.tools import tool_query_db

        result = await tool_query_db("DELETE FROM meetings")
        assert "Only SELECT" in result

    @pytest.mark.asyncio
    async def test_blocks_drop(self):
        from punty.telegram.tools import tool_query_db

        result = await tool_query_db("DROP TABLE meetings")
        assert "Only SELECT" in result

    @pytest.mark.asyncio
    async def test_blocks_update(self):
        from punty.telegram.tools import tool_query_db

        result = await tool_query_db("UPDATE meetings SET venue='test'")
        assert "Only SELECT" in result


# ── Agent Chat Tests ─────────────────────────────────────────────────────────


class TestAgentChat:
    """Test ClaudeAgent.chat() with mocked Anthropic API."""

    @pytest.mark.asyncio
    async def test_simple_text_response(self):
        from punty.telegram.agent import ClaudeAgent

        agent = ClaudeAgent(api_key="test-key")

        mock_text = MagicMock()
        mock_text.type = "text"
        mock_text.text = "Hello Rochey!"

        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [mock_text]

        agent.client.messages.create = AsyncMock(return_value=mock_response)
        result = await agent.chat(1, "hi")

        assert result == "Hello Rochey!"

    @pytest.mark.asyncio
    async def test_tool_use_loop(self):
        from punty.telegram.agent import ClaudeAgent

        agent = ClaudeAgent(api_key="test-key")

        # First response: tool use
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.id = "tool_1"
        tool_block.name = "bash"
        tool_block.input = {"command": "echo test"}

        first_response = MagicMock()
        first_response.stop_reason = "tool_use"
        first_response.content = [tool_block]

        # Second response: final text
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Done! Output was: test"

        second_response = MagicMock()
        second_response.stop_reason = "end_turn"
        second_response.content = [text_block]

        call_count = 0

        async def mock_create(**kwargs):
            nonlocal call_count
            call_count += 1
            return first_response if call_count == 1 else second_response

        agent.client.messages.create = mock_create

        async def mock_bash(cmd, timeout=30):
            return "test"

        with patch("punty.telegram.agent.tool_bash", mock_bash):
            result = await agent.chat(1, "run echo test")

        assert "Done!" in result
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_conversation_history_persists(self):
        from punty.telegram.agent import ClaudeAgent

        agent = ClaudeAgent(api_key="test-key")

        mock_text = MagicMock()
        mock_text.type = "text"
        mock_text.text = "Response"

        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [mock_text]

        agent.client.messages.create = AsyncMock(return_value=mock_response)

        await agent.chat(1, "first message")
        await agent.chat(1, "second message")

        # Should have 4 messages: user, assistant, user, assistant
        assert len(agent.conversations[1]) == 4
        assert agent.conversations[1][0]["role"] == "user"
        assert agent.conversations[1][0]["content"] == "first message"

    @pytest.mark.asyncio
    async def test_clear_history(self):
        from punty.telegram.agent import ClaudeAgent

        agent = ClaudeAgent(api_key="test-key")
        agent.conversations[1] = [{"role": "user", "content": "old message"}]

        agent.clear_history(1)
        assert 1 not in agent.conversations

    @pytest.mark.asyncio
    async def test_api_error_handled(self):
        from punty.telegram.agent import ClaudeAgent

        agent = ClaudeAgent(api_key="test-key")

        agent.client.messages.create = AsyncMock(
            side_effect=Exception("API rate limit exceeded")
        )
        result = await agent.chat(1, "test")

        assert "API error" in result

    @pytest.mark.asyncio
    async def test_max_turns_limit(self):
        from punty.telegram.agent import ClaudeAgent

        agent = ClaudeAgent(api_key="test-key")

        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.id = "tool_1"
        tool_block.name = "bash"
        tool_block.input = {"command": "echo loop"}

        mock_response = MagicMock()
        mock_response.stop_reason = "tool_use"
        mock_response.content = [tool_block]

        agent.client.messages.create = AsyncMock(return_value=mock_response)

        async def mock_bash(cmd, timeout=30):
            return "output"

        with patch("punty.telegram.agent.tool_bash", mock_bash):
            result = await agent.chat(1, "infinite loop")

        assert "max tool turns" in result.lower()
