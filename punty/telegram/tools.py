"""Server management tools for the Claude agent via Telegram."""

import asyncio
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

MAX_OUTPUT_LENGTH = 10_000
BASH_TIMEOUT = 30
MAX_BASH_TIMEOUT = 120
PROJECT_ROOT = "/opt/puntyai"
DB_PATH = "/opt/puntyai/data/punty.db"


def _resolve_path(path: str, must_be_within_root: bool = True) -> str:
    """Resolve a path, treating relative paths as relative to PROJECT_ROOT.

    Args:
        path: File path (relative or absolute).
        must_be_within_root: If True, reject paths that resolve outside PROJECT_ROOT.

    Raises:
        ValueError: If the resolved path escapes PROJECT_ROOT.
    """
    p = Path(path)
    if not p.is_absolute():
        p = Path(PROJECT_ROOT) / p
    resolved = p.resolve()
    if must_be_within_root and not str(resolved).startswith(str(Path(PROJECT_ROOT).resolve())):
        raise ValueError(f"Path escapes project root: {path}")
    return str(resolved)


def _truncate(text: str, limit: int = MAX_OUTPUT_LENGTH) -> str:
    """Truncate text with a notice if it exceeds the limit."""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n\n... (truncated, {len(text)} chars total)"


async def tool_bash(command: str, timeout: int = BASH_TIMEOUT) -> str:
    """Run a shell command and return stdout+stderr."""
    try:
        timeout = min(max(timeout, 1), MAX_BASH_TIMEOUT)

        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=PROJECT_ROOT,
        )

        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return f"Command timed out after {timeout}s"

        output = ""
        if stdout:
            output += stdout.decode("utf-8", errors="replace")
        if stderr:
            if output:
                output += "\n--- stderr ---\n"
            output += stderr.decode("utf-8", errors="replace")

        if not output.strip():
            output = f"(no output, exit code {proc.returncode})"
        elif proc.returncode != 0:
            output += f"\n(exit code {proc.returncode})"

        return _truncate(output)
    except Exception as e:
        return f"Error running command: {e}"


async def tool_read_file(path: str, max_lines: int = 500) -> str:
    """Read a file's contents."""
    try:
        resolved = _resolve_path(path)
    except ValueError as e:
        return str(e)
    try:
        p = Path(resolved)

        if not p.exists():
            return f"File not found: {resolved}"
        if not p.is_file():
            return f"Not a file: {resolved}"

        text = p.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines(keepends=True)

        if len(lines) > max_lines:
            result = "".join(lines[:max_lines])
            result += f"\n... ({len(lines)} lines total, showing first {max_lines})"
            return _truncate(result)

        return _truncate(text)
    except Exception as e:
        return f"Error reading file: {e}"


async def tool_write_file(path: str, content: str) -> str:
    """Write content to a file."""
    try:
        resolved = _resolve_path(path)
    except ValueError as e:
        return str(e)
    try:
        p = Path(resolved)

        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")

        return f"Written {len(content)} chars to {resolved}"
    except Exception as e:
        return f"Error writing file: {e}"


async def tool_edit_file(path: str, old_text: str, new_text: str) -> str:
    """Replace first occurrence of old_text with new_text in a file."""
    try:
        resolved = _resolve_path(path)
    except ValueError as e:
        return str(e)
    try:
        p = Path(resolved)

        if not p.exists():
            return f"File not found: {resolved}"

        content = p.read_text(encoding="utf-8")

        if old_text not in content:
            return f"Text not found in {resolved}. No changes made."

        count = content.count(old_text)
        new_content = content.replace(old_text, new_text, 1)
        p.write_text(new_content, encoding="utf-8")

        if count > 1:
            return f"Replaced 1 of {count} occurrences in {resolved}"
        return f"Replaced text in {resolved}"
    except Exception as e:
        return f"Error editing file: {e}"


async def tool_list_files(path: str = PROJECT_ROOT, pattern: str = "*") -> str:
    """List files in a directory with optional glob pattern."""
    try:
        resolved = _resolve_path(path)
    except ValueError as e:
        return str(e)
    try:
        p = Path(resolved)

        if not p.exists():
            return f"Directory not found: {resolved}"
        if not p.is_dir():
            return f"Not a directory: {resolved}"

        entries = sorted(p.glob(pattern))
        if not entries:
            return f"No matches for '{pattern}' in {resolved}"

        lines = []
        for entry in entries[:200]:
            suffix = "/" if entry.is_dir() else ""
            lines.append(f"{entry.name}{suffix}")

        result = "\n".join(lines)
        if len(entries) > 200:
            result += f"\n... ({len(entries)} entries total, showing first 200)"

        return result
    except Exception as e:
        return f"Error listing files: {e}"


async def tool_update_track_condition(venue: str, date: str, condition: str) -> str:
    """Update track condition for a meeting and all its races."""
    try:
        import aiosqlite

        # Validate condition format (e.g. "Good 4", "Soft 5", "Heavy 8")
        valid_labels = ["firm", "good", "soft", "heavy", "synthetic"]
        cond_lower = condition.lower().strip()
        if not any(cond_lower.startswith(label) for label in valid_labels):
            return f"Invalid condition '{condition}'. Must start with Firm/Good/Soft/Heavy/Synthetic."

        async with aiosqlite.connect(DB_PATH) as db:
            # Find the meeting
            cursor = await db.execute(
                "SELECT id, venue, track_condition FROM meetings WHERE LOWER(venue) = LOWER(?) AND date = ?",
                (venue.strip(), date.strip()),
            )
            row = await cursor.fetchone()
            if not row:
                # Try partial match
                cursor = await db.execute(
                    "SELECT id, venue, track_condition FROM meetings WHERE LOWER(venue) LIKE LOWER(?) AND date = ?",
                    (f"%{venue.strip()}%", date.strip()),
                )
                row = await cursor.fetchone()

            if not row:
                return f"No meeting found for {venue} on {date}"

            meeting_id, actual_venue, old_condition = row

            # Update meeting and lock it from auto-updates
            await db.execute(
                "UPDATE meetings SET track_condition = ?, track_condition_locked = 1 WHERE id = ?",
                (condition.strip(), meeting_id),
            )

            # Update all races
            cursor = await db.execute(
                "UPDATE races SET track_condition = ? WHERE meeting_id = ?",
                (condition.strip(), meeting_id),
            )
            race_count = cursor.rowcount

            await db.commit()

            return (
                f"Updated {actual_venue} ({date}):\n"
                f"  Meeting: {old_condition} → {condition}\n"
                f"  Races updated: {race_count}"
            )
    except Exception as e:
        return f"Error updating track condition: {e}"


async def tool_query_db(sql: str) -> str:
    """Run a read-only SQL query against the PuntyAI database."""
    try:
        # Block write operations
        sql_upper = sql.strip().upper()
        if not sql_upper.startswith("SELECT") and not sql_upper.startswith("WITH"):
            return "Only SELECT queries are allowed. Use bash for write operations."

        blocked = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "ATTACH"]
        for keyword in blocked:
            if keyword in sql_upper:
                return f"Blocked: {keyword} statements not allowed via query_db."

        import aiosqlite
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(sql)
            rows = await cursor.fetchmany(100)

            if not rows:
                return "(no results)"

            # Format as table
            columns = [desc[0] for desc in cursor.description]
            lines = [" | ".join(columns)]
            lines.append("-" * len(lines[0]))

            for row in rows:
                values = [str(v) if v is not None else "NULL" for v in row]
                lines.append(" | ".join(values))

            result = "\n".join(lines)

            total = len(rows)
            if total >= 100:
                result += "\n... (limited to 100 rows)"

            return _truncate(result)
    except Exception as e:
        return f"SQL error: {e}"
