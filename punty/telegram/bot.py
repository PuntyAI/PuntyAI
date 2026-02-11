"""Telegram bot for PuntyAI server management."""

import asyncio
import logging
from typing import Optional

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from punty.telegram.agent import ClaudeAgent

logger = logging.getLogger(__name__)

TELEGRAM_MAX_LENGTH = 4096


class TelegramBot:
    """Telegram bot that lets the owner manage PuntyAI via Claude.

    Runs as a background asyncio task alongside ResultsMonitor.
    Only the configured owner Telegram user ID is authorized.
    """

    def __init__(self, app):
        self.fastapi_app = app
        self.application: Optional[Application] = None
        self.agent: Optional[ClaudeAgent] = None
        self._running = False
        self._owner_id: Optional[int] = None
        self._chat_locks: dict[int, asyncio.Lock] = {}

    async def start(self) -> bool:
        """Initialize and start the Telegram bot.

        Returns False if not configured (missing keys).
        """
        from punty.models.database import async_session
        from punty.models.settings import get_api_key

        try:
            async with async_session() as db:
                bot_token = await get_api_key(db, "telegram_bot_token")
                owner_id_str = await get_api_key(db, "telegram_owner_id")
                anthropic_key = await get_api_key(db, "anthropic_api_key")

            if not bot_token or not owner_id_str or not anthropic_key:
                missing = []
                if not bot_token:
                    missing.append("telegram_bot_token")
                if not owner_id_str:
                    missing.append("telegram_owner_id")
                if not anthropic_key:
                    missing.append("anthropic_api_key")
                logger.info(f"Telegram bot not configured (missing: {', '.join(missing)})")
                return False

            try:
                self._owner_id = int(owner_id_str)
            except ValueError:
                logger.error(f"Invalid telegram_owner_id: {owner_id_str}")
                return False

            # Build the telegram application
            self.application = ApplicationBuilder().token(bot_token).build()

            # Add handlers
            self.application.add_handler(CommandHandler("start", self._handle_start))
            self.application.add_handler(CommandHandler("clear", self._handle_clear))
            self.application.add_handler(
                MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message)
            )

            # Create the Claude agent
            self.agent = ClaudeAgent(api_key=anthropic_key)

            # Start the bot with manual lifecycle (not run_polling which blocks)
            await self.application.initialize()
            await self.application.start()
            await self.application.updater.start_polling(drop_pending_updates=True)

            self._running = True
            logger.info(f"Telegram bot started (owner: {self._owner_id})")
            return True

        except Exception as e:
            logger.error(f"Failed to start Telegram bot: {e}")
            return False

    async def stop(self):
        """Stop the Telegram bot gracefully."""
        if not self._running:
            return

        self._running = False

        try:
            if self.application and self.application.updater:
                await self.application.updater.stop()
            if self.application:
                await self.application.stop()
                await self.application.shutdown()
            if self.agent:
                await self.agent.close()
            logger.info("Telegram bot stopped")
        except Exception as e:
            logger.error(f"Error stopping Telegram bot: {e}")

    def is_running(self) -> bool:
        return self._running

    def _check_authorized(self, update: Update) -> bool:
        """Check if the message is from the authorized owner."""
        if not update.effective_user:
            return False
        return update.effective_user.id == self._owner_id

    async def _handle_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        if not self._check_authorized(update):
            await update.message.reply_text("Unauthorized.")
            return

        await update.message.reply_text(
            "G'day Rochey! PuntyAI server bot ready.\n\n"
            "Ask me anything about the server — check logs, fix bugs, query the DB, "
            "restart services.\n\n"
            "/clear — reset conversation"
        )

    async def _handle_clear(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /clear command to reset conversation."""
        if not self._check_authorized(update):
            await update.message.reply_text("Unauthorized.")
            return

        chat_id = update.effective_chat.id
        if self.agent:
            self.agent.clear_history(chat_id)
        await update.message.reply_text("Conversation cleared.")

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming text messages."""
        if not self._check_authorized(update):
            await update.message.reply_text("Unauthorized.")
            return

        chat_id = update.effective_chat.id
        user_text = update.message.text

        if not user_text or not user_text.strip():
            return

        # Get or create lock for this chat
        if chat_id not in self._chat_locks:
            self._chat_locks[chat_id] = asyncio.Lock()

        lock = self._chat_locks[chat_id]
        if lock.locked():
            await update.message.reply_text("Still working on your last request...")
            return

        async with lock:
            # Send typing indicator
            await context.bot.send_chat_action(chat_id=chat_id, action="typing")

            try:
                response = await self.agent.chat(chat_id, user_text)
            except Exception as e:
                logger.error(f"Agent error: {e}")
                response = f"Error: {e}"

            # Send response in chunks
            chunks = self._chunk_message(response)
            for chunk in chunks:
                try:
                    await update.message.reply_text(
                        chunk, parse_mode=ParseMode.MARKDOWN
                    )
                except Exception:
                    # Markdown parse failed — send as plain text
                    try:
                        await update.message.reply_text(chunk)
                    except Exception as e:
                        logger.error(f"Failed to send message: {e}")

    @staticmethod
    def _chunk_message(text: str, max_length: int = TELEGRAM_MAX_LENGTH) -> list[str]:
        """Split a message into chunks that fit Telegram's limit."""
        if not text:
            return ["(empty response)"]

        if len(text) <= max_length:
            return [text]

        chunks = []
        remaining = text

        while remaining:
            if len(remaining) <= max_length:
                chunks.append(remaining)
                break

            # Try to split on double newline (paragraph break)
            split_at = remaining.rfind("\n\n", 0, max_length)
            if split_at > max_length // 2:
                chunks.append(remaining[:split_at])
                remaining = remaining[split_at + 2:]
                continue

            # Try to split on single newline
            split_at = remaining.rfind("\n", 0, max_length)
            if split_at > max_length // 2:
                chunks.append(remaining[:split_at])
                remaining = remaining[split_at + 1:]
                continue

            # Hard split at max_length
            chunks.append(remaining[:max_length])
            remaining = remaining[max_length:]

        return chunks
