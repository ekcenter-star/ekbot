"""
EK Bot — EK clinic Telegram assistant.

Two features:
  1. Tag the bot on a photo of a consult letter / invoice -> get back a
     cleaned-up, color-enhanced, "scanned" version of the SAME letter
     (no content is retyped or regenerated — only the image is cleaned up).
  2. Mention the bot with /how -> get a random pre-recorded Khmer greeting mp3.

Trigger rule (important): the bot only acts when explicitly @mentioned.
It ignores photos posted without a mention, since the group also has patient
photos that should never be auto-processed.

Name-change-safe: the bot reads its own @username live from the Telegram API
(context.bot.username) so renaming in BotFather takes effect on next restart
with zero config changes.
"""

import logging
import os

from dotenv import load_dotenv
from telegram import Message, MessageEntity, Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from scanner import clean_document
from greetings import pick_greeting

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("ek-bot")

BOT_TOKEN = os.environ["BOT_TOKEN"]


def _bot_is_mentioned(message: Message, bot_username: str) -> bool:
    """
    Returns True only if the bot is explicitly @mentioned in this message
    (text or caption). Uses Telegram message entities for an exact handle
    match — no substring false-positives, and no hardcoded username.

    bot_username comes from context.bot.username which the Telegram API
    fills automatically, so renaming the bot in BotFather takes effect
    on the next restart with zero config changes.
    """
    handle = f"@{bot_username}".lower()

    # Entities in plain text messages
    for mention in message.parse_entities(types=[MessageEntity.MENTION]).values():
        if mention.lower() == handle:
            return True

    # Entities in photo / video captions
    for mention in message.parse_caption_entities(types=[MessageEntity.MENTION]).values():
        if mention.lower() == handle:
            return True

    return False


async def handle_how(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/how (or /how@BotUsername in groups) -> send a random Khmer greeting mp3."""
    greeting = pick_greeting()
    if greeting is None:
        await update.message.reply_text(
            "No greeting files found yet — drop some .mp3 files into assets/greetings/."
        )
        return

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.UPLOAD_VOICE)
    with open(greeting, "rb") as f:
        await update.message.reply_voice(voice=f)


async def handle_photo_mention(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handles two cases:
      A) Photo posted WITH a caption that @mentions the bot
      B) A text message that @mentions the bot is a reply to a photo message

    Uses entity-based detection via context.bot.username (live from Telegram
    API) so this works correctly regardless of the bot's registered @username.
    """
    message = update.message
    if message is None:
        return

    # Read the bot's current username live from the Telegram API.
    # This auto-updates if the bot is renamed in BotFather (after restart).
    bot_username: str = context.bot.username  # type: ignore[assignment]

    photo_message = None

    if message.photo and _bot_is_mentioned(message, bot_username):
        # Case A: photo with a caption that tags the bot
        photo_message = message
    elif (
        message.reply_to_message
        and message.reply_to_message.photo
        and _bot_is_mentioned(message, bot_username)
    ):
        # Case B: text reply to a photo that tags the bot
        photo_message = message.reply_to_message

    if photo_message is None:
        return  # not a relevant trigger, ignore silently

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.UPLOAD_PHOTO)

    try:
        tg_photo = photo_message.photo[-1]  # highest resolution
        file = await context.bot.get_file(tg_photo.file_id)
        photo_bytes = bytes(await file.download_as_bytearray())

        cleaned_bytes = clean_document(photo_bytes)

        await message.reply_photo(
            photo=cleaned_bytes,
            caption="✅ Cleaned up and ready to attach in Cliniko.",
        )
    except Exception:
        logger.exception("Failed to process document photo")
        await message.reply_text(
            "⚠️ Sorry, I couldn't clean that image up. Try a clearer / better-lit photo of the letter."
        )


def main() -> None:
    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("how", handle_how))
    app.add_handler(
        MessageHandler(
            (filters.PHOTO | (filters.TEXT & filters.REPLY)) & ~filters.COMMAND,
            handle_photo_mention,
        )
    )

    logger.info("EK Bot starting — username will be fetched live from Telegram API.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
