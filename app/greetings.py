"""
greetings.py
Sends a pre-recorded Khmer greeting mp3 when someone triggers /How.

No TTS API call at runtime — mp3 files are generated once (e.g. via Kiri TTS)
and dropped into assets/greetings/. Drop in as many as you like; the bot
picks one at random each time so it doesn't feel too robotic on repeat.

To add/change greetings: just add .mp3 files to assets/greetings/ and
redeploy. No code changes needed.
"""

import random
from pathlib import Path

GREETINGS_DIR = Path(__file__).resolve().parent.parent / "assets" / "greetings"


def pick_greeting() -> Path | None:
    files = sorted(GREETINGS_DIR.glob("*.mp3"))
    if not files:
        return None
    return random.choice(files)
