"""
greetings.py
Sends pre-recorded Khmer mp3s for two situations:
  1. /how command          → random greeting from assets/greetings/
  2. Letter not in frame   → random feedback from assets/feedback/

To add/change files: just drop .mp3 files into the relevant folder and
redeploy. No code changes needed.
"""

import random
from pathlib import Path

GREETINGS_DIR = Path(__file__).resolve().parent.parent / "assets" / "greetings"
FEEDBACK_DIR  = Path(__file__).resolve().parent.parent / "assets" / "feedback"


def pick_greeting() -> Path | None:
    files = sorted(GREETINGS_DIR.glob("*.mp3"))
    if not files:
        return None
    return random.choice(files)


def pick_feedback() -> Path | None:
    """Pick a random feedback mp3 to play when the letter is out of frame."""
    files = sorted(FEEDBACK_DIR.glob("*.mp3"))
    if not files:
        return None
    return random.choice(files)
