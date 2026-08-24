"""
gemini_ocr.py
Uses Google Gemini Vision to extract text from a cleaned document image.
Returns a formatted string with all text found in the document.
"""

import os
import logging

import google.generativeai as genai

logger = logging.getLogger("ek-bot")

_MODEL_NAME = "gemini-1.5-pro"  # Pro tier — best for handwritten medical documents

PROMPT = (
    "You are reading a medical consult letter or invoice from EK Aesthetic Center clinic. "
    "Extract ALL text you can see in this document — patient name, date, treatments, prices, "
    "notes, signatures, everything. "
    "Format clearly with line breaks. Keep all numbers, dates, and medical terms exactly as written. "
    "If handwriting is unclear, write your best guess followed by (?) "
    "Reply in the same language as the document."
)


def extract_text(image_bytes: bytes) -> str | None:
    """
    Send image_bytes to Gemini and return the extracted text.
    Returns None if GEMINI_API_KEY is not set or on API error.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY not set — skipping OCR")
        return None

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(_MODEL_NAME)

        image_part = {"mime_type": "image/jpeg", "data": image_bytes}
        response = model.generate_content([PROMPT, image_part])

        text = response.text.strip()
        logger.info("Gemini OCR returned %d characters", len(text))
        return text

    except Exception:
        logger.exception("Gemini OCR failed")
        return None
