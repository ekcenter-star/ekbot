# Phirunn Bot

EK clinic Telegram assistant with two features:

1. **Document clean-up** — tag the bot on a photo of a consult letter or
   invoice and get back a color-enhanced, deskewed, "scanned" version of
   the *same* letter, ready to attach in Cliniko. Nothing is retyped or
   regenerated — only the image is cleaned up (deskew, crop, white-balance,
   contrast, sharpen). Stamps and signatures stay in color.
2. **`/how` greeting** — sends a random pre-recorded Khmer greeting mp3.
   No TTS API at runtime; you generate the mp3s once (e.g. with Kiri TTS)
   and drop them into `assets/greetings/`.

The bot only acts when explicitly **mentioned/tagged** — it ignores plain
photos so it never touches patient photos posted in the group.

## How to trigger it

- **Clean up a letter**: send the photo with caption `@Phirunn_Bot`, OR
  send the photo first, then reply to it with `@Phirunn_Bot`.
- **Greeting**: send `/how@Phirunn_Bot` (or just `/how` in a DM).

## 1. Create the bot in Telegram

1. Message [@BotFather](https://t.me/BotFather) → `/newbot` → follow prompts.
2. Save the token it gives you — that's `BOT_TOKEN`.
3. Set the bot's username to `Phirunn_Bot` (or update `BOT_USERNAME` below
   to match whatever you actually pick).
4. **Important — group visibility**: message BotFather → `/mybots` → select
   your bot → **Bot Settings** → **Group Privacy** → **Turn off**.
   This lets the bot reliably see captions/replies that mention it in the
   group (otherwise Telegram may withhold some of those messages from it).
5. Add the bot to the EK Telegram group as a normal member.

## 2. Local setup (optional, for testing before deploying)

```bash
cd phirunn-bot
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in BOT_TOKEN
python app/main.py
```

## 3. Generate greeting mp3s

Use Kiri TTS (https://www.kiritts.com) to generate one or more Khmer
greeting clips (e.g. "សួស្តីកញ្ញា..."), export as mp3, and drop them into:

```
assets/greetings/
```

Add as many as you like — the bot randomly picks one each time so repeat
uses don't feel robotic. No code changes needed.

## 4. Deploy to Railway

1. Push this folder to a new GitHub repo.
2. In Railway: **New Project → Deploy from GitHub repo** → pick the repo.
3. Railway will detect the `Procfile` and run `python app/main.py` as a
   worker (no public web port needed — it's a polling bot, not a server).
4. In Railway → **Variables**, add:
   - `BOT_TOKEN` = the token from BotFather
   - `BOT_USERNAME` = `Phirunn_Bot`
5. Deploy. Check the logs for `Phirunn Bot starting...`.

## Tuning the clean-up

All the image-processing logic lives in `app/scanner.py`:

- `_find_document_contour` — detects the paper's 4 edges to deskew/crop.
  If it can't find a confident rectangle (e.g. edges are cut off in the
  photo), it skips cropping entirely rather than risk cutting off content.
- `_white_balance` / `_boost_contrast` / `_sharpen` — the actual "make it
  look printed" steps. These are deliberately mild — if letters print with
  too much/little contrast for your taste, tweak `clipLimit` in
  `_boost_contrast` or the sharpen weights in `_sharpen`.

## Notes / next steps

- Currently sent back as a Telegram **photo** (compressed by Telegram, but
  fine for screen viewing/attaching to Cliniko). If you ever need
  full-resolution, uncompressed output, we can switch to sending as a
  **document** instead — trivial change in `main.py`.
- This is a fresh Railway service, separate from ChatBot Azasshinz, so it
  won't affect the existing content-checking bot.
