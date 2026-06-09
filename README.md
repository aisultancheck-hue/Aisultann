# Telegram Bot

A Telegram AI bot with Groq, PDF/RAG, optional Supabase vector storage, web search, and multi-step answering.

## Run

Open PowerShell in this folder:

```powershell
cd C:\Users\Acer\Documents\Codex\2026-06-08\8647636709-aagalpw1j0yzclp3xwbdlohccylrjff5kce\outputs\telegram_bot
```

If normal Python is installed:

```powershell
pip install -r requirements.txt
python bot.py
```

On this Codex machine you can also use the bundled Python:

```powershell
& 'C:\Users\Acer\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' bot.py
```

Then open your bot in Telegram and send `/start`.

## Token

The token is stored in the local `.env` file:

```env
TELEGRAM_BOT_TOKEN=put_your_bot_token_here
```

Do not publish `.env`.

## AI / RAG / Web

- `GROQ_API_KEY` enables AI answers through Groq.
- Send a PDF file to the bot to index it for RAG.
- Use `/rag your question` to ask over indexed PDFs.
- Use `/web your query` to search the web.
- Add `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` to store vectors in Supabase.
- Run `supabase_schema.sql` in Supabase SQL Editor before enabling Supabase vector search.

## Commands

- `/start` - greeting
- `/help` - help
- `/echo text` - repeat text

For a normal text message, the bot repeats the received text.
