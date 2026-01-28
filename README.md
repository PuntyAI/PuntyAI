# PuntyAI

AI-powered horse racing content generator - meet Punty, the cheeky Aussie tipster!

## Features

- **Web Dashboard** - Manage race meetings, content, and approvals
- **Multi-source Scraping** - Racing.com, TAB, Punters.com.au
- **AI Content Generation** - OpenAI-powered with Punty's personality
- **Context Versioning** - Detects when speed maps change tips
- **Platform Formatting** - WhatsApp and Twitter/X ready
- **Tiered Approval** - Auto-send routine, manual review for big tips

## Quick Start

```bash
# Install
pip install -e .

# Configure
cp .env.example .env
# Edit .env with your OPENAI_API_KEY

# Run
uvicorn punty.main:app --reload
```

Open http://localhost:8000

## Gamble Responsibly

1800 858 858
