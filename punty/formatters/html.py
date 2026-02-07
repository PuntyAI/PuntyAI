"""HTML formatter for public tips pages."""

import re
from typing import Optional

# Rotating greetings (replacing "Rightio Cunts")
GREETINGS = [
    "Rightio Legends",
    "Rightio Degenerates",
    "Rightio You Beautiful Bastards",
    "Rightio Sickos",
    "Rightio Loose Units",
    "Rightio Dropkicks",
    "Rightio Ratbags",
    "Rightio Galah Gang",
    "Rightio Drongos",
    "Rightio You Feral Lot",
    "Rightio Ticket Munchers",
    "Rightio Cooked Units",
    "Rightio Form Freaks",
    "Rightio Chaos Merchants",
    "Rightio Punty People",
    "Rightio You Grubby Lot",
    "Rightio Muppets",
    "Rightio Absolute Units",
    "Rightio Filthy Animals",
    "Rightio You Sick Puppies",
]

# Rotating section names (replacing "CUNT FACTORY")
SECTION_NAMES = [
    "THE SICKO SANCTUARY",
    "THE CHAOS KITCHEN",
    "THE DEGEN DEN",
    "THE LOOSE UNIT LOUNGE",
    "THE RATBAG REPORT",
    "THE COOKED CORNER",
    "PUNTY'S PULPIT",
    "THE FERAL FACTORY",
    "THE DROPKICK DISPATCH",
    "THE GALAH GAZETTE",
    "THE GRUB HUB",
    "THE SICKO SYNDICATE",
    "THE MUPPET MANIFESTO",
    "THE FILTH FILTER",
    "THE BASTARD BUREAU",
    "THE UNHINGED OFFICE",
    "PUNTY'S PARTING SHOT",
    "THE DRONGO DISPATCH",
    "THE TICKET TEMPLE",
    "THE ABSOLUTE UNIT ARCHIVES",
]


def get_greeting(seed: int = 0) -> str:
    """Get a greeting based on seed (e.g., meeting date hash)."""
    return GREETINGS[seed % len(GREETINGS)]


def get_section_name(seed: int = 0) -> str:
    """Get a section name based on seed."""
    return SECTION_NAMES[seed % len(SECTION_NAMES)]


def replace_banned_words(content: str, seed: int = 0) -> str:
    """Replace banned words with rotating alternatives."""
    greeting = get_greeting(seed)
    section_name = get_section_name(seed)

    # Replace greeting variations
    content = re.sub(r"Rightio\s+[Cc]unts?\s*[-—]?", f"{greeting} —", content)

    # Replace section name variations
    content = re.sub(
        r"\*?FINAL\s+WORD\s+FROM\s+THE\s+CUNT\s+FACTORY\*?",
        f"*FINAL WORD FROM {section_name}*",
        content,
        flags=re.IGNORECASE
    )
    content = re.sub(
        r"\*?THE\s+CUNT\s+FACTORY\*?",
        f"*{section_name}*",
        content,
        flags=re.IGNORECASE
    )

    # Replace character name
    content = re.sub(
        r"Punty\s+the\s+[Cc]unty",
        "Punty the Loose Unit",
        content
    )

    # Replace any standalone usage (case insensitive, but not in URLs or technical text)
    # Only replace when it appears as a noun/term of address
    content = re.sub(r"\bcunts?\b", "legends", content, flags=re.IGNORECASE)

    return content


def format_html(raw_content: str, content_type: str = "early_mail", seed: int = 0) -> str:
    """Format raw content as HTML for public display.

    Args:
        raw_content: The raw markdown-style content
        content_type: Type of content (early_mail, meeting_wrapup)
        seed: Seed for rotating word replacements

    Returns:
        HTML-formatted content
    """
    content = raw_content

    # Replace banned words first
    content = replace_banned_words(content, seed)

    # Remove the title line if it starts with *PUNTY EARLY MAIL*
    content = re.sub(r'^\*PUNTY EARLY MAIL[^*]*\*\s*\n*', '', content, flags=re.IGNORECASE)

    # Convert headers (### *TEXT* or ### TEXT)
    def convert_header(m):
        level = len(m.group(1))
        text = m.group(2).strip().strip('*')
        tag = f"h{min(level + 1, 4)}"  # h2, h3, h4
        return f'<{tag} class="tips-heading">{text}</{tag}>'

    content = re.sub(r'^(#{1,3})\s+(.+)$', convert_header, content, flags=re.MULTILINE)

    # Convert *bold* to <strong>
    content = re.sub(r'\*([^*\n]+)\*', r'<strong>\1</strong>', content)

    # Convert _italic_ to <em>
    content = re.sub(r'_([^_\n]+)_', r'<em>\1</em>', content)

    # Convert --- or === horizontal rules
    content = re.sub(r'^[-=]{3,}$', '<hr class="tips-divider">', content, flags=re.MULTILINE)

    # Convert bullet points
    def convert_bullets(m):
        items = m.group(0).strip().split('\n')
        list_items = ''.join(f'<li>{item.lstrip("- •").strip()}</li>' for item in items if item.strip())
        return f'<ul class="tips-list">{list_items}</ul>'

    content = re.sub(r'(?:^[-•]\s+.+$\n?)+', convert_bullets, content, flags=re.MULTILINE)

    # Convert numbered lists
    def convert_numbered(m):
        items = m.group(0).strip().split('\n')
        list_items = ''.join(f'<li>{re.sub(r"^[0-9]+[.)]\s*", "", item).strip()}</li>' for item in items if item.strip())
        return f'<ol class="tips-list">{list_items}</ol>'

    content = re.sub(r'(?:^[0-9]+[.)]\s+.+$\n?)+', convert_numbered, content, flags=re.MULTILINE)

    # Convert paragraphs (double newlines)
    paragraphs = re.split(r'\n\n+', content)
    formatted_paragraphs = []

    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        # Skip if already has block-level HTML
        if p.startswith('<h') or p.startswith('<ul') or p.startswith('<ol') or p.startswith('<hr'):
            formatted_paragraphs.append(p)
        else:
            # Convert single newlines to <br> within paragraph
            p = p.replace('\n', '<br>')
            formatted_paragraphs.append(f'<p>{p}</p>')

    content = '\n'.join(formatted_paragraphs)

    return content


class HTMLFormatter:
    """Format content for HTML display on public tips pages."""

    @classmethod
    def format(cls, raw_content: str, content_type: str = "early_mail", seed: int = 0) -> str:
        """Format raw content as HTML."""
        return format_html(raw_content, content_type, seed)
