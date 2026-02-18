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

    # Pre-process: convert bold/italic heading patterns to markdown # headers
    # before the generic bold/italic conversion runs.

    # Remove RACE-BY-RACE divider (redundant with accordion structure on website)
    content = re.sub(
        r'^(?:#{1,3}\s+(?:\d+\)\s*)?)?\*{0,2}RACE[-\s]BY[-\s]RACE\*{0,2}\s*$',
        '', content, flags=re.MULTILINE | re.IGNORECASE,
    )

    # *Race 1 – The Big Title* → ## Race 1 – The Big Title (becomes <h3>)
    content = re.sub(
        r'^\*+(Race\s+\d+\s*[-–—].+?)\*+\s*$',
        r'## \1', content, flags=re.MULTILINE,
    )

    # Sequence headings: EARLY QUADDIE (...), MAIN QUADDIE (...), BIG 6 (...)
    content = re.sub(
        r'^((?:EARLY\s+|MAIN\s+)?QUADDIE|BIG\s*6)\s*(\(.*?\))?\s*$',
        lambda m: f'## {m.group(1)}{" " + m.group(2) if m.group(2) else ""}',
        content, flags=re.MULTILINE,
    )

    # Trailing section headings: NUGGETS FROM THE TRACK, FINAL WORD FROM ...
    content = re.sub(
        r'^\*{1,2}(NUGGETS\s+FROM\s+THE\s+TRACK)\*{1,2}\s*$',
        r'## \1', content, flags=re.MULTILINE | re.IGNORECASE,
    )
    content = re.sub(
        r'^\*{1,2}(FINAL\s+WORD\s+FROM\s+.+?)\*{1,2}\s*$',
        r'## \1', content, flags=re.MULTILINE | re.IGNORECASE,
    )

    # Remove FIND OUT MORE section (shown on other delivery platforms, not website)
    content = re.sub(
        r'^(?:#{1,3}\s+(?:\d+\)\s*)?)?\*{1,2}FIND\s+OUT\s+MORE\*{1,2}[^\n]*\n(?:[^\n]+\n)*',
        '', content, flags=re.MULTILINE | re.IGNORECASE,
    )

    # Style exotic section header as inline heading (no trailing blank line gap)
    content = re.sub(
        r'^\*{1,2}(Degenerate\s+Exotic\s+(?:of\s+the\s+Race)?)\*{1,2}\s*\n?',
        r'<div class="exotic-title">\1</div>',
        content, flags=re.MULTILINE | re.IGNORECASE,
    )

    # Convert known intro section headings (AI may use ** instead of ###)
    content = re.sub(
        r'^\*{1,2}(MEET(?:ING)?\s+SNAPSHOT)\*{1,2}\s*$',
        r'## \1', content, flags=re.MULTILINE | re.IGNORECASE,
    )
    content = re.sub(
        r'^\*{1,2}((?:PUNTY.S\s+)?BIG\s*(?:3|THREE)[^*\n]*)\*{1,2}\s*$',
        r'## \1', content, flags=re.MULTILINE | re.IGNORECASE,
    )

    # Convert standalone bold lines to section headings (wrap-ups, reviews)
    # Matches lines that are entirely *Bold Text* with 2+ words starting uppercase
    # Excludes Jockeys/Stables lines — those stay as inline bold labels
    content = re.sub(
        r'^\*{1,2}(?!Jockeys\s+to\s+follow|Stables\s+to\s+respect)([A-Z][^*\n]*\s[^*\n]{2,})\*{1,2}\s*$',
        r'## \1', content, flags=re.MULTILINE | re.IGNORECASE,
    )

    # Convert headers (### *TEXT* or ### 1) TEXT)
    def convert_header(m):
        level = len(m.group(1))
        text = m.group(2).strip().strip('*')
        # Strip leading section numbers like "1) ", "2) "
        text = re.sub(r'^\d+\)\s*', '', text).strip().strip('*')
        tag = f"h{min(level + 1, 4)}"  # h2, h3, h4
        return f'<{tag} class="tips-heading">{text}</{tag}>'

    # Remove literal "### 1) HEADER" lines (AI template artifact)
    content = re.sub(r'^#{1,3}\s+\d+\)\s*HEADER\s*$', '', content, flags=re.MULTILINE)

    # Convert headers and ensure they're in their own paragraph block
    def convert_header_with_breaks(m):
        result = convert_header(m)
        return f'\n\n{result}\n\n'

    content = re.sub(r'^(#{1,3})\s+(.+)$', convert_header_with_breaks, content, flags=re.MULTILINE)

    # Convert markdown links [text](url) to <a> tags
    content = re.sub(
        r'\[([^\]]+)\]\(([^)]+)\)',
        r'<a href="\2" target="_blank" class="tips-link">\1</a>',
        content
    )

    # Convert **bold** to <strong> (handle double asterisks first)
    content = re.sub(r'\*\*([^*\n]+)\*\*', r'<strong>\1</strong>', content)

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
        strip_num = re.compile(r"^[0-9]+[.)]\s*")
        list_items = ''.join(f'<li>{strip_num.sub("", item).strip()}</li>' for item in items if item.strip())
        return f'<ol class="tips-list">{list_items}</ol>'

    content = re.sub(r'(?:^[0-9]+[.)]\s+.+$\n?)+', convert_numbered, content, flags=re.MULTILINE)

    # Style selection metadata lines (before paragraph wrapping)
    content = re.sub(
        r'^\s*Bet:\s*(.+)$',
        r'<span class="sel-bet"><span class="sel-label">Bet</span> \1</span>',
        content, flags=re.MULTILINE
    )
    content = re.sub(
        r'^\s*Win\s*%:\s*(\d+(?:\.\d+)?%)',
        r'<span class="sel-winpct"><span class="sel-label">Win%</span> \1</span>',
        content, flags=re.MULTILINE
    )
    # Strip Confidence lines (replaced by Probability)
    content = re.sub(
        r'^\s*Confidence:\s*(high|med|medium|low)\s*$',
        '', content, flags=re.MULTILINE | re.IGNORECASE
    )
    content = re.sub(
        r'^\s*Why:\s*(.+)$',
        r'<span class="sel-why"><span class="sel-label">Why</span> \1</span>',
        content, flags=re.MULTILINE
    )
    content = re.sub(
        r'^\s*Est\.\s*return:\s*(.+)$',
        r'<span class="sel-bet"><span class="sel-label">Return</span> \1</span>',
        content, flags=re.MULTILINE | re.IGNORECASE
    )
    content = re.sub(
        r'^\s*Probability:\s*(.+)$',
        r'<span class="sel-prob"><span class="sel-label">Prob</span> \1</span>',
        content, flags=re.MULTILINE | re.IGNORECASE
    )

    # Convert paragraphs (double newlines)
    paragraphs = re.split(r'\n\n+', content)
    formatted_paragraphs = []

    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        # Skip if already has block-level HTML
        if p.startswith(('<h', '<ul', '<ol', '<hr')):
            formatted_paragraphs.append(p)
        elif p.startswith('<div'):
            # If a <div>...</div> is followed by non-block content,
            # split so the div stays block-level and trailing content gets <p> wrapped
            # Find last </div> and check for trailing content
            last_close = p.rfind('</div>')
            if last_close >= 0:
                after_div = p[last_close + 6:].strip()
                div_part = p[:last_close + 6]
                formatted_paragraphs.append(div_part)
                if after_div:
                    after_div = after_div.replace('\n', '<br>')
                    formatted_paragraphs.append(f'<p>{after_div}</p>')
            else:
                formatted_paragraphs.append(p)
        else:
            # Convert single newlines to <br> within paragraph
            p = p.replace('\n', '<br>')
            formatted_paragraphs.append(f'<p>{p}</p>')

    content = '\n'.join(formatted_paragraphs)

    # Highlight "Punty's Pick:" lines with special styling
    # Match various apostrophe forms and optional colon
    content = re.sub(
        "(<strong>Punty['\u2018\u2019]?s\\s+Pick:?</strong>)",
        r'<span class="puntys-pick">\1</span>',
        content,
        flags=re.IGNORECASE,
    )

    # Remove top spacing on Jockeys/Stables paragraphs (keep tight with Tempo Profile)
    content = re.sub(
        r'<p>(<strong>(?:Jockeys to follow|Stables to respect):?</strong>)',
        r'<p style="margin-top:0.15rem">\1',
        content,
        flags=re.IGNORECASE,
    )

    return content


class HTMLFormatter:
    """Format content for HTML display on public tips pages."""

    @classmethod
    def format(cls, raw_content: str, content_type: str = "early_mail", seed: int = 0) -> str:
        """Format raw content as HTML."""
        return format_html(raw_content, content_type, seed)
