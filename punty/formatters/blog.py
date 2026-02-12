"""Blog formatting — markdown to HTML and social media teasers."""

import re
from typing import Optional


def format_blog_html(raw_content: str) -> str:
    """Convert blog markdown to styled HTML for the public site.

    Handles: headings, bold, italic, lists, horizontal rules, paragraphs.
    Emojis are passed through as-is.
    """
    if not raw_content:
        return ""

    lines = raw_content.split("\n")
    html_lines: list[str] = []
    in_list = False

    for line in lines:
        stripped = line.strip()

        # Horizontal rule
        if stripped in ("---", "***", "___"):
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append('<hr class="blog-divider">')
            continue

        # Headings
        if stripped.startswith("### "):
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            text = _inline_format(stripped[4:])
            html_lines.append(f'<h3 class="blog-h3">{text}</h3>')
            continue
        if stripped.startswith("## "):
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            text = _inline_format(stripped[3:])
            html_lines.append(f'<h2 class="blog-h2">{text}</h2>')
            continue
        if stripped.startswith("# "):
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            text = _inline_format(stripped[2:])
            html_lines.append(f'<h1 class="blog-h1">{text}</h1>')
            continue

        # List items
        if stripped.startswith("- ") or stripped.startswith("* "):
            if not in_list:
                html_lines.append('<ul class="blog-list">')
                in_list = True
            text = _inline_format(stripped[2:])
            html_lines.append(f"<li>{text}</li>")
            continue

        # Close list if not a list item
        if in_list and stripped:
            html_lines.append("</ul>")
            in_list = False

        # Empty line = paragraph break
        if not stripped:
            html_lines.append("")
            continue

        # Italic block (entire line wrapped in * *)
        if stripped.startswith("*") and stripped.endswith("*") and not stripped.startswith("**"):
            text = _inline_format(stripped)
            html_lines.append(f'<p class="blog-subtitle">{text}</p>')
            continue

        # Regular paragraph
        text = _inline_format(stripped)
        html_lines.append(f"<p>{text}</p>")

    if in_list:
        html_lines.append("</ul>")

    return "\n".join(html_lines)


def _inline_format(text: str) -> str:
    """Apply inline formatting: bold, italic, links."""
    # Bold: **text**
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    # Italic: *text*
    text = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", text)
    # Links: [text](url)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2" target="_blank" rel="noopener">\1</a>', text)
    return text


def format_blog_teaser(raw_content: str, blog_url: str) -> str:
    """Create a Twitter/Facebook teaser from blog content.

    Extracts the opening paragraph + awards summary, adds link.
    """
    if not raw_content:
        return ""

    lines = raw_content.strip().split("\n")

    # Find first substantial paragraph (skip title/header lines)
    opening = ""
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("*FROM"):
            continue
        if stripped.startswith("---"):
            continue
        if len(stripped) > 50:
            opening = stripped[:200]
            if len(stripped) > 200:
                opening = opening.rsplit(" ", 1)[0] + "..."
            break

    # Find awards section
    awards_lines: list[str] = []
    in_awards = False
    for line in lines:
        stripped = line.strip()
        if "PUNTY AWARDS" in stripped.upper():
            in_awards = True
            continue
        if in_awards:
            if stripped.startswith("###") or stripped.startswith("---"):
                break
            if stripped.startswith("- **") or stripped.startswith("* **"):
                # Extract just the award name and winner
                clean = re.sub(r"\*\*([^*]+)\*\*", r"\1", stripped[2:])
                awards_lines.append(clean.split("—")[0].strip() if "—" in clean else clean[:60])

    teaser_parts = []
    if opening:
        teaser_parts.append(opening)
    if awards_lines:
        teaser_parts.append("\n".join(f"🏆 {a}" for a in awards_lines[:3]))
    teaser_parts.append(f"\nRead the full blog: {blog_url}")

    return "\n\n".join(teaser_parts)


def extract_blog_title(raw_content: str) -> str:
    """Extract blog title from content, or generate a default."""
    if not raw_content:
        return "From the Horse's Mouth"

    # Look for the first italic title line: *FROM THE HORSE'S MOUTH — Week of ...*
    for line in raw_content.split("\n")[:10]:
        stripped = line.strip()
        if stripped.startswith("*FROM") and stripped.endswith("*"):
            return stripped.strip("*").strip()
        if stripped.startswith("# FROM"):
            return stripped.lstrip("# ").strip()

    return "From the Horse's Mouth"


def generate_blog_slug(week_start) -> str:
    """Generate URL-friendly slug from week start date."""
    if hasattr(week_start, "isoformat"):
        return f"from-the-horses-mouth-{week_start.isoformat()}"
    return f"from-the-horses-mouth-{week_start}"
