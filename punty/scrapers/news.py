"""Lightweight racing news scraper — headlines + snippets for blog context.

Uses Google News RSS for broad Australian racing coverage plus 3 direct sites.
"""

import logging
import re
from datetime import datetime, timedelta
from typing import Optional
from xml.etree import ElementTree

import httpx

from punty.config import melb_today

logger = logging.getLogger(__name__)


SOURCES = [
    {
        "name": "Google News AU Racing",
        "url": "https://news.google.com/rss/search?q=horse+racing+australia&hl=en-AU&gl=AU&ceid=AU:en",
        "type": "rss",
    },
    {
        "name": "7NEWS Racing",
        "url": "https://7news.com.au/sport/horse-racing?page=1",
        "type": "html",
    },
    {
        "name": "Racing.com",
        "url": "https://www.racing.com/news",
        "type": "html",
    },
    {
        "name": "Just Horse Racing",
        "url": "https://www.justhorseracing.com.au/category/news/australian-racing",
        "type": "html",
    },
]


class NewsScraper:
    """Scrape racing news headlines for blog context."""

    def __init__(self):
        self.client = httpx.AsyncClient(
            timeout=15.0,
            follow_redirects=True,
            headers={"User-Agent": "PuntyAI/1.0 NewsBot"},
        )

    async def scrape_headlines(self, max_days: int = 7) -> list[dict]:
        """Scrape headlines from all sources. Returns list of dicts."""
        all_headlines: list[dict] = []

        for source in SOURCES:
            try:
                if source["type"] == "rss":
                    headlines = await self._parse_rss(source["url"], source["name"])
                else:
                    headlines = await self._parse_html(source["url"], source["name"])
                all_headlines.extend(headlines)
                logger.info(f"News: {source['name']} returned {len(headlines)} headlines")
            except Exception as e:
                logger.warning(f"News: {source['name']} failed: {e}")

        # Deduplicate by title similarity
        seen_titles: set[str] = set()
        unique: list[dict] = []
        for h in all_headlines:
            normalised = re.sub(r"\s+", " ", h["title"].lower().strip())
            if normalised not in seen_titles:
                seen_titles.add(normalised)
                unique.append(h)

        # Sort by date descending
        unique.sort(key=lambda h: h.get("date", ""), reverse=True)

        return unique[:50]  # cap at 50 headlines

    async def _parse_rss(self, url: str, source_name: str) -> list[dict]:
        """Parse RSS/Atom feed (Google News)."""
        resp = await self.client.get(url)
        resp.raise_for_status()

        root = ElementTree.fromstring(resp.text)
        headlines = []

        # RSS 2.0 format: channel/item
        for item in root.findall(".//item"):
            title_el = item.find("title")
            link_el = item.find("link")
            pubdate_el = item.find("pubDate")
            desc_el = item.find("description")

            if not title_el or not title_el.text:
                continue

            date_str = ""
            if pubdate_el is not None and pubdate_el.text:
                date_str = _parse_rss_date(pubdate_el.text)

            snippet = ""
            if desc_el is not None and desc_el.text:
                # Strip HTML tags from description
                snippet = re.sub(r"<[^>]+>", "", desc_el.text)[:200]

            headlines.append({
                "title": title_el.text.strip(),
                "url": link_el.text.strip() if link_el is not None and link_el.text else "",
                "snippet": snippet,
                "source": source_name,
                "date": date_str,
            })

        return headlines

    async def _parse_html(self, url: str, source_name: str) -> list[dict]:
        """Parse HTML news page for headlines using regex patterns."""
        resp = await self.client.get(url)
        resp.raise_for_status()
        html = resp.text
        headlines = []

        if "7news.com.au" in url:
            headlines = self._parse_7news(html, source_name)
        elif "racing.com" in url:
            headlines = self._parse_racing_com(html, source_name)
        elif "justhorseracing" in url:
            headlines = self._parse_jhr(html, source_name)

        return headlines

    def _parse_7news(self, html: str, source: str) -> list[dict]:
        """Extract headlines from 7NEWS racing page."""
        results = []
        # Look for article links with headlines
        pattern = r'<a[^>]*href="(https://7news\.com\.au/sport/horse-racing/[^"]+)"[^>]*>\s*<[^>]*>([^<]+)</'
        for match in re.finditer(pattern, html):
            url, title = match.group(1), match.group(2).strip()
            if title and len(title) > 15:
                results.append({"title": title, "url": url, "snippet": "", "source": source, "date": ""})
        # Fallback: look for heading tags with text
        if not results:
            for match in re.finditer(r'<h[23][^>]*>.*?<a[^>]*href="([^"]*)"[^>]*>([^<]+)</a>', html, re.DOTALL):
                url, title = match.group(1), match.group(2).strip()
                if title and len(title) > 15:
                    full_url = url if url.startswith("http") else f"https://7news.com.au{url}"
                    results.append({"title": title, "url": full_url, "snippet": "", "source": source, "date": ""})
        return results[:15]

    def _parse_racing_com(self, html: str, source: str) -> list[dict]:
        """Extract headlines from racing.com news page."""
        results = []
        # Look for article cards with titles
        for match in re.finditer(
            r'<a[^>]*href="(/news/[^"]+)"[^>]*>.*?<h[234][^>]*>([^<]+)</h',
            html, re.DOTALL,
        ):
            url, title = match.group(1), match.group(2).strip()
            if title and len(title) > 10:
                results.append({
                    "title": title,
                    "url": f"https://www.racing.com{url}",
                    "snippet": "",
                    "source": source,
                    "date": "",
                })
        return results[:15]

    def _parse_jhr(self, html: str, source: str) -> list[dict]:
        """Extract headlines from Just Horse Racing."""
        results = []
        for match in re.finditer(
            r'<h[23][^>]*>\s*<a[^>]*href="(https://www\.justhorseracing\.com\.au/[^"]+)"[^>]*>([^<]+)</a>',
            html,
        ):
            url, title = match.group(1), match.group(2).strip()
            if title and len(title) > 10:
                results.append({"title": title, "url": url, "snippet": "", "source": source, "date": ""})
        return results[:15]

    async def close(self):
        await self.client.aclose()


def _parse_rss_date(date_str: str) -> str:
    """Parse RSS date string to ISO format."""
    # RFC 822: "Thu, 06 Feb 2026 02:30:00 GMT"
    for fmt in (
        "%a, %d %b %Y %H:%M:%S %Z",
        "%a, %d %b %Y %H:%M:%S %z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
    ):
        try:
            dt = datetime.strptime(date_str.strip(), fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return ""
