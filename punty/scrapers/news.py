"""Lightweight racing news scraper — headlines + snippets for blog context.

Uses Google News RSS as primary source plus 7NEWS HTML as secondary.
"""

import logging
import re
from datetime import datetime
from typing import Optional
from xml.etree import ElementTree

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Browser-like User-Agent to avoid bot blocking
_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

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
]


class NewsScraper:
    """Scrape racing news headlines for blog context."""

    def __init__(self):
        self.client = httpx.AsyncClient(
            timeout=15.0,
            follow_redirects=True,
            headers={"User-Agent": _UA},
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

        # Deduplicate by normalised title
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

            if title_el is None or not title_el.text:
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
        """Parse HTML news page for headlines using BeautifulSoup."""
        resp = await self.client.get(url)
        resp.raise_for_status()

        if "7news.com.au" in url:
            return self._parse_7news(resp.text, source_name)

        return []

    def _parse_7news(self, html: str, source: str) -> list[dict]:
        """Extract headlines from 7NEWS racing page using BeautifulSoup."""
        soup = BeautifulSoup(html, "lxml")
        results = []
        seen_hrefs: set[str] = set()

        # Find all links to individual horse racing articles
        links = soup.find_all(
            "a",
            href=lambda h: (
                h
                and "/sport/horse-racing/" in h
                and h not in ("/sport/horse-racing", "/sport/horse-racing/")
            ),
        )

        for a in links:
            href = a.get("href", "")
            if href in seen_hrefs:
                continue

            # Get the visible text — could be the headline
            text = a.get_text(strip=True)

            # Skip navigation/category links (too short to be headlines)
            if not text or len(text) < 20:
                continue

            # Build full URL if relative
            full_url = href if href.startswith("http") else f"https://7news.com.au{href}"
            seen_hrefs.add(href)

            results.append({
                "title": text[:200],
                "url": full_url,
                "snippet": "",
                "source": source,
                "date": "",
            })

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
