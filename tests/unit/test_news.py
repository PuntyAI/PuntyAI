"""Tests for news scraper — RSS parsing, HTML parsing, deduplication."""

import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock

from punty.scrapers.news import NewsScraper, _parse_rss_date


# ── Helper: create a scraper with mocked HTTP client ─────────────────────────

def _mock_scraper(*responses):
    """Create a NewsScraper with a mock client returning given responses in order."""
    scraper = NewsScraper()
    mock_client = MagicMock()
    mock_client.get = AsyncMock(side_effect=list(responses))
    mock_client.aclose = AsyncMock()
    scraper.client = mock_client
    return scraper


def _mock_response(text, status=200):
    """Create a mock httpx response."""
    resp = MagicMock()
    resp.text = text
    resp.status_code = status
    resp.raise_for_status = MagicMock()
    return resp


# ── RSS Date Parsing ─────────────────────────────────────────────────────────


class TestParseRssDate:

    def test_rfc822_gmt(self):
        assert _parse_rss_date("Thu, 06 Feb 2026 02:30:00 GMT") == "2026-02-06"

    def test_rfc822_timezone_offset(self):
        assert _parse_rss_date("Mon, 10 Feb 2026 14:00:00 +1100") == "2026-02-10"

    def test_iso_format_with_z(self):
        assert _parse_rss_date("2026-02-08T09:15:00Z") == "2026-02-08"

    def test_iso_format_with_offset(self):
        assert _parse_rss_date("2026-02-08T09:15:00+0000") == "2026-02-08"

    def test_invalid_date_returns_empty(self):
        assert _parse_rss_date("not a date") == ""

    def test_empty_string(self):
        assert _parse_rss_date("") == ""


# ── RSS Parsing ──────────────────────────────────────────────────────────────


SAMPLE_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
<channel>
<title>Horse Racing Australia - Google News</title>
<item>
  <title>Champion Jockey Suspended After Melbourne Cup</title>
  <link>https://example.com/article1</link>
  <pubDate>Thu, 13 Feb 2026 08:00:00 GMT</pubDate>
  <description>&lt;b&gt;Big news&lt;/b&gt; from the racing world</description>
</item>
<item>
  <title>Black Caviar Lightning Preview: Who Will Win?</title>
  <link>https://example.com/article2</link>
  <pubDate>Wed, 12 Feb 2026 14:30:00 GMT</pubDate>
  <description>Full preview of tomorrow&apos;s Group 1 sprint</description>
</item>
<item>
  <title></title>
  <link>https://example.com/empty</link>
</item>
</channel>
</rss>"""


class TestRssParsing:

    async def test_extracts_items(self):
        scraper = _mock_scraper(_mock_response(SAMPLE_RSS))
        results = await scraper._parse_rss("https://news.google.com/rss", "Google")
        assert len(results) == 2  # empty title item skipped

    async def test_extracts_title(self):
        scraper = _mock_scraper(_mock_response(SAMPLE_RSS))
        results = await scraper._parse_rss("https://news.google.com/rss", "Google")
        assert results[0]["title"] == "Champion Jockey Suspended After Melbourne Cup"

    async def test_extracts_date(self):
        scraper = _mock_scraper(_mock_response(SAMPLE_RSS))
        results = await scraper._parse_rss("https://news.google.com/rss", "Google")
        assert results[0]["date"] == "2026-02-13"
        assert results[1]["date"] == "2026-02-12"

    async def test_strips_html_from_snippet(self):
        scraper = _mock_scraper(_mock_response(SAMPLE_RSS))
        results = await scraper._parse_rss("https://news.google.com/rss", "Google")
        assert "<b>" not in results[0]["snippet"]
        assert "Big news" in results[0]["snippet"]

    async def test_sets_source(self):
        scraper = _mock_scraper(_mock_response(SAMPLE_RSS))
        results = await scraper._parse_rss("https://news.google.com/rss", "TestSource")
        assert all(r["source"] == "TestSource" for r in results)

    async def test_extracts_url(self):
        scraper = _mock_scraper(_mock_response(SAMPLE_RSS))
        results = await scraper._parse_rss("https://news.google.com/rss", "Google")
        assert results[0]["url"] == "https://example.com/article1"


# ── 7NEWS HTML Parsing ───────────────────────────────────────────────────────


SAMPLE_7NEWS_HTML = """
<html><body>
<div class="content">
  <a href="/sport/horse-racing/champion-jockey-cops-sanction-c-12345">
    <span>Champion jockey cops heavy sanction after $2m triumph</span>
  </a>
  <a href="/sport/horse-racing/derby-winner-first-up-victory-c-67890">
    Victoria Derby winner powers to first-up victory at Caulfield
  </a>
  <a href="/sport/horse-racing">Horse Racing</a>
  <a href="/sport/horse-racing/">All Racing News</a>
  <a href="/sport/horse-racing/short-c-111">Short</a>
  <a href="/sport/horse-racing/cox-plate-winner-retired-c-99999">
    Cox Plate winner retired on the spot after sudden realisation in race
  </a>
  <a href="/sport/horse-racing/champion-jockey-cops-sanction-c-12345">
    Champion jockey cops heavy sanction after $2m triumph duplicate
  </a>
</div>
</body></html>
"""


class TestParse7news:

    def test_extracts_article_links(self):
        scraper = NewsScraper()
        results = scraper._parse_7news(SAMPLE_7NEWS_HTML, "7NEWS")
        assert len(results) == 3

    def test_skips_nav_links(self):
        scraper = NewsScraper()
        results = scraper._parse_7news(SAMPLE_7NEWS_HTML, "7NEWS")
        urls = [r["url"] for r in results]
        assert "https://7news.com.au/sport/horse-racing" not in urls
        assert "https://7news.com.au/sport/horse-racing/" not in urls

    def test_skips_short_text(self):
        scraper = NewsScraper()
        results = scraper._parse_7news(SAMPLE_7NEWS_HTML, "7NEWS")
        titles = [r["title"] for r in results]
        assert not any("Short" == t for t in titles)

    def test_deduplicates_same_href(self):
        scraper = NewsScraper()
        results = scraper._parse_7news(SAMPLE_7NEWS_HTML, "7NEWS")
        urls = [r["url"] for r in results]
        champion_urls = [u for u in urls if "champion-jockey" in u]
        assert len(champion_urls) == 1

    def test_builds_full_url_from_relative(self):
        scraper = NewsScraper()
        results = scraper._parse_7news(SAMPLE_7NEWS_HTML, "7NEWS")
        assert all(r["url"].startswith("https://7news.com.au") for r in results)

    def test_sets_source(self):
        scraper = NewsScraper()
        results = scraper._parse_7news(SAMPLE_7NEWS_HTML, "MySource")
        assert all(r["source"] == "MySource" for r in results)

    def test_empty_html_returns_empty(self):
        scraper = NewsScraper()
        results = scraper._parse_7news("<html><body></body></html>", "7NEWS")
        assert results == []

    def test_caps_at_15(self):
        links = "\n".join(
            f'<a href="/sport/horse-racing/article-{i}-c-{i}">This is a long enough headline for article number {i} here</a>'
            for i in range(20)
        )
        html = f"<html><body>{links}</body></html>"
        scraper = NewsScraper()
        results = scraper._parse_7news(html, "7NEWS")
        assert len(results) == 15


# ── Deduplication & scrape_headlines ─────────────────────────────────────────


class TestScrapeHeadlines:

    async def test_deduplicates_by_title(self):
        rss_resp = _mock_response("""<?xml version="1.0"?>
        <rss><channel>
        <item><title>Same Headline Here</title><link>https://a.com</link></item>
        </channel></rss>""")
        html_resp = _mock_response("""<html><body>
        <a href="/sport/horse-racing/same-headline-c-1">Same Headline Here</a>
        </body></html>""")

        scraper = _mock_scraper(rss_resp, html_resp)
        results = await scraper.scrape_headlines()
        titles = [r["title"] for r in results]
        assert titles.count("Same Headline Here") == 1

    async def test_caps_at_50(self):
        items = "\n".join(
            f"<item><title>Headline {i}</title><link>https://x.com/{i}</link></item>"
            for i in range(60)
        )
        rss_resp = _mock_response(f'<?xml version="1.0"?><rss><channel>{items}</channel></rss>')
        html_resp = _mock_response("<html><body></body></html>")

        scraper = _mock_scraper(rss_resp, html_resp)
        results = await scraper.scrape_headlines()
        assert len(results) == 50

    async def test_sorts_by_date_descending(self):
        items = """
        <item><title>Old</title><link>https://x.com/1</link><pubDate>Mon, 10 Feb 2026 08:00:00 GMT</pubDate></item>
        <item><title>New</title><link>https://x.com/2</link><pubDate>Thu, 13 Feb 2026 08:00:00 GMT</pubDate></item>
        """
        rss_resp = _mock_response(f'<?xml version="1.0"?><rss><channel>{items}</channel></rss>')
        html_resp = _mock_response("<html><body></body></html>")

        scraper = _mock_scraper(rss_resp, html_resp)
        results = await scraper.scrape_headlines()
        assert results[0]["title"] == "New"
        assert results[1]["title"] == "Old"

    async def test_combines_multiple_sources(self):
        rss_resp = _mock_response("""<?xml version="1.0"?>
        <rss><channel>
        <item><title>RSS Article</title><link>https://a.com</link></item>
        </channel></rss>""")
        html_resp = _mock_response("""<html><body>
        <a href="/sport/horse-racing/html-article-c-1">HTML Article From Seven News</a>
        </body></html>""")

        scraper = _mock_scraper(rss_resp, html_resp)
        results = await scraper.scrape_headlines()
        titles = [r["title"] for r in results]
        assert "RSS Article" in titles
        assert "HTML Article From Seven News" in titles


# ── Error Handling ───────────────────────────────────────────────────────────


class TestErrorHandling:

    async def test_source_failure_doesnt_break_others(self):
        async def side_effect(url):
            if "google" in url:
                raise httpx.ConnectError("Connection refused")
            return _mock_response("""<html><body>
            <a href="/sport/horse-racing/working-article-c-1">This article works perfectly fine and has enough text</a>
            </body></html>""")

        scraper = NewsScraper()
        mock_client = MagicMock()
        mock_client.get = AsyncMock(side_effect=side_effect)
        scraper.client = mock_client
        results = await scraper.scrape_headlines()
        assert len(results) >= 1

    async def test_malformed_rss_handled(self):
        rss_resp = _mock_response("not xml at all")
        html_resp = _mock_response("<html><body></body></html>")

        scraper = _mock_scraper(rss_resp, html_resp)
        results = await scraper.scrape_headlines()
        assert isinstance(results, list)
