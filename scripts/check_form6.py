"""Scrape the actual rendered full form content from the DOM."""
import asyncio
import json
import sys
sys.path.insert(0, "/opt/puntyai")

async def main():
    from punty.scrapers.playwright_base import new_page

    async with new_page() as page:
        await page.goto(
            "https://www.racing.com/form/2026-02-03/bet365-seymour/race/3",
            wait_until="load",
        )
        await page.wait_for_timeout(5000)

        try:
            btn = page.locator("text=Full Form").first
            if await btn.is_visible(timeout=2000):
                await btn.click()
                await page.wait_for_timeout(4000)
        except Exception:
            pass

        # Get the full form content from DOM
        form_html = await page.evaluate("""() => {
            // Find full form containers
            const containers = document.querySelectorAll('[id*="full-form"], [class*="full-form"], [class*="fullForm"]');
            if (containers.length > 0) {
                return Array.from(containers).map(c => ({
                    id: c.id,
                    cls: c.className,
                    html: c.innerHTML.substring(0, 500),
                    text: c.textContent.substring(0, 500)
                }));
            }

            // Try finding the form table/grid
            const tables = document.querySelectorAll('table');
            const result = [];
            for (const t of tables) {
                const text = t.textContent.substring(0, 300);
                if (text.includes('Margin') || text.includes('Jockey') || text.includes('Track')) {
                    result.push({tag: 'table', text: text});
                }
            }
            if (result.length > 0) return result;

            // Last resort: get __NEXT_DATA__ or similar
            const scripts = document.querySelectorAll('script');
            for (const s of scripts) {
                if (s.id === '__NEXT_DATA__') {
                    return [{script: s.textContent.substring(0, 2000)}];
                }
            }

            return [{msg: 'nothing found', body: document.body.textContent.substring(0, 1000)}];
        }""")

        print(json.dumps(form_html, indent=2)[:3000])

asyncio.run(main())
