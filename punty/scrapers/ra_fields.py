"""Racing Australia Free Fields scraper.

Scrapes official race fields and form from racingaustralia.horse/FreeFields.
Used as a fallback when PuntingForm is unavailable.
Static HTML page — uses httpx (no Playwright needed).

URL pattern:
  https://www.racingaustralia.horse/FreeFields/AllForm.aspx
    ?Key={YYYYMmmDD},{STATE},{Venue Name}&recentForm=N
"""

import json
import logging
import re
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

import httpx
from bs4 import BeautifulSoup, Tag

from punty.scrapers.track_conditions import _resolve_state

# RA displays times in the venue's local timezone.
# Map state → timezone for conversion to Melbourne time.
_STATE_TZ: dict[str, ZoneInfo] = {
    "VIC": ZoneInfo("Australia/Melbourne"),
    "NSW": ZoneInfo("Australia/Sydney"),
    "ACT": ZoneInfo("Australia/Sydney"),
    "QLD": ZoneInfo("Australia/Brisbane"),     # UTC+10, no DST
    "SA":  ZoneInfo("Australia/Adelaide"),      # UTC+10:30 in summer (ACDT)
    "WA":  ZoneInfo("Australia/Perth"),         # UTC+8, no DST
    "TAS": ZoneInfo("Australia/Hobart"),
    "NT":  ZoneInfo("Australia/Darwin"),        # UTC+9:30, no DST
}

logger = logging.getLogger(__name__)

_BASE_URL = (
    "https://www.racingaustralia.horse/FreeFields/AllForm.aspx"
    "?Key={date_key}%2C{state}%2C{venue}&recentForm=N"
)


def _format_date_key(d: date) -> str:
    """Format date as '2026Feb16' for RA URL."""
    return d.strftime("%Y%b%d")


def _make_slug(text: str) -> str:
    """Convert text to a URL slug."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _parse_stats_str(text: str) -> str | None:
    """Parse '6:1-0-2' format into JSON stats dict string."""
    m = re.search(r"(\d+)\s*:\s*(\d+)-(\d+)-(\d+)", text)
    if m:
        return json.dumps({
            "starts": int(m.group(1)),
            "wins": int(m.group(2)),
            "seconds": int(m.group(3)),
            "thirds": int(m.group(4)),
        })
    return None


async def _resolve_ra_venue_name(venue: str, state: str) -> str:
    """Resolve the exact RA venue name by checking the track conditions page.

    RA Free Fields requires the exact venue name (including sponsor prefix).
    E.g. "Cannon Park" needs "Ladbrokes Cannon Park" in the URL.
    """
    from punty.scrapers.track_conditions import (
        scrape_track_conditions, _match_venue,
    )
    try:
        conditions = await scrape_track_conditions(state)
        for cond in conditions:
            ra_venue = cond.get("venue", "")
            if _match_venue(ra_venue, venue):
                logger.info(f"RA venue name resolved: '{venue}' → '{ra_venue}'")
                return ra_venue
    except Exception as e:
        logger.debug(f"Failed to resolve RA venue name for {venue}: {e}")
    return venue


async def scrape_ra_fields(
    venue: str,
    race_date: date,
    meeting_id: str | None = None,
) -> dict[str, Any] | None:
    """Scrape race fields from Racing Australia Free Fields page.

    Returns dict with "meeting", "races", "runners" keys matching the format
    expected by orchestrator._upsert_meeting_data(), or None on failure.
    """
    state = _resolve_state(venue)
    if not state:
        logger.warning(f"Cannot resolve state for RA fields: {venue}")
        return None

    if not meeting_id:
        venue_slug = _make_slug(venue)
        meeting_id = f"{venue_slug}-{race_date.isoformat()}"

    date_key = _format_date_key(race_date)

    # Try with the raw venue name first
    async with httpx.AsyncClient(timeout=30) as client:
        for attempt_venue in [venue]:
            url = _BASE_URL.format(
                date_key=date_key,
                state=state,
                venue=attempt_venue.replace(" ", "%20"),
            )
            logger.info(f"Scraping RA fields for {venue}: {url}")
            try:
                resp = await client.get(url, follow_redirects=True)
                resp.raise_for_status()
                html = resp.text
                if "raceNum" in html:
                    return _parse_ra_html(html, meeting_id, venue, race_date, state)
            except Exception as e:
                logger.debug(f"RA fields attempt failed for {attempt_venue}: {e}")

        # Venue name didn't work — resolve the exact RA name via track conditions
        ra_venue = await _resolve_ra_venue_name(venue, state)
        if ra_venue != venue:
            url = _BASE_URL.format(
                date_key=date_key,
                state=state,
                venue=ra_venue.replace(" ", "%20"),
            )
            logger.info(f"Retrying RA fields with resolved name '{ra_venue}': {url}")
            try:
                resp = await client.get(url, follow_redirects=True)
                resp.raise_for_status()
                html = resp.text
                if "raceNum" in html:
                    return _parse_ra_html(html, meeting_id, venue, race_date, state)
            except Exception as e:
                logger.error(f"RA fields retry failed for {ra_venue}: {e}")

    logger.warning(f"RA fields: could not find data for {venue}")
    return None


def _parse_ra_html(
    html: str,
    meeting_id: str,
    venue: str,
    race_date: date,
    state: str = "VIC",
) -> dict[str, Any] | None:
    """Parse RA Free Fields HTML into the standard scraper data format.

    HTML structure per race:
      Table A: Race header — <th> has "Race N - TIME NAME (DISTANCE)", row 1 has prize money
      Table B: Runner summary — header row "No | Last 10 | Horse | Trainer | ..."
      Tables C1..Cn: Runner detail tables — first cell starts with "1CALL ME SON..."
      (some form history tables interspersed — first cell like "T7 of 10")
    Meeting metadata is in <div class='race-venue-bottom'> with <b> label : value pairs.
    """
    soup = BeautifulSoup(html, "html.parser")

    # --- Meeting-level metadata from div.race-venue-bottom ---
    meeting_data: dict[str, Any] = {}
    meta_div = soup.find("div", class_="race-venue-bottom")
    if meta_div:
        for bold in meta_div.find_all("b"):
            label = bold.get_text(strip=True).rstrip(":")
            next_node = bold.next_sibling
            if next_node and isinstance(next_node, str):
                val = next_node.strip()
            else:
                continue
            if not val:
                continue
            label_lower = label.lower()
            if "track condition" in label_lower:
                meeting_data["track_condition"] = val
            elif "rail" in label_lower and "position" in label_lower:
                meeting_data["rail_position"] = val
            elif label_lower == "weather":
                meeting_data["weather"] = val
            elif "penetrometer" in label_lower:
                try:
                    meeting_data["penetrometer"] = float(val)
                except ValueError:
                    pass

    # --- Find all tables and classify them ---
    all_tables = soup.find_all("table")
    if not all_tables:
        logger.warning(f"No tables found in RA fields for {venue}")
        return None

    races: list[dict[str, Any]] = []
    runners: list[dict[str, Any]] = []

    i = 0
    while i < len(all_tables):
        table = all_tables[i]

        # Race header: first cell is a <th> containing <span class='raceNum'>
        race_num_span = table.find("span", class_="raceNum")
        if not race_num_span:
            i += 1
            continue

        # Parse race header from the <th> text
        th = race_num_span.parent
        header_text = th.get_text(strip=True) if th else ""

        race_match = re.match(
            r"Race\s+(\d+)\s*-\s*(\d{1,2}:\d{2}\s*[AP]M)\s+(.*)",
            header_text, re.IGNORECASE
        )
        if not race_match:
            i += 1
            continue

        race_num = int(race_match.group(1))
        time_str = race_match.group(2).strip()
        race_name_raw = race_match.group(3).strip()
        race_id = f"{meeting_id}-r{race_num}"

        # Parse start time — RA shows times in the venue's local timezone.
        # Convert to Melbourne local time (AEDT/AEST) for consistency.
        start_time = None
        try:
            from punty.config import MELB_TZ
            venue_tz = _STATE_TZ.get(state, MELB_TZ)
            t = datetime.strptime(time_str.upper().replace(" ", ""), "%I:%M%p")
            local_dt = datetime(
                race_date.year, race_date.month, race_date.day, t.hour, t.minute,
                tzinfo=venue_tz,
            )
            start_time = local_dt.astimezone(MELB_TZ).replace(tzinfo=None)
        except ValueError:
            pass

        # Extract distance
        distance = None
        dist_match = re.search(r"\((\d{3,5})\s*METRES?\)", header_text, re.IGNORECASE)
        if dist_match:
            distance = int(dist_match.group(1))

        # Clean race name — remove distance suffix and "Times displayed..." suffix
        race_name = re.sub(r"\s*\(\d+\s*METRES?\)\s*", "", race_name_raw).strip()
        race_name = re.sub(r"\s*Times\s+displayed.*$", "", race_name, flags=re.IGNORECASE).strip()

        # Extract prize money from the table (row 1 has "Of $27,000...")
        table_text = table.get_text()
        prize_money = None
        prize_match = re.search(r"Of\s+\$([0-9,]+)", table_text)
        if prize_match:
            prize_money = int(prize_match.group(1).replace(",", ""))

        # Extract class
        class_ = None
        class_match = re.search(
            r"(Class\s*\d|Maiden|BM\s*\d+|Benchmark\s*\d+|Group\s*\d|"
            r"Listed|Open\s*Handicap|Handicap|Set\s*Weights?|Restricted|Quality)",
            race_name, re.IGNORECASE
        )
        if class_match:
            class_ = class_match.group(1).strip()

        # Weight type
        weight_type = None
        if re.search(r"Set\s*Weights?", table_text, re.IGNORECASE):
            weight_type = "Set Weights"
        elif re.search(r"Handicap", table_text, re.IGNORECASE):
            weight_type = "Handicap"

        race = {
            "id": race_id,
            "meeting_id": meeting_id,
            "race_number": race_num,
            "name": race_name,
            "distance": distance,
            "class_": class_,
            "prize_money": prize_money,
            "start_time": start_time,
            "status": "scheduled",
            "race_type": "Thoroughbred",
            "age_restriction": None,
            "weight_type": weight_type,
            "field_size": None,
        }

        # Next table should be the runner summary table (header row starts with "No")
        if i + 1 < len(all_tables):
            summary_table = all_tables[i + 1]
            first_row = summary_table.find("tr")
            if first_row:
                first_cells = first_row.find_all(["th", "td"])
                headers = [c.get_text(strip=True) for c in first_cells]
                if headers and headers[0] == "No":
                    race_runners = _parse_summary_table(summary_table, race_id)
                    race["field_size"] = len([r for r in race_runners if not r["scratched"]])

                    # Detail tables follow the summary table — one per runner
                    detail_idx = i + 2
                    while detail_idx < len(all_tables):
                        detail_table = all_tables[detail_idx]
                        # Stop if we hit the next race header
                        if detail_table.find("span", class_="raceNum"):
                            break
                        # Stop if we hit the next summary table
                        dr = detail_table.find("tr")
                        if dr:
                            dc = dr.find_all(["th", "td"])
                            dh = [c.get_text(strip=True) for c in dc]
                            if dh and dh[0] == "No" and len(dh) > 3:
                                break

                        # Runner detail: first cell starts with saddlecloth + horse name
                        first_td = detail_table.find("td")
                        if first_td:
                            td_text = first_td.get_text(strip=True)
                            sc_match = re.match(r"^(\d{1,2})([A-Z])", td_text)
                            if sc_match:
                                saddlecloth = int(sc_match.group(1))
                                for runner in race_runners:
                                    if runner["saddlecloth"] == saddlecloth:
                                        _enrich_runner_from_detail(runner, detail_table)
                                        break

                        detail_idx += 1

                    runners.extend(race_runners)

        races.append(race)
        i += 1

    if not races:
        logger.warning(f"Failed to parse any races from RA fields for {venue}")
        return None

    logger.info(f"RA fields: {len(races)} races, {len(runners)} runners for {venue}")

    return {
        "meeting": meeting_data,
        "races": races,
        "runners": runners,
    }


def _parse_summary_table(table: Tag, race_id: str) -> list[dict[str, Any]]:
    """Parse the runner summary table (No|Last10|Horse|Trainer|Jockey|Barrier|Weight|...)."""
    runners: list[dict[str, Any]] = []
    rows = table.find_all("tr")
    if len(rows) < 2:
        return runners

    # Parse header to find column indices
    header_cells = rows[0].find_all(["th", "td"])
    headers = [c.get_text(strip=True).lower() for c in header_cells]

    col_map: dict[str, int] = {}
    for idx, h in enumerate(headers):
        if h == "no":
            col_map["saddlecloth"] = idx
        elif "last" in h:
            col_map["form"] = idx
        elif h == "horse":
            col_map["horse"] = idx
        elif h == "trainer":
            col_map["trainer"] = idx
        elif h == "jockey":
            col_map["jockey"] = idx
        elif h == "barrier":
            col_map["barrier"] = idx
        elif h == "weight" and "probable" not in h:
            col_map["weight"] = idx
        elif "hcp" in h or "rating" in h:
            col_map["rating"] = idx

    for row in rows[1:]:
        cells = row.find_all("td")
        if len(cells) < 3:
            continue

        def get_cell(key: str) -> str | None:
            idx = col_map.get(key)
            if idx is not None and idx < len(cells):
                return cells[idx].get_text(strip=True) or None
            return None

        sc_text = get_cell("saddlecloth")
        horse_text = get_cell("horse")
        if not sc_text or not horse_text:
            continue

        try:
            saddlecloth = int(sc_text)
        except ValueError:
            continue

        # Clean horse name — remove (NZ), (Blks), EM marker
        horse_name = re.sub(r"\s*\((?:NZ|USA|IRE|GB|FR|GER|SAF|JPN|HK|IT|SIN)\)", "", horse_text)
        horse_name = re.sub(r"\s*\(Blks?\)", "", horse_name)
        horse_name = re.sub(r"\s*EM$", "", horse_name).strip()

        # Check if scratched — look for "SCRATCHED" in the row text
        row_text = row.get_text()
        scratched = "SCRATCHED" in row_text.upper()

        # Barrier
        barrier = None
        barrier_text = get_cell("barrier")
        if barrier_text:
            try:
                barrier = int(barrier_text)
            except ValueError:
                pass

        # Weight
        weight = None
        weight_text = get_cell("weight")
        if weight_text:
            try:
                weight = float(weight_text.replace("kg", ""))
            except ValueError:
                pass

        # Handicap rating
        hcp_rating = None
        rating_text = get_cell("rating")
        if rating_text:
            try:
                hcp_rating = float(rating_text)
            except ValueError:
                pass

        # Jockey
        jockey = get_cell("jockey")

        # Trainer
        trainer = get_cell("trainer")

        # Form
        form = get_cell("form")
        last_five = form[:5] if form else None

        horse_slug = _make_slug(horse_name)
        runner_id = f"{race_id}-{saddlecloth}-{horse_slug}"

        runners.append({
            "id": runner_id,
            "race_id": race_id,
            "horse_name": horse_name,
            "saddlecloth": saddlecloth,
            "barrier": barrier,
            "weight": weight,
            "jockey": jockey,
            "trainer": trainer,
            "form": form,
            "last_five": last_five,
            "career_record": None,
            "horse_age": None,
            "horse_sex": None,
            "horse_colour": None,
            "sire": None,
            "dam": None,
            "dam_sire": None,
            "career_prize_money": None,
            "handicap_rating": hcp_rating,
            "gear": None,
            "first_up_stats": None,
            "second_up_stats": None,
            "track_stats": None,
            "distance_stats": None,
            "track_dist_stats": None,
            "good_track_stats": None,
            "soft_track_stats": None,
            "heavy_track_stats": None,
            "days_since_last_run": None,
            "scratched": scratched,
        })

    return runners


def _enrich_runner_from_detail(runner: dict[str, Any], table: Tag) -> None:
    """Enrich a runner dict with data from its detail table (breeding, stats, gear)."""
    text = table.get_text(separator="\n")

    # Age/Sex/Colour — "4 year old bay gelding" or "4yo bay gelding"
    age_match = re.search(
        r"(\d+)\s*(?:yo|year\s*old)\s+([\w\s/]+?)\s*"
        r"(gelding|mare|filly|colt|stallion|horse|entire|ridgling)",
        text, re.IGNORECASE
    )
    if age_match:
        runner["horse_age"] = int(age_match.group(1))
        colour_raw = age_match.group(2).strip().lower()
        sex_raw = age_match.group(3).strip().lower()
        # Sex
        if sex_raw in ("mare", "filly"):
            runner["horse_sex"] = "Mare"
        elif sex_raw == "gelding":
            runner["horse_sex"] = "Gelding"
        elif sex_raw == "colt":
            runner["horse_sex"] = "Colt"
        else:
            runner["horse_sex"] = "Horse"
        # Colour
        for c in ("bay or brown", "bay", "brown", "chestnut", "grey", "black", "roan"):
            if c in colour_raw:
                runner["horse_colour"] = c.title()
                break

    # Country codes to strip from names
    _COUNTRIES = r"(?:NZ|USA|IRE|GB|FR|GER|SAF|JPN|HK|IT|SIN|AUS)"

    # Sire — text has newlines: "Sire:\n \nThe Autumn Sun\n \nDam:"
    sire_match = re.search(r"Sire:\s*(.+?)\s*Dam:", text, re.DOTALL | re.IGNORECASE)
    if sire_match:
        sire = sire_match.group(1).strip()
        sire = re.sub(r"\s*\(" + _COUNTRIES + r"\)\s*$", "", sire)
        runner["sire"] = sire

    # Dam + Dam Sire — "Dam:\n \nTomelilla (NZ)\n (\nTavistock (NZ)\n)\n...View Pedigree"
    dam_match = re.search(r"Dam:\s*(.+?)\s*(?:View\s*Pedigree|Breeder)", text, re.DOTALL | re.IGNORECASE)
    if dam_match:
        dam_block = dam_match.group(1).strip()
        # Collapse whitespace/newlines
        dam_block = re.sub(r"\s+", " ", dam_block).strip()
        # Pattern: "DamName (Country) (DamSire (Country))" or "DamName (DamSire)"
        # First try with country code before dam sire
        ds_match = re.match(
            r"(.+?)\s*\(" + _COUNTRIES + r"\)\s*\((.+)\)\s*$",
            dam_block
        )
        if ds_match:
            runner["dam"] = ds_match.group(1).strip()
            dam_sire = ds_match.group(2).strip()
            dam_sire = re.sub(r"\s*\(" + _COUNTRIES + r"\)\s*$", "", dam_sire)
            runner["dam_sire"] = dam_sire
        else:
            # Try without country: "DamName (DamSire)"
            ds_match2 = re.match(r"(.+?)\s*\(([^)]+)\)\s*$", dam_block)
            if ds_match2:
                runner["dam"] = ds_match2.group(1).strip()
                dam_sire = ds_match2.group(2).strip()
                dam_sire = re.sub(r"\s*\(" + _COUNTRIES + r"\)\s*$", "", dam_sire)
                runner["dam_sire"] = dam_sire
            else:
                # Just dam name
                dam = re.sub(r"\s*\(" + _COUNTRIES + r"\)\s*$", "", dam_block)
                runner["dam"] = dam

    # Career record
    record_match = re.search(r"(\d{1,3})\s*:\s*(\d+)-(\d+)-(\d+)", text)
    if record_match:
        runner["career_record"] = f"{record_match.group(1)}: {record_match.group(2)}-{record_match.group(3)}-{record_match.group(4)}"

    # Prize money
    prize_match = re.search(r"Prizemoney\s*:?\s*\$([0-9,]+)", text, re.IGNORECASE)
    if prize_match:
        runner["career_prize_money"] = int(prize_match.group(1).replace(",", ""))

    # Gear
    gear_match = re.search(r"\(Blks?[^)]*\)", text)
    if gear_match:
        runner["gear"] = gear_match.group(0).strip("()")

    # Trainer location — "Trainer Name (Location)"
    trainer_loc = re.search(r"Trainer\s*:?\s*[^\n]+?\(([^)]+)\)", text, re.IGNORECASE)
    if trainer_loc:
        runner["trainer_location"] = trainer_loc.group(1).strip()

    # Stats — "1st Up: 2:1-0-0", "Good: 2:0-0-1", etc.
    stats_map = {
        r"1st\s*Up": "first_up_stats",
        r"2nd\s*Up": "second_up_stats",
        r"Track/Dist": "track_dist_stats",
        r"Track(?!/)": "track_stats",
        r"Dist(?:ance)?(?!/)": "distance_stats",
        r"(?<!\w)Good(?:\s|:)": "good_track_stats",
        r"(?<!\w)Soft(?:\s|:)": "soft_track_stats",
        r"(?<!\w)Heavy(?:\s|:)": "heavy_track_stats",
    }
    for pattern, key in stats_map.items():
        stat_match = re.search(
            rf"{pattern}\s*:?\s*(\d+\s*:\s*\d+-\d+-\d+)",
            text, re.IGNORECASE
        )
        if stat_match:
            runner[key] = _parse_stats_str(stat_match.group(1))

    # Days since last run — find most recent race date in form
    # Format: "BEAU 02Feb26:" in the form history
    date_matches = re.findall(r"(\d{1,2}[A-Z][a-z]{2}\d{2})", text)
    if date_matches:
        most_recent = None
        for d_str in date_matches:
            try:
                d = datetime.strptime(d_str, "%d%b%y").date()
                if most_recent is None or d > most_recent:
                    most_recent = d
            except ValueError:
                continue
        if most_recent:
            runner["days_since_last_run"] = (datetime.now().date() - most_recent).days
