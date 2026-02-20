"""Tests for early mail social media truncation."""

import pytest

from punty.formatters.truncate import truncate_for_socials, CUTOFF_RACE


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _build_early_mail(num_races: int = 8, include_sequences: bool = True) -> str:
    """Build a realistic mock early mail with N races."""
    parts = [
        "*PUNTY EARLY MAIL – Cranbourne (2026-02-20)*",
        "Rightio Legends, Cranbourne on a Good 4...",
        "",
        "### 2) *MEET SNAPSHOT*",
        "*Track:* Cranbourne, 1000m to 2025m card",
        "*Rail:* True Entire Circuit",
        "*Official going:* Good 4",
        "",
        "### 3) *PUNTY'S BIG 3 + MULTI*",
        "*1 - Phineas* (Race 1, No.5) — $2.20",
        "*2 - Katsu* (Race 4, No.1) — $2.90",
        "*3 - Stage n Screen* (Race 6, No.1) — $1.98",
        "Multi (all three to win): $10 x ~12.62 = ~$126.20 collect",
        "",
        "### 4) *RACE-BY-RACE*",
    ]

    for r in range(1, num_races + 1):
        parts.extend([
            "",
            f"*Race {r} – The Great Race {r}*",
            f"*Race type:* Bm66, {1000 + r * 100}m",
            f"*Map & tempo:* Moderate pace expected",
            f"**Punty read:** This is race {r} and it looks interesting.",
            "",
            f"*Top 3 + Roughie ($20 pool)*",
            f"*1. Horse{r}A* (No.1) — $3.00 / $1.80",
            f"   Win: 30% | Place: 75% | Value: 1.10x",
            f"   Bet: $8 Win, return $24",
            f"   Why: Top pick for race {r}.",
            f"*2. Horse{r}B* (No.2) — $5.00 / $2.50",
            f"   Probability: 20% | Value: 1.20x",
            f"   Bet: $5 Place, return $12.50",
            f"   Why: Solid each way chance.",
            f"*3. Horse{r}C* (No.3) — $8.00 / $3.50",
            f"   Probability: 15% | Value: 1.40x",
            f"   Bet: $4 Place, return $14",
            f"   Why: Value runner at odds.",
            "",
            f"*Roughie: Horse{r}D* (No.4) — $15.00 / $5.50",
            f"Bet: $3 Place, return $16.50",
            "",
            f"*Degenerate Exotic of the Race*",
            f"Trifecta Box: 1, 2, 3 — $15",
            f"6 combos — 250% flexi",
            "",
            f"*Punty's Pick:* Horse{r}A (No.1) $3.00 Win",
        ])

    if include_sequences:
        parts.extend([
            "",
            "### 5) *SEQUENCE LANES — SINGLE OPTIMISED TICKET*",
            "EARLY QUADDIE (R1–R4)",
            "Smart: 1, 2 / 3, 4 / 1 / 2, 3, 4 (24 combos x $2.08 = $50) — 208% flexi",
            "*Punty's take:* Anchor R3, go wide on R4.",
            "",
            "QUADDIE (R5–R8)",
            "Smart: 1 / 2, 3 / 1, 4 / 2, 3 (12 combos x $4.17 = $50) — 417% flexi",
            "*Punty's take:* Banker in R5.",
        ])

    parts.extend([
        "",
        "### 6) *NUGGETS FROM THE TRACK*",
        "*1 - First Nugget*",
        "Something interesting about the track.",
        "*2 - Second Nugget*",
        "Another observation.",
        "",
        "### 7) *FIND OUT MORE*",
        "Want to know more about Punty? Check out https://punty.ai",
        "",
        "### 8) *FINAL WORD FROM THE SICKO SANCTUARY*",
        "Stay sharp, stay hydrated, and remember — it's a marathon, not a sprint.",
        "Gamble Responsibly.",
    ])

    return "\n".join(parts)


# ──────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────

class TestTruncateForSocials:
    def test_truncates_after_race_2(self):
        """R1-R2 should be present, R3+ should be replaced with teaser."""
        content = _build_early_mail(8)
        result = truncate_for_socials(content, meeting_id="cranbourne-2026-02-20")

        # R1 and R2 should be fully present
        assert "*Race 1 – The Great Race 1*" in result
        assert "*Race 2 – The Great Race 2*" in result
        assert "Horse1A" in result
        assert "Horse2A" in result

        # R3+ should NOT be present
        assert "*Race 3 – The Great Race 3*" not in result
        assert "Horse3A" not in result
        assert "*Race 8 – The Great Race 8*" not in result
        assert "Horse8A" not in result

    def test_preserves_header_sections(self):
        """Title, snapshot, and Big 3 should all be preserved."""
        content = _build_early_mail(8)
        result = truncate_for_socials(content)

        assert "*PUNTY EARLY MAIL" in result
        assert "MEET SNAPSHOT" in result
        assert "BIG 3" in result
        assert "Phineas" in result
        assert "Katsu" in result

    def test_preserves_sequence_lanes(self):
        """Quaddie/Big6 section should be kept."""
        content = _build_early_mail(8, include_sequences=True)
        result = truncate_for_socials(content, meeting_id="cranbourne-2026-02-20")

        assert "SEQUENCE LANES" in result
        assert "EARLY QUADDIE" in result
        assert "QUADDIE (R5" in result

    def test_preserves_final_word(self):
        """Responsible gambling section should be kept."""
        content = _build_early_mail(8)
        result = truncate_for_socials(content)

        assert "FINAL WORD" in result
        assert "Gamble Responsibly" in result

    def test_removes_nuggets_and_find_out_more(self):
        """NUGGETS and FIND OUT MORE sections should be removed."""
        content = _build_early_mail(8)
        result = truncate_for_socials(content)

        assert "NUGGETS FROM THE TRACK" not in result
        assert "FIND OUT MORE" not in result

    def test_no_truncation_few_races(self):
        """2-race meeting should not be truncated."""
        content = _build_early_mail(2, include_sequences=False)
        result = truncate_for_socials(content)

        # Should be unchanged (minus nuggets/find out more which are after final word)
        assert "*Race 1 – The Great Race 1*" in result
        assert "*Race 2 – The Great Race 2*" in result
        assert "Horse2A" in result

    def test_cta_contains_link(self):
        """Truncated output should contain CTA with punty.ai link."""
        content = _build_early_mail(8)
        result = truncate_for_socials(content)

        assert "punty.ai" in result

    def test_meeting_id_in_cta(self):
        """CTA should deep-link to the specific meeting page."""
        content = _build_early_mail(8)
        result = truncate_for_socials(content, meeting_id="cranbourne-2026-02-20")

        assert "punty.ai/tips/cranbourne-2026-02-20" in result

    def test_teaser_mentions_remaining_count(self):
        """Teaser should mention how many races remain."""
        content = _build_early_mail(8)
        result = truncate_for_socials(content)

        # 8 races total, showing 2, so 6 remaining
        assert "Six more races" in result
        assert "R3" in result
        assert "R8" in result

    def test_teaser_has_separator(self):
        """Teaser should start with a visual separator."""
        content = _build_early_mail(8)
        result = truncate_for_socials(content)

        assert "━━━━" in result

    def test_3_race_meeting_truncates_race_3(self):
        """3-race meeting should truncate R3, keep R1-R2."""
        content = _build_early_mail(3, include_sequences=False)
        result = truncate_for_socials(content, meeting_id="test-2026-01-01")

        assert "*Race 1 –" in result
        assert "*Race 2 –" in result
        assert "*Race 3 –" not in result
        assert "One more race" in result

    def test_no_content_returns_unchanged(self):
        """Non-early-mail content should pass through unchanged."""
        content = "Just a regular paragraph with no race headings."
        result = truncate_for_socials(content)
        assert result == content
