"""Unit tests for early mail parser."""

import pytest
from punty.results.parser import (
    parse_early_mail,
    _parse_big3,
    _parse_race_sections,
    _parse_sequences,
    _normalize_bet_type,
)


class TestNormalizeBetType:
    """Tests for bet type normalization."""

    def test_normalize_win(self):
        assert _normalize_bet_type("Win") == "win"
        assert _normalize_bet_type("WIN") == "win"

    def test_normalize_place(self):
        assert _normalize_bet_type("Place") == "place"
        assert _normalize_bet_type("PLACE") == "place"

    def test_normalize_each_way(self):
        assert _normalize_bet_type("Each Way") == "each_way"
        assert _normalize_bet_type("E/W") == "each_way"
        assert _normalize_bet_type("each way") == "each_way"

    def test_normalize_saver_win(self):
        assert _normalize_bet_type("Saver Win") == "saver_win"
        assert _normalize_bet_type("Win (Saver)") == "saver_win"


class TestParseBig3:
    """Tests for Big 3 parsing."""

    def test_parse_big3_standard_format(self):
        content = """
*PUNTY'S BIG 3 + MULTI*
1) *GOLD RUSH* (Race 2, No.5) — $4.50
   Confidence: high
   Why: Best form
2) *SILVER STAR* (Race 4, No.3) — $6.00
   Confidence: med
   Why: Good value
3) *BRONZE MEDAL* (Race 6, No.8) — $12.00
   Confidence: low
   Why: Live chance
Multi (all three to win): 10U × ~324.00 = ~3240U collect

*Race 1 – First Race*
"""
        counter = [0]

        def next_id():
            counter[0] += 1
            return f"pk-test-{counter[0]:03d}"

        picks = _parse_big3(content, "test-content", "test-meeting", next_id)

        assert len(picks) == 4  # 3 horses + 1 multi
        assert picks[0]["horse_name"] == "GOLD RUSH"
        assert picks[0]["race_number"] == 2
        assert picks[0]["saddlecloth"] == 5
        assert picks[0]["odds_at_tip"] == 4.50
        assert picks[0]["tip_rank"] == 1
        assert picks[0]["pick_type"] == "big3"

        assert picks[1]["horse_name"] == "SILVER STAR"
        assert picks[2]["horse_name"] == "BRONZE MEDAL"

        # Multi pick
        assert picks[3]["pick_type"] == "big3_multi"
        assert picks[3]["multi_odds"] == 324.00

    def test_parse_big3_bold_format_unicode_quote(self):
        """Test **bold** headers with Unicode right single quote."""
        content = """
**PUNTY\u2019S BIG 3 + MULTI**
*1 - GOLD RUSH* (Race 2, No.5) \u2014 $4.50
   Why: Best form
*2 - SILVER STAR* (Race 4, No.3) \u2014 $6.00
   Why: Good value
*3 - BRONZE MEDAL* (Race 6, No.8) \u2014 $12.00
   Why: Live chance
Multi (all three to win): 10U \u00d7 ~324.00 = ~3240U collect

**RACE-BY-RACE**
"""
        counter = [0]

        def next_id():
            counter[0] += 1
            return f"pk-test-{counter[0]:03d}"

        picks = _parse_big3(content, "test-content", "test-meeting", next_id)

        assert len(picks) == 4  # 3 horses + 1 multi
        assert picks[0]["horse_name"] == "GOLD RUSH"
        assert picks[0]["race_number"] == 2
        assert picks[0]["saddlecloth"] == 5
        assert picks[0]["odds_at_tip"] == 4.50
        assert picks[1]["horse_name"] == "SILVER STAR"
        assert picks[2]["horse_name"] == "BRONZE MEDAL"
        assert picks[3]["pick_type"] == "big3_multi"

    def test_parse_big3_no_multi(self):
        content = """
*PUNTY'S BIG 3*
1) *HORSE A* (Race 1, No.1) — $3.00

*Race 1*
"""
        counter = [0]

        def next_id():
            counter[0] += 1
            return f"pk-test-{counter[0]:03d}"

        picks = _parse_big3(content, "test-content", "test-meeting", next_id)

        # Should find the horse but no multi
        assert len(picks) >= 1
        assert picks[0]["horse_name"] == "HORSE A"


class TestParseRaceSections:
    """Tests for race-by-race selection parsing."""

    def test_parse_selections_with_place_odds(self):
        content = """
*Race 1 – Test Race*
Race type: Maiden, 1200m

*Top 3 + Roughie ($20 pool)*
1. *FAST HORSE* (No.1) — $3.50 / $1.45
   Bet: $10 Win, return $35.00
   Confidence: high
2. *QUICK RUNNER* (No.2) — $5.00 / $1.80
   Bet: $6 Place, return $10.80
   Confidence: med
3. *STEADY EDDIE* (No.3) — $8.00 / $2.50
   Bet: $4 Each Way, return $42.00
   Confidence: low
Roughie: *LONGSHOT* (No.8) — $21.00 / $5.00
Bet: Exotics only
"""
        counter = [0]

        def next_id():
            counter[0] += 1
            return f"pk-test-{counter[0]:03d}"

        picks = _parse_race_sections(content, "test-content", "test-meeting", next_id)

        # Filter to selections only
        selections = [p for p in picks if p["pick_type"] == "selection"]

        assert len(selections) == 4  # 3 selections + 1 roughie

        # Check first selection
        sel1 = selections[0]
        assert sel1["horse_name"] == "FAST HORSE"
        assert sel1["saddlecloth"] == 1
        assert sel1["odds_at_tip"] == 3.50
        assert sel1["place_odds_at_tip"] == 1.45
        assert sel1["bet_type"] == "win"
        assert sel1["bet_stake"] == 10.0

        # Check place bet
        sel2 = selections[1]
        assert sel2["bet_type"] == "place"
        assert sel2["bet_stake"] == 6.0

        # Check each way auto-converts to place (E/W killed in Batch 1)
        sel3 = selections[2]
        assert sel3["bet_type"] == "place"
        assert sel3["bet_stake"] == 4.0

        # Check roughie
        roughie = selections[3]
        assert roughie["horse_name"] == "LONGSHOT"
        assert roughie["tip_rank"] == 4  # Roughie is rank 4
        assert roughie["bet_type"] == "exotics_only"  # Exotics only is a valid bet type

    def test_parse_exotic_exacta(self):
        content = """
*Race 1 – Test Race*

*Degenerate Exotic of the Race*
Exacta: 1, 2 — $20
Est. return: 100% on $20
Why: Clear top two
"""
        counter = [0]

        def next_id():
            counter[0] += 1
            return f"pk-test-{counter[0]:03d}"

        picks = _parse_race_sections(content, "test-content", "test-meeting", next_id)

        exotics = [p for p in picks if p["pick_type"] == "exotic"]

        assert len(exotics) == 1
        assert exotics[0]["exotic_type"].lower() == "exacta"
        # exotic_runners may be stored as JSON string or list
        runners = exotics[0]["exotic_runners"]
        if isinstance(runners, str):
            import json
            runners = json.loads(runners)
        assert runners == [1, 2]
        assert exotics[0]["exotic_stake"] == 20.0

    def test_parse_exotic_trifecta_boxed(self):
        content = """
*Race 2 – Test Race*

*Degenerate Exotic of the Race*
Boxed Trifecta: 1, 3, 5 — $20
"""
        counter = [0]

        def next_id():
            counter[0] += 1
            return f"pk-test-{counter[0]:03d}"

        picks = _parse_race_sections(content, "test-content", "test-meeting", next_id)

        exotics = [p for p in picks if p["pick_type"] == "exotic"]

        assert len(exotics) == 1
        assert "trifecta" in exotics[0]["exotic_type"].lower()
        # exotic_runners may be stored as JSON string or list
        runners = exotics[0]["exotic_runners"]
        if isinstance(runners, str):
            import json
            runners = json.loads(runners)
        assert set(runners) == {1, 3, 5}


class TestParseSequences:
    """Tests for sequence (Quaddie/Big6) parsing."""

    def test_parse_main_quaddie(self):
        content = """
MAIN QUADDIE (R5–R8)
Skinny ($10): 1 / 3 / 5 / 2 (1 combos × $10.00 = $10) — est. return: 1000%
Balanced ($50): 1, 2 / 3, 4 / 5, 6 / 2, 7 (16 combos × $3.13 = $50.08) — est. return: 313%
Wide ($100): 1, 2, 3 / 3, 4, 5 / 5, 6, 7 / 2, 7, 8 (81 combos × $1.23 = $99.63) — est. return: 123%
"""
        counter = [0]

        def next_id():
            counter[0] += 1
            return f"pk-test-{counter[0]:03d}"

        picks = _parse_sequences(content, "test-content", "test-meeting", next_id)

        assert len(picks) == 3  # Skinny, Balanced, Wide

        # Check skinny
        skinny = [p for p in picks if p["sequence_variant"] == "skinny"][0]
        assert skinny["sequence_type"] == "quaddie"
        assert skinny["sequence_start_race"] == 5

        # Check balanced
        balanced = [p for p in picks if p["sequence_variant"] == "balanced"][0]
        assert balanced["sequence_type"] == "quaddie"

    def test_parse_big6(self):
        content = """
BIG 6 (R3–R8)
Skinny ($10): 1 / 2 / 3 / 4 / 5 / 6 (1 combos × $10.00 = $10) — est. return: 1000%
"""
        counter = [0]

        def next_id():
            counter[0] += 1
            return f"pk-test-{counter[0]:03d}"

        picks = _parse_sequences(content, "test-content", "test-meeting", next_id)

        assert len(picks) >= 1
        assert picks[0]["sequence_type"] == "big6"
        assert picks[0]["sequence_start_race"] == 3

    def test_parse_abc_tickets(self):
        """ABC ticket variants should be parsed as ticket_a, ticket_b, ticket_c."""
        content = """
EARLY QUADDIE (R1–R4)
Ticket A ($15): 1, 2 / 3, 4 / 5, 6 / 2, 7 (16 combos x $0.94 = $15) — est. return: 94%
Ticket B ($9): 1, 2, 3 / 3, 4, 5 / 5, 6, 7 / 2, 7, 8 (81 combos x $0.11 = $9) — est. return: 11%
Ticket C ($6): 1, 2, 3, 4 / 3, 4, 5, 6 / 5, 6, 7, 8 / 2, 7, 8, 1 (256 combos x $0.02 = $6) — est. return: 2%
"""
        counter = [0]

        def next_id():
            counter[0] += 1
            return f"pk-test-{counter[0]:03d}"

        picks = _parse_sequences(content, "test-content", "test-meeting", next_id)

        assert len(picks) == 3
        variants = {p["sequence_variant"] for p in picks}
        assert variants == {"ticket_a", "ticket_b", "ticket_c"}
        for p in picks:
            assert p["sequence_type"] == "early_quaddie"
            assert p["sequence_start_race"] == 1


class TestPuntysPick:
    """Tests for Punty's Pick detection."""

    def test_puntys_pick_single_horse(self):
        content = """
*Race 1 – Test Race*
Race type: Maiden, 1200m

*Top 3 + Roughie ($20 pool)*
1. *FAST HORSE* (No.1) — $3.50 / $1.45
   Bet: $10 Win
   Confidence: high
2. *QUICK RUNNER* (No.2) — $5.00 / $1.80
   Bet: $6 Place
   Confidence: med
3. *STEADY EDDIE* (No.3) — $8.00 / $2.50
   Bet: $4 Each Way
   Confidence: low
Roughie: *LONGSHOT* (No.8) — $21.00 / $5.00
Bet: Exotics only

*Punty's Pick:* QUICK RUNNER (No.2) $5.00 Place
"""
        counter = [0]

        def next_id():
            counter[0] += 1
            return f"pk-test-{counter[0]:03d}"

        picks = _parse_race_sections(content, "test-content", "test-meeting", next_id)

        selections = [p for p in picks if p["pick_type"] == "selection"]
        assert len(selections) == 4

        # Only saddlecloth 2 (QUICK RUNNER) should be Punty's Pick
        puntys = [p for p in selections if p.get("is_puntys_pick")]
        assert len(puntys) == 1
        assert puntys[0]["horse_name"] == "QUICK RUNNER"
        assert puntys[0]["saddlecloth"] == 2

        # Others should NOT have is_puntys_pick set
        non_puntys = [p for p in selections if not p.get("is_puntys_pick")]
        assert len(non_puntys) == 3

    def test_puntys_pick_two_horses(self):
        content = """
*Race 3 – Big Race*

1. *HORSE A* (No.4) — $2.80 / $1.30
   Bet: $8 Win
2. *HORSE B* (No.7) — $6.00 / $2.10
   Bet: $7 Place
3. *HORSE C* (No.1) — $9.00 / $3.00
   Bet: $5 Each Way
Roughie: *HORSE D* (No.12) — $31.00 / $6.00
Bet: Exotics only

*Punty's Pick:* HORSE A (No.4) $2.80 Win + HORSE B (No.7) $6.00 Place
"""
        counter = [0]

        def next_id():
            counter[0] += 1
            return f"pk-test-{counter[0]:03d}"

        picks = _parse_race_sections(content, "test-content", "test-meeting", next_id)

        selections = [p for p in picks if p["pick_type"] == "selection"]
        puntys = [p for p in selections if p.get("is_puntys_pick")]
        assert len(puntys) == 2
        puntys_saddles = {p["saddlecloth"] for p in puntys}
        assert puntys_saddles == {4, 7}

    def test_no_puntys_pick_line(self):
        content = """
*Race 2 – Normal Race*

1. *SOME HORSE* (No.5) — $4.00 / $1.60
   Bet: $10 Win
2. *OTHER HORSE* (No.9) — $7.00 / $2.30
   Bet: $6 Place
3. *THIRD HORSE* (No.3) — $12.00 / $3.50
   Bet: $4 Each Way
Roughie: *OUTSIDER* (No.11) — $25.00 / $5.50
Bet: Exotics only
"""
        counter = [0]

        def next_id():
            counter[0] += 1
            return f"pk-test-{counter[0]:03d}"

        picks = _parse_race_sections(content, "test-content", "test-meeting", next_id)

        selections = [p for p in picks if p["pick_type"] == "selection"]
        puntys = [p for p in selections if p.get("is_puntys_pick")]
        assert len(puntys) == 0


    def test_puntys_pick_bet_type_override(self):
        """When Punty's Pick recommends a different bet type than the main selection."""
        content = """
*Race 3 – Korumburra Hotel Mdn Plate*
Race type: Maiden, 2100m

*Top 3 + Roughie ($20 pool)*
1. *ATEYATE* (No.5) — $2.10 / $1.37
   Bet: $7 Win
   Confidence: high
2. *FEDORA* (No.9) — $3.90 / $1.97
   Bet: $5.50 Place
   Confidence: med
3. *DR SLUMP* (No.6) — $5.50 / $2.50
   Bet: $4.50 Place
   Confidence: med
Roughie: *ON THE BRINK* (No.10) — $23.00 / $8.33
Bet: $3 Place

*Punty's Pick:* ATEYATE (No.5) $1.37 Place
I want the stayer in the frame.
"""
        counter = [0]

        def next_id():
            counter[0] += 1
            return f"pk-test-{counter[0]:03d}"

        picks = _parse_race_sections(content, "test-content", "test-meeting", next_id)

        selections = [p for p in picks if p["pick_type"] == "selection"]
        puntys = [p for p in selections if p.get("is_puntys_pick")]
        assert len(puntys) == 1
        assert puntys[0]["horse_name"] == "ATEYATE"
        assert puntys[0]["saddlecloth"] == 5
        # Main bet type is Win
        assert puntys[0]["bet_type"] == "win"
        # Punty's Pick overrides to Place
        assert puntys[0]["pp_bet_type"] == "place"
        assert puntys[0]["pp_odds"] == 1.37

    def test_puntys_pick_same_bet_type_no_override(self):
        """When Punty's Pick uses the same bet type, no override is set."""
        content = """
*Race 1 – Sprint Stakes*

1. *SPEEDY* (No.3) — $4.00 / $1.60
   Bet: $10 Win
2. *RUNNER* (No.7) — $6.00 / $2.00
   Bet: $6 Place
3. *THIRD* (No.1) — $8.00 / $2.50
   Bet: $4 Each Way
Roughie: *OUTSIDER* (No.12) — $21.00 / $5.00
Bet: Exotics only

*Punty's Pick:* SPEEDY (No.3) $4.00 Win
Leads and controls.
"""
        counter = [0]

        def next_id():
            counter[0] += 1
            return f"pk-test-{counter[0]:03d}"

        picks = _parse_race_sections(content, "test-content", "test-meeting", next_id)

        puntys = [p for p in picks if p.get("is_puntys_pick")]
        assert len(puntys) == 1
        assert puntys[0]["bet_type"] == "win"
        # No override when bet types match
        assert puntys[0].get("pp_bet_type") is None
        assert puntys[0].get("pp_odds") is None


class TestDuplicateExoticPrevention:
    """Tests that exotic Punty's Pick doesn't create duplicate exotic picks."""

    def test_pp_exotic_suppresses_standard_exotic(self):
        """When Punty's Pick is an exotic, the standard exotic handler should be skipped."""
        content = """
*Race 5 – Test Sprint*

1. *HORSE A* (No.6) — $3.00 / $1.50
   Bet: $11 Place
2. *HORSE B* (No.3) — $5.00 / $2.00
   Bet: $5 Place
3. *HORSE C* (No.8) — $7.00 / $2.50
   Bet: $4 Place
Roughie: *HORSE D* (No.11) — $15.00 / $4.00
Bet: No Bet (Tracked)

*Punty's Pick:* Quinella [6, 3] — $15 (Value: 1.8x)

*Degenerate Exotic*
Quinella: 6, 3 — $15
"""
        counter = [0]

        def next_id():
            counter[0] += 1
            return f"pk-test-{counter[0]:03d}"

        picks = _parse_race_sections(content, "test-content", "test-meeting", next_id)

        exotics = [p for p in picks if p["pick_type"] == "exotic"]
        # Should be exactly 1 exotic, not 2
        assert len(exotics) == 1
        assert exotics[0].get("is_puntys_pick") is True
        assert exotics[0]["exotic_type"] == "Quinella"

    def test_standard_exotic_still_parsed_without_pp(self):
        """Standard exotics should still be parsed when there's no PP exotic."""
        content = """
*Race 2 – Test Race*

1. *HORSE A* (No.1) — $3.00 / $1.50
   Bet: $10 Win
2. *HORSE B* (No.4) — $5.00 / $2.00
   Bet: $6 Place
3. *HORSE C* (No.7) — $8.00 / $2.50
   Bet: $4 Each Way
Roughie: *HORSE D* (No.9) — $20.00 / $5.00
Bet: No Bet (Tracked)

*Punty's Pick:* HORSE A (No.1) $3.00 Win

*Degenerate Exotic*
Trifecta Box: 1, 4, 7 — $15
"""
        counter = [0]

        def next_id():
            counter[0] += 1
            return f"pk-test-{counter[0]:03d}"

        picks = _parse_race_sections(content, "test-content", "test-meeting", next_id)

        exotics = [p for p in picks if p["pick_type"] == "exotic"]
        assert len(exotics) == 1
        assert exotics[0]["exotic_type"] == "Trifecta Box"
        assert exotics[0].get("is_puntys_pick") is None  # Not a PP exotic


class TestJsonParserPuntysPick:
    """Tests for Punty's Pick in JSON parser path."""

    def test_json_puntys_pick_selection(self):
        """JSON path should set is_puntys_pick on matching selection."""
        content = '''Some early mail text.

```json
{
  "big3": [],
  "races": {
    "1": {
      "selections": [
        {"horse": "FAST ONE", "saddlecloth": 3, "rank": 1, "win_odds": 4.0, "place_odds": 1.60, "bet_type": "Win", "stake": 10, "probability": 35.0, "value": 1.2},
        {"horse": "STEADY", "saddlecloth": 7, "rank": 2, "win_odds": 6.0, "place_odds": 2.00, "bet_type": "Place", "stake": 6, "probability": 25.0}
      ],
      "exotic": {"type": "Exacta", "runners": [3, 7], "stake": 20},
      "puntys_pick": {"saddlecloth": 3, "bet_type": "Win", "odds": 4.0}
    }
  },
  "sequences": []
}
```
'''
        picks = parse_early_mail(content, "test-content", "test-meeting")
        selections = [p for p in picks if p["pick_type"] == "selection"]
        pp = [p for p in selections if p.get("is_puntys_pick")]
        assert len(pp) == 1
        assert pp[0]["horse_name"] == "FAST ONE"
        assert pp[0]["saddlecloth"] == 3

    def test_json_win_probability_normalized(self):
        """JSON parser should normalize win_probability from percentage to decimal."""
        content = '''Some text.

```json
{
  "big3": [],
  "races": {
    "1": {
      "selections": [
        {"horse": "SPEEDY", "saddlecloth": 1, "rank": 1, "win_odds": 3.0, "place_odds": 1.50, "bet_type": "Place", "stake": 10, "probability": 95.0}
      ]
    }
  },
  "sequences": []
}
```
'''
        picks = parse_early_mail(content, "test-content", "test-meeting")
        sel = [p for p in picks if p["pick_type"] == "selection"]
        assert len(sel) == 1
        assert sel[0]["win_probability"] == 0.95  # Normalized from 95.0


class TestParseEarlyMailIntegration:
    """Integration tests for full early mail parsing."""

    def test_parse_full_early_mail(self, sample_early_mail_content):
        picks = parse_early_mail(
            sample_early_mail_content, "test-content-123", "test-meeting-456"
        )

        # Should have multiple picks
        assert len(picks) > 0

        # Check pick types are present
        pick_types = {p["pick_type"] for p in picks}
        assert "big3" in pick_types or "selection" in pick_types

        # All picks should have required fields
        for pick in picks:
            assert pick["id"] is not None
            assert pick["content_id"] == "test-content-123"
            assert pick["meeting_id"] == "test-meeting-456"

    def test_parse_empty_content(self):
        picks = parse_early_mail("", "test", "test")
        assert picks == []

    def test_parse_none_content(self):
        picks = parse_early_mail(None, "test", "test")
        assert picks == []


class TestTrackedBetParsing:
    """Tests for No Bet (Tracked) format parsing."""

    def test_tracked_bet_selection(self):
        """Bet: No Bet (Tracked) should set tracked_only=True and stake=0."""
        content = """
*Race 1 – Test Race*
1. *STRONG HORSE* (No.5) — $1.50 / $1.20
Bet: No Bet (Tracked)
Confidence: HIGH
Probability: 45%

2. *GOOD HORSE* (No.3) — $4.50 / $2.00
Bet: $10 Win
Confidence: MED
"""
        picks = parse_early_mail(content, "test-content", "test-meeting")
        selections = [p for p in picks if p["pick_type"] == "selection"]
        assert len(selections) >= 2

        tracked = [p for p in selections if p.get("tracked_only")]
        staked = [p for p in selections if not p.get("tracked_only")]

        assert len(tracked) == 1
        assert tracked[0]["horse_name"] == "STRONG HORSE"
        assert tracked[0]["bet_stake"] == 0.0

        assert len(staked) >= 1
        assert staked[0]["bet_stake"] == 10.0

    def test_tracked_roughie(self):
        """Roughie with No Bet (Tracked) should be parsed correctly."""
        content = """
*Race 2 – Test Race*
1. *PICK ONE* (No.1) — $3.00 / $1.60
Bet: $8 Win

Roughie: *LONG SHOT* (No.9) — $15.00 / $4.50
Bet: No Bet (Tracked)
"""
        picks = parse_early_mail(content, "test-content", "test-meeting")
        selections = [p for p in picks if p["pick_type"] == "selection"]
        roughie = [p for p in selections if p.get("tip_rank") == 4]
        assert len(roughie) == 1
        assert roughie[0].get("tracked_only") is True
        assert roughie[0]["bet_stake"] == 0.0

    def test_tracked_with_reason(self):
        """Bet: No Bet — reason should capture the reason text."""
        content = """
*Race 1 – Test Race*
1. *SHORT HORSE* (No.5) — $1.50 / $1.20
Bet: No Bet — Too short to back (Win < $2.00)
Confidence: HIGH
Probability: 45%

2. *GOOD HORSE* (No.3) — $4.50 / $2.00
Bet: $10 Win
Confidence: MED
"""
        picks = parse_early_mail(content, "test-content", "test-meeting")
        selections = [p for p in picks if p["pick_type"] == "selection"]
        tracked = [p for p in selections if p.get("tracked_only")]
        assert len(tracked) == 1
        assert tracked[0]["no_bet_reason"] == "Too short to back (Win < $2.00)"

    def test_tracked_without_reason_no_key(self):
        """Bet: No Bet without reason should not have no_bet_reason key."""
        content = """
*Race 1 – Test Race*
1. *STRONG HORSE* (No.5) — $1.50 / $1.20
Bet: No Bet
Confidence: HIGH
"""
        picks = parse_early_mail(content, "test-content", "test-meeting")
        tracked = [p for p in picks if p.get("tracked_only")]
        assert len(tracked) == 1
        assert "no_bet_reason" not in tracked[0]

    def test_tracked_roughie_with_reason(self):
        """Roughie with No Bet — reason should capture reason."""
        content = """
*Race 2 – Test Race*
1. *PICK ONE* (No.1) — $3.00 / $1.60
Bet: $8 Win

Roughie: *LONG SHOT* (No.9) — $15.00 / $4.50
Bet: No Bet — Place prob too low (20% < 35%)
"""
        picks = parse_early_mail(content, "test-content", "test-meeting")
        roughie = [p for p in picks if p.get("tip_rank") == 4]
        assert len(roughie) == 1
        assert roughie[0].get("tracked_only") is True
        assert roughie[0]["no_bet_reason"] == "Place prob too low (20% < 35%)"
