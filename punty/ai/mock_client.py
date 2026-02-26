"""Mock AI client for staging environment.

Returns canned early mail content that parser.py can fully parse,
without making any real API calls. Used when PUNTY_MOCK_EXTERNAL=true.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TokenUsage:
    """Zero-cost token usage for mock responses."""

    model: str = "mock"
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# Canned early mail that parser.py can fully extract: Big 3, selections, exotics, sequences
CANNED_EARLY_MAIL = """*PUNTY EARLY MAIL – {venue} ({date})*
Rightio Legends — welcome to the staging environment mock-up. This is canned content for testing.

*PUNTY'S BIG 3 + MULTI*
1) *MOCK HERO* (Race 1, No.1) — $3.50
   Confidence: high
   Why: Staging mock selection — consistent form, drawn well
2) *STAGING STAR* (Race 2, No.4) — $5.00
   Confidence: med
   Why: Staging mock selection — market mover, class rise
3) *TEST RUNNER* (Race 3, No.2) — $8.00
   Confidence: low
   Why: Staging mock selection — wet track specialist
Multi (all three to win): 10U × ~140.00 = ~1400U collect

*Race 1 – Mock Sprint*
Race type: Maiden, 1200m
Map & tempo: Genuine pace expected with MOCK HERO leading
Punty read: MOCK HERO has the gate speed to cross and control. STAGING FLYER the danger from the draw.

*Punty's Pick: MOCK HERO*

*Top 3 + Roughie ($20 pool)*
1. *MOCK HERO* (No.1) — $3.50 / $1.45
   Bet: $10 Win, return $35.00
   Confidence: high
   Why: Clear leader, best form on the board
2. *STAGING FLYER* (No.3) — $5.00 / $1.80
   Bet: $6 Place, return $10.80
   Confidence: med
   Why: Maps to sit behind the speed
3. *DATA DASHER* (No.5) — $8.00 / $2.50
   Bet: $4 Place, return $10.00
   Confidence: low
   Why: Strong closing sectionals last start
Roughie: *LONGSHOT LARRY* (No.8) — $15.00 / $4.00
Bet: Exotics only
Why: Only if the pace collapses

*Degenerate Exotic of the Race*
Trifecta: 1, 3, 5 — $20
Est. return: 180% on $20
Why: Top three look the clear class

*Race 2 – Mock Mile*
Race type: BM72, 1600m
Map & tempo: Slow tempo, likely sit-sprint affair
Punty read: STAGING STAR gets a soft lead and kicks. PACE MAKER the spoiler.

*Punty's Pick: STAGING STAR*

*Top 3 + Roughie ($20 pool)*
1. *STAGING STAR* (No.4) — $5.00 / $1.80
   Bet: $8 Saver Win, return $40.00
   Confidence: med
   Why: Gets all favours in running from barrier 2
2. *PACE MAKER* (No.2) — $4.00 / $1.50
   Bet: $6 Place, return $9.00
   Confidence: med
   Why: Honest type, never far away
3. *FORM FINDER* (No.6) — $12.00 / $3.50
   Bet: $6 Place, return $21.00
   Confidence: low
   Why: Steps up from weaker grade
Roughie: *BIG UPSET* (No.9) — $18.00 / $5.00
Bet: Exotics only
Why: Blinkers first time, watch for improvement

*Degenerate Exotic of the Race*
Exacta: 4, 2 — $20
Est. return: 120% on $20
Why: Star over Maker looks the right order

*Race 3 – Mock Distance*
Race type: BM64, 2000m
Map & tempo: Genuine tempo, should suit on-pacers
Punty read: TEST RUNNER relishes the wet and gets the right tempo. STAYER SUPREME the class horse.

*Punty's Pick: TEST RUNNER*

*Top 3 + Roughie ($20 pool)*
1. *TEST RUNNER* (No.2) — $8.00 / $2.50
   Bet: $6 Place, return $15.00
   Confidence: med
   Why: Wet track form is superb, maps to get a nice trail
2. *STAYER SUPREME* (No.1) — $2.80 / $1.30
   Bet: $8 Place, return $10.40
   Confidence: high
   Why: Class horse but needs to produce best
3. *DISTANCE KING* (No.7) — $10.00 / $3.00
   Bet: $6 Place, return $18.00
   Confidence: low
   Why: Honest stayer at peak fitness
Roughie: *OUTSIDER OZ* (No.10) — $16.00 / $4.50
Bet: Exotics only
Why: Needs luck from wide gate but has the talent

*Degenerate Exotic of the Race*
Quinella: 1, 2 — $20
Est. return: 150% on $20
Why: Class and conditions runner the clear top two

MAIN QUADDIE (R1–R4)
Skinny ($10): 1 / 4 / 2 / 1 (1 combo × $10.00 = $10) — est. return: 250%
Balanced ($40): 1, 3 / 4, 2 / 1, 2, 7 / 1, 3 (12 combos × $3.33 = $39.96) — est. return: 320%
Wide ($60): 1, 3, 5 / 4, 2, 6 / 1, 2, 7 / 1, 3, 5 (27 combos × $2.22 = $59.94) — est. return: 400%

Tail winds and tailored tips, legends. This is staging — no real money on the line! 🐎
"""

# JSON block appended after format() to avoid brace conflicts
_CANNED_JSON_BLOCK = """
```json
{
  "big3": [
    {"rank": 1, "horse": "MOCK HERO", "race": 1, "saddlecloth": 1, "odds": 3.50},
    {"rank": 2, "horse": "STAGING STAR", "race": 2, "saddlecloth": 4, "odds": 5.00},
    {"rank": 3, "horse": "TEST RUNNER", "race": 3, "saddlecloth": 2, "odds": 8.00}
  ],
  "big3_multi": {"stake": 10, "multi_odds": 140.00, "collect": 1400.00},
  "races": {
    "1": {
      "selections": [
        {"rank": 1, "horse": "MOCK HERO", "saddlecloth": 1, "win_odds": 3.50, "place_odds": 1.45, "bet_type": "Win", "stake": 10.0, "confidence": "HIGH", "probability": 32.0, "value": 1.2},
        {"rank": 2, "horse": "STAGING FLYER", "saddlecloth": 3, "win_odds": 5.00, "place_odds": 1.80, "bet_type": "Place", "stake": 6.0, "confidence": "MED", "probability": 22.0, "value": 1.1},
        {"rank": 3, "horse": "DATA DASHER", "saddlecloth": 5, "win_odds": 8.00, "place_odds": 2.50, "bet_type": "Place", "stake": 4.0, "confidence": "LOW", "probability": 14.0, "value": 1.0}
      ],
      "roughie": {"horse": "LONGSHOT LARRY", "saddlecloth": 8, "win_odds": 15.00, "place_odds": 4.00, "bet_type": "exotics_only"},
      "exotic": {"type": "Trifecta Box", "runners": [1, 3, 5], "stake": 20},
      "puntys_pick": {"saddlecloth": 1, "bet_type": "Win", "odds": 3.50}
    },
    "2": {
      "selections": [
        {"rank": 1, "horse": "STAGING STAR", "saddlecloth": 4, "win_odds": 5.00, "place_odds": 1.80, "bet_type": "Saver Win", "stake": 8.0, "confidence": "MED", "probability": 24.0, "value": 1.15},
        {"rank": 2, "horse": "PACE MAKER", "saddlecloth": 2, "win_odds": 4.00, "place_odds": 1.50, "bet_type": "Place", "stake": 6.0, "confidence": "MED", "probability": 20.0, "value": 1.0},
        {"rank": 3, "horse": "FORM FINDER", "saddlecloth": 6, "win_odds": 12.00, "place_odds": 3.50, "bet_type": "Place", "stake": 6.0, "confidence": "LOW", "probability": 10.0, "value": 1.0}
      ],
      "roughie": {"horse": "BIG UPSET", "saddlecloth": 9, "win_odds": 18.00, "place_odds": 5.00, "bet_type": "exotics_only"},
      "exotic": {"type": "Exacta", "runners": [4, 2], "stake": 20},
      "puntys_pick": {"saddlecloth": 4, "bet_type": "Saver Win", "odds": 5.00}
    },
    "3": {
      "selections": [
        {"rank": 1, "horse": "TEST RUNNER", "saddlecloth": 2, "win_odds": 8.00, "place_odds": 2.50, "bet_type": "Place", "stake": 6.0, "confidence": "MED", "probability": 16.0, "value": 1.1},
        {"rank": 2, "horse": "STAYER SUPREME", "saddlecloth": 1, "win_odds": 2.80, "place_odds": 1.30, "bet_type": "Place", "stake": 8.0, "confidence": "HIGH", "probability": 38.0, "value": 1.0},
        {"rank": 3, "horse": "DISTANCE KING", "saddlecloth": 7, "win_odds": 10.00, "place_odds": 3.00, "bet_type": "Place", "stake": 6.0, "confidence": "LOW", "probability": 12.0, "value": 1.0}
      ],
      "roughie": {"horse": "OUTSIDER OZ", "saddlecloth": 10, "win_odds": 16.00, "place_odds": 4.50, "bet_type": "exotics_only"},
      "exotic": {"type": "Quinella", "runners": [1, 2], "stake": 20},
      "puntys_pick": {"saddlecloth": 2, "bet_type": "Place", "odds": 8.00}
    }
  },
  "sequences": [
    {"type": "Quaddie", "start_race": 1, "end_race": 4, "variants": [
      {"name": "Skinny", "legs": "1/4/2/1", "combos": 1, "unit_price": 10.00, "total": 10.00},
      {"name": "Balanced", "legs": "1,3/4,2/1,2,7/1,3", "combos": 12, "unit_price": 3.33, "total": 39.96},
      {"name": "Wide", "legs": "1,3,5/4,2,6/1,2,7/1,3,5", "combos": 27, "unit_price": 2.22, "total": 59.94}
    ]}
  ]
}
```
"""


class MockAIClient:
    """Mock AI client that returns canned content without API calls.

    Matches the interface of AIClient so it can be swapped in transparently.
    """

    def __init__(self, **kwargs):
        self.last_usage: Optional[TokenUsage] = TokenUsage()
        self._api_key = None
        self._client = None

    async def close(self) -> None:
        """No-op — nothing to close."""
        pass

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.8,
        max_tokens: int = 32000,
    ) -> str:
        """Return canned content."""
        logger.info("[MOCK] generate() called — returning canned content")
        return CANNED_EARLY_MAIL.format(venue="Mock Venue", date="2026-01-01") + _CANNED_JSON_BLOCK

    async def generate_with_context(
        self,
        system_prompt: str,
        context: str,
        instruction: str,
        temperature: float = 0.8,
        max_tokens: int = 32000,
    ) -> str:
        """Return canned early mail, extracting venue/date from the instruction if possible."""
        import re

        # Try to extract venue and date from the instruction string
        venue = "Mock Venue"
        dt = "2026-01-01"
        venue_match = re.search(r"for (.+?) on (\d{4}-\d{2}-\d{2})", instruction)
        if venue_match:
            venue = venue_match.group(1)
            dt = venue_match.group(2)

        logger.info(f"[MOCK] generate_with_context() for {venue} — returning canned content")
        self.last_usage = TokenUsage()
        return CANNED_EARLY_MAIL.format(venue=venue, date=dt) + _CANNED_JSON_BLOCK
