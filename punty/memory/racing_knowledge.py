"""Australian horse racing domain knowledge for RAG context injection.

Compiled from:
- Edge Thoroughbreds (track conditions guide)
- Racing Victoria (handicapping rules)
- Racing Australia (rules of racing)
- horseracing.com.au (track specifications)
- TAB (apprentice claims)

Used by the AI prompt builder and future blog tool.
"""

# =============================================================================
# TRACK CONDITIONS — Penetrometer Ranges & Meanings
# =============================================================================

TRACK_CONDITIONS = {
    "Firm 1": {
        "penetrometer_range": (0.0, 2.0),
        "description": "Extremely hard, road-like surface. Rarely seen — risk of injury.",
        "impact": "Favours pure speed horses. Hard on joints. Fast times.",
        "rating": 1,
    },
    "Firm 2": {
        "penetrometer_range": (2.1, 2.5),
        "description": "Very firm with minimal give.",
        "impact": "Favours speed horses. Slight cushion but still hard.",
        "rating": 2,
    },
    "Good 3": {
        "penetrometer_range": (2.6, 3.0),
        "description": "Optimal racing surface for most horses.",
        "impact": "Ideal conditions. True form guide. Even contest.",
        "rating": 3,
    },
    "Good 4": {
        "penetrometer_range": (3.1, 3.5),
        "description": "Slightly softer than ideal, more cushioning.",
        "impact": "Good racing. Slight advantage for horses that handle a touch of give.",
        "rating": 4,
    },
    "Soft 5": {
        "penetrometer_range": (3.6, 4.0),
        "description": "Noticeable softness, slowing race times.",
        "impact": "Form on soft matters. Leaders can struggle if pace is genuine. "
                  "Horses with soft-track form get an edge.",
        "rating": 5,
    },
    "Soft 6": {
        "penetrometer_range": (4.1, 4.5),
        "description": "Soft ground requiring more effort from runners.",
        "impact": "Genuine wet trackers favoured. Inside rail often deteriorates. "
                  "Stamina becomes more important.",
        "rating": 6,
    },
    "Soft 7": {
        "penetrometer_range": (4.6, 5.0),
        "description": "Very soft, approaching heavy conditions.",
        "impact": "Major impact on form. Speed horses struggle. Need proven soft form. "
                  "Races run slower. Wide runners sometimes advantaged as rail chews out.",
        "rating": 7,
    },
    "Heavy 8": {
        "penetrometer_range": (5.1, 5.5),
        "description": "Heavy going, waterlogged areas appearing.",
        "impact": "True form is unreliable — wet-track form essential. Leaders often "
                  "struggle. Staying power and mud form critical.",
        "rating": 8,
    },
    "Heavy 9": {
        "penetrometer_range": (5.6, 6.0),
        "description": "Very heavy, deep footing throughout.",
        "impact": "Only genuine mudlarks thrive. Race times significantly slower. "
                  "Small fields can favour on-pace runners if rail holds.",
        "rating": 9,
    },
    "Heavy 10": {
        "penetrometer_range": (6.1, 99.0),
        "description": "Worst conditions — waterlogged, dangerous footing.",
        "impact": "Survival test. Only horses with Heavy 10 form should be considered. "
                  "Meetings may be abandoned if conditions worsen.",
        "rating": 10,
    },
}

# Going stick: advanced tool measuring penetration + shear strength (turf stability)
# Provides more detailed surface assessment than penetrometer alone.

# =============================================================================
# TRACK SPECIFICATIONS — Australian Racecourses
# =============================================================================

TRACKS = {
    # ----- VICTORIA (Metropolitan) -----
    "Flemington": {
        "state": "VIC", "tier": "metro",
        "circumference": 2312, "straight": 450, "direction": "anti-clockwise",
        "surface": "turf",
        "notes": "Unique 1200m straight course. Long straight favours closers in "
                 "staying races. Wide track. Downhill into straight.",
    },
    "Caulfield": {
        "state": "VIC", "tier": "metro",
        "circumference": 2080, "straight": 367, "direction": "anti-clockwise",
        "surface": "turf",
        "notes": "Heath course (shorter circumference 1,446m also used). Tight turns "
                 "can disadvantage wide runners. Inside barrier advantage.",
    },
    "Moonee Valley": {
        "state": "VIC", "tier": "metro",
        "circumference": 1805, "straight": 173, "direction": "anti-clockwise",
        "surface": "turf",
        "notes": "Shortest straight in Australia. Huge on-pace bias — leaders/on-pace "
                 "runners dominate. Barrier draw critical. Must be near the lead.",
    },
    "Sandown": {
        "state": "VIC", "tier": "metro",
        "circumference": 2097, "straight": 491, "direction": "anti-clockwise",
        "surface": "turf",
        "notes": "Longest straight of Victorian metro tracks. Suits closers in "
                 "staying races. Hillside course also used (1502m circ, 403m straight).",
    },

    # ----- VICTORIA (Provincial/Country) -----
    "Pakenham": {
        "state": "VIC", "tier": "provincial",
        "circumference": 1400, "straight": 300, "direction": "anti-clockwise",
        "surface": "synthetic",
        "notes": "Racing Club synthetic surface. Smaller track — barrier and on-pace "
                 "running helps. Also has turf track.",
    },
    "Geelong": {
        "state": "VIC", "tier": "provincial",
        "circumference": 2043, "straight": 400, "direction": "anti-clockwise",
        "surface": "synthetic",
        "notes": "Near-perfect oval shape. Converted from grass to synthetic for "
                 "all-weather racing.",
    },
    "Ballarat": {
        "state": "VIC", "tier": "provincial",
        "circumference": 1900, "straight": 450, "direction": "anti-clockwise",
        "surface": "turf",
        "notes": "Long straight for a provincial track. Hosts most grass meetings "
                 "in regional Victoria (32+ per year). Also has synthetic.",
    },
    "Cranbourne": {
        "state": "VIC", "tier": "provincial",
        "circumference": 1700, "straight": 300, "direction": "anti-clockwise",
        "surface": "turf",
        "notes": "Smallish track. On-pace bias likely due to short straight.",
    },
    "Mornington": {
        "state": "VIC", "tier": "provincial",
        "circumference": 1800, "straight": None, "direction": "anti-clockwise",
        "surface": "turf",
        "notes": "Won Victoria's best country circuit 3 times. Attracts metro trainers.",
    },
    "Sale": {
        "state": "VIC", "tier": "country",
        "circumference": 2040, "straight": 400, "direction": "anti-clockwise",
        "surface": "turf",
        "notes": "Good-sized track. Wide racing surface. Distances 1000m-2200m. "
                 "Max 16 runners (14 for 1400m start). Also hosts jumps.",
    },
    "Wangaratta": {
        "state": "VIC", "tier": "country",
        "circumference": None, "straight": None, "direction": None,
        "surface": "turf",
        "notes": "20m wide track. Regular starts up to 2000m.",
    },
    "Swan Hill": {
        "state": "VIC", "tier": "country",
        "circumference": 1800, "straight": 250, "direction": None,
        "surface": "turf",
        "notes": "Distances 800m-2100m. Close, clean racing with long runs to the turn.",
    },
    "Seymour": {
        "state": "VIC", "tier": "country",
        "circumference": None, "straight": None, "direction": None,
        "surface": "turf",
        "notes": "Country track. Hosts regular meetings.",
    },
    "Horsham": {
        "state": "VIC", "tier": "country",
        "circumference": None, "straight": None, "direction": None,
        "surface": "turf",
        "notes": "Wimmera region country track.",
    },
    "Bairnsdale": {
        "state": "VIC", "tier": "country",
        "circumference": None, "straight": None, "direction": None,
        "surface": "turf",
        "notes": "East Gippsland country track.",
    },
    "Bendigo": {
        "state": "VIC", "tier": "provincial",
        "circumference": 2000, "straight": 370, "direction": "anti-clockwise",
        "surface": "turf",
        "notes": "Good-sized provincial track. Fair track — no strong bias.",
    },
    "Kilmore": {
        "state": "VIC", "tier": "country",
        "circumference": 1800, "straight": 300, "direction": "anti-clockwise",
        "surface": "turf",
        "notes": "Country track north of Melbourne.",
    },
    "Echuca": {
        "state": "VIC", "tier": "country",
        "circumference": None, "straight": None, "direction": None,
        "surface": "turf",
        "notes": "Murray River region country track.",
    },
    "Stony Creek": {
        "state": "VIC", "tier": "country",
        "circumference": None, "straight": None, "direction": None,
        "surface": "turf",
        "notes": "South Gippsland country track. Also hosts jumps racing.",
    },

    # ----- NEW SOUTH WALES (Metropolitan) -----
    "Randwick": {
        "state": "NSW", "tier": "metro",
        "circumference": 2224, "straight": 410, "direction": "anti-clockwise",
        "surface": "turf",
        "notes": "Kensington track also used (1,489m circ, 306m straight). "
                 "Big galloping track. Suits horses that can sustain a run.",
    },
    "Rosehill": {
        "state": "NSW", "tier": "metro",
        "circumference": 2048, "straight": 408, "direction": "anti-clockwise",
        "surface": "turf",
        "notes": "Tight-turning track. Can get heavy in winter. Inside draw helps.",
    },
    "Canterbury": {
        "state": "NSW", "tier": "metro",
        "circumference": 1567, "straight": 308, "direction": "anti-clockwise",
        "surface": "turf",
        "notes": "Small, tight track. On-pace bias. Barrier draw very important. "
                 "Short straight means leaders hard to run down.",
    },
    "Warwick Farm": {
        "state": "NSW", "tier": "metro",
        "circumference": 1937, "straight": 326, "direction": "anti-clockwise",
        "surface": "turf",
        "notes": "Georges River track. Can get heavy quickly in rain.",
    },
    "Newcastle": {
        "state": "NSW", "tier": "provincial",
        "circumference": 2036, "straight": 355, "direction": "anti-clockwise",
        "surface": "turf",
        "notes": "Broadmeadow racecourse. Good provincial track.",
    },
    "Gosford": {
        "state": "NSW", "tier": "provincial",
        "circumference": 1710, "straight": 250, "direction": "anti-clockwise",
        "surface": "turf",
        "notes": "Short straight. On-pace advantage. Small track.",
    },
    "Hawkesbury": {
        "state": "NSW", "tier": "provincial",
        "circumference": 1800, "straight": 300, "direction": "anti-clockwise",
        "surface": "turf",
        "notes": "Clarendon track. Provincial circuit.",
    },
    "Scone": {
        "state": "NSW", "tier": "country",
        "circumference": None, "straight": None, "direction": None,
        "surface": "turf",
        "notes": "Upper Hunter region. Horse breeding capital of Australia.",
    },
    "Wagga": {
        "state": "NSW", "tier": "country",
        "circumference": 2200, "straight": 420, "direction": "anti-clockwise",
        "surface": "turf",
        "notes": "Large country track with long straight. Fair racing surface.",
    },
    "Nowra": {
        "state": "NSW", "tier": "country",
        "circumference": None, "straight": None, "direction": None,
        "surface": "turf",
        "notes": "South Coast NSW country track.",
    },
    "Tamworth": {
        "state": "NSW", "tier": "country",
        "circumference": None, "straight": None, "direction": None,
        "surface": "turf",
        "notes": "New England region country track.",
    },
    "Port Macquarie": {
        "state": "NSW", "tier": "country",
        "circumference": None, "straight": None, "direction": None,
        "surface": "turf",
        "notes": "Mid-North Coast country track.",
    },
    "Muswellbrook": {
        "state": "NSW", "tier": "country",
        "circumference": None, "straight": None, "direction": None,
        "surface": "turf",
        "notes": "Hunter Valley country track.",
    },

    # ----- QUEENSLAND (Metropolitan) -----
    "Doomben": {
        "state": "QLD", "tier": "metro",
        "circumference": 1715, "straight": 341, "direction": "anti-clockwise",
        "surface": "turf",
        "notes": "Smaller metro track. On-pace bias common. Tight turns.",
    },
    "Eagle Farm": {
        "state": "QLD", "tier": "metro",
        "circumference": 2027, "straight": 434, "direction": "anti-clockwise",
        "surface": "turf",
        "notes": "Long straight suits closers. Major carnival track. Has had "
                 "surface issues historically.",
    },

    # ----- QUEENSLAND (Provincial/Country) -----
    "Gold Coast": {
        "state": "QLD", "tier": "provincial",
        "circumference": 2060, "straight": 410, "direction": "anti-clockwise",
        "surface": "turf",
        "notes": "Good-sized track. Fair racing surface.",
    },
    "Sunshine Coast": {
        "state": "QLD", "tier": "provincial",
        "circumference": 1976, "straight": 400, "direction": None,
        "surface": "turf",
        "notes": "Also has cushion track (1760m circ). Cushion track built 2008 "
                 "for all-weather racing.",
    },
    "Ipswich": {
        "state": "QLD", "tier": "provincial",
        "circumference": None, "straight": None, "direction": None,
        "surface": "turf",
        "notes": "SE Queensland provincial track.",
    },
    "Toowoomba": {
        "state": "QLD", "tier": "country",
        "circumference": None, "straight": None, "direction": None,
        "surface": "turf",
        "notes": "Darling Downs region. Clifford Park racecourse.",
    },
    "Rockhampton": {
        "state": "QLD", "tier": "country",
        "circumference": None, "straight": None, "direction": None,
        "surface": "turf",
        "notes": "Central Queensland country track. Callaghan Park.",
    },
    "Townsville": {
        "state": "QLD", "tier": "country",
        "circumference": None, "straight": None, "direction": None,
        "surface": "turf",
        "notes": "North Queensland country track.",
    },

    # ----- SOUTH AUSTRALIA -----
    "Morphettville": {
        "state": "SA", "tier": "metro",
        "circumference": 2339, "straight": 334, "direction": "anti-clockwise",
        "surface": "turf",
        "notes": "Main SA metro track. Also has Parks course (2100m circ). "
                 "Large circumference but shorter straight than expected.",
    },
    "Murray Bridge": {
        "state": "SA", "tier": "provincial",
        "circumference": None, "straight": None, "direction": None,
        "surface": "turf",
        "notes": "SA provincial track.",
    },
    "Strathalbyn": {
        "state": "SA", "tier": "country",
        "circumference": 1690, "straight": 340, "direction": None,
        "surface": "turf",
        "notes": "SA country track.",
    },

    # ----- WESTERN AUSTRALIA -----
    "Ascot": {
        "state": "WA", "tier": "metro",
        "circumference": 2022, "straight": 294, "direction": "anti-clockwise",
        "surface": "turf",
        "notes": "Summer metro track. Short straight — on-pace advantage. "
                 "Inside barrier critical.",
    },
    "Belmont": {
        "state": "WA", "tier": "metro",
        "circumference": 1699, "straight": 333, "direction": "anti-clockwise",
        "surface": "turf",
        "notes": "Winter metro track. Small track. On-pace bias common.",
    },
    "Bunbury": {
        "state": "WA", "tier": "provincial",
        "circumference": None, "straight": None, "direction": None,
        "surface": "turf",
        "notes": "SW WA provincial track.",
    },
    "Pinjarra": {
        "state": "WA", "tier": "provincial",
        "circumference": 1837, "straight": 350, "direction": None,
        "surface": "turf",
        "notes": "Also has 1000m straight. Provincial track south of Perth.",
    },
    "Geraldton": {
        "state": "WA", "tier": "country",
        "circumference": None, "straight": None, "direction": None,
        "surface": "turf",
        "notes": "Mid-West WA country track.",
    },

    # ----- TASMANIA -----
    "Hobart": {
        "state": "TAS", "tier": "metro",
        "circumference": 1990, "straight": 350, "direction": "anti-clockwise",
        "surface": "turf",
        "notes": "Elwick racecourse. Main Tasmanian track.",
    },
    "Launceston": {
        "state": "TAS", "tier": "metro",
        "circumference": 1830, "straight": 230, "direction": "anti-clockwise",
        "surface": "turf",
        "notes": "Mowbray racecourse. Short straight — on-pace bias.",
    },
    "Devonport": {
        "state": "TAS", "tier": "country",
        "circumference": None, "straight": None, "direction": None,
        "surface": "turf",
        "notes": "NW Tasmania country track.",
    },

    # ----- ACT / NT -----
    "Canberra": {
        "state": "ACT", "tier": "provincial",
        "circumference": None, "straight": None, "direction": None,
        "surface": "turf",
        "notes": "Thoroughbred Park.",
    },
    "Darwin": {
        "state": "NT", "tier": "country",
        "circumference": None, "straight": None, "direction": None,
        "surface": "turf",
        "notes": "Fannie Bay racecourse. Dry season racing (April-October).",
    },
    "Alice Springs": {
        "state": "NT", "tier": "country",
        "circumference": None, "straight": None, "direction": None,
        "surface": "turf",
        "notes": "Pioneer Park. Hosts Alice Springs Cup carnival.",
    },
}

# =============================================================================
# HANDICAPPING RULES
# =============================================================================

HANDICAPPING = {
    "rating_system": {
        "name": "Ratings Based Handicapping (RBH)",
        "description": "Numeric ratings reflecting overall performance. Emphasises "
                       "recent form. All ratings equivalent to a fully mature 4yo male. "
                       "Ratings assessed within 48-72 hours of racing.",
    },
    "minimum_weights": {
        "standard_handicap": {"min": 54, "top": 60},
        "group_races": {"min": "52-53", "top": 59},
        "caulfield_melbourne_cup": {"min": 51, "top": 59},
        "two_year_old": {"min": None, "top": 59},
    },
    "sex_allowance": {
        "description": "Fillies and mares receive 2kg allowance in mixed-sex races "
                       "across the entire season.",
        "amount_kg": 2.0,
    },
    "age_allowance_3yo_country": {
        "description": "3yo vs older horses in country races: sliding allowance "
                       "from 2.5kg (August) declining to nil (May-July).",
    },
    "age_allowance_3yo_metro": {
        "description": "Metropolitan: distance-dependent, ranging from 0-4.5kg "
                       "depending on distance and month.",
    },
    "win_penalties": {
        "metropolitan_saturday": "6-8 rating points",
        "metropolitan_midweek": "4-6 rating points",
        "country_standard": "3-5 rating points",
        "rated_100_plus": "1.5-2kg increase",
        "dominant_win": "Larger increase at handicapper discretion",
        "place_getter": "1-2 rating points if performance merits",
    },
    "unplaced_performance": {
        "standard": "Rating unchanged",
        "uncompetitive": "0.5-1.0kg reduction",
        "unsuitable_distance": "No reduction",
        "returning_from_injury": "Greater reductions possible (>1kg)",
    },
    "benchmark_races": {
        "description": "All raced horses qualify. For each rating point above the "
                       "benchmark threshold, horses carry additional 0.5kg.",
        "example": "BM64: a horse rated 70 carries 3kg extra (6 points × 0.5kg).",
    },
    "black_type_minimums": {
        "Listed": {"range": "68-84 (age dependent)"},
        "Group 3": {"range": "74-90"},
        "Group 2": {"range": "78-95"},
        "Group 1": {"range": "96-105"},
    },
}

# =============================================================================
# APPRENTICE CLAIMS
# =============================================================================

APPRENTICE_CLAIMS = {
    "description": "Apprentice jockeys can claim weight allowances based on their "
                   "career wins. Claims typically range from 1.5kg to 4kg. The claim "
                   "is deducted from the allocated weight.",
    "typical_tiers": {
        "4kg": "0-20 metropolitan wins",
        "3kg": "21-40 metropolitan wins",
        "2kg": "41-60 metropolitan wins",
        "1.5kg": "61-80 metropolitan wins",
    },
    "restrictions": {
        "no_claim": "Races with prize money exceeding $50,000 (varies by state). "
                    "Group and Listed races generally do not allow claims.",
        "must_claim_to_riding_weight": "Apprentices must claim down to (not below) "
                                       "their notified riding weight.",
    },
    "impact": "A 3kg claim on a 58kg horse means 55kg carried. Significant advantage "
              "especially in handicap races. Good apprentice + claim = value play.",
}

# =============================================================================
# KEY RACING RULES (affecting outcomes)
# =============================================================================

RACING_RULES = {
    "barrier_draw": {
        "procedure": "Conducted 2-4 days before race. Random allocation.",
        "impact": "Inside barriers advantage on tight tracks (Moonee Valley, Canterbury, "
                  "Ascot). Less impact on big galloping tracks (Flemington, Randwick). "
                  "If a scratching occurs, horses from outside move in one gate.",
    },
    "scratchings": {
        "procedure": "Lodged via Stable Assist. Released live to public.",
        "barrier_impact": "When a horse scratches from a non-outside gate, all horses "
                          "outside move in one gate to fill the gap.",
    },
    "dead_heat": {
        "rule": "Prizes divided equally. Each horse treated as a full winner for "
                "penalty and future eligibility purposes.",
        "betting": "Dividends halved for dead heat winners.",
    },
    "weight_rules": {
        "overweight": "Fractions under 0.5kg disregarded. Over 0.5kg = disqualification.",
        "gear_changes": "No changes permitted after weighing out without steward approval.",
    },
    "whip_rules": {
        "rule": "Padded whip mandatory. Strike limit per race. Excessive use = penalty. "
                "Jockeys penalised for whip offences may cop suspension.",
    },
    "race_frequency": {
        "rule": "Max 5 starts in 30 days without permission. No consecutive-day starts "
                "without approval.",
    },
    "race_classifications": {
        "Group 1": "Highest level. Major carnivals. Weight-for-age or set weights.",
        "Group 2": "Second tier black type. Quality races.",
        "Group 3": "Third tier black type.",
        "Listed": "Fourth tier black type. Quality races below Group level.",
        "BM (Benchmark)": "Handicap based on rating. BM58, BM64, BM70, BM78, etc. "
                          "Horse carries 0.5kg per rating point above the benchmark.",
        "Class 1-6": "Based on number of wins. Class 1 = won once only.",
        "Maiden": "Never won a flat race. First win leads to Class 1.",
        "Restricted": "CG&E (colts, geldings & entire), F&M (fillies & mares), "
                      "age-restricted (2yo, 3yo, 3yo+).",
    },
}

# =============================================================================
# TRACK BIAS PATTERNS — General knowledge
# =============================================================================

TRACK_BIAS_KNOWLEDGE = {
    "short_straight_bias": {
        "description": "Tracks with straights under 300m strongly favour on-pace runners. "
                       "Leaders are hard to run down.",
        "tracks": ["Moonee Valley (173m)", "Launceston (230m)", "Gosford (250m)",
                   "Ascot (294m)", "Cranbourne (300m)", "Pakenham (300m)"],
    },
    "long_straight_advantage": {
        "description": "Tracks with straights over 400m give closers a chance. "
                       "Patient rides rewarded.",
        "tracks": ["Sandown (491m)", "Flemington (450m)", "Ballarat (450m)",
                   "Eagle Farm (434m)", "Wagga (420m)", "Randwick (410m)"],
    },
    "wet_weather_patterns": {
        "description": "As rain falls, inside rail deteriorates first. Early in a wet "
                       "meeting, inside is fine. By mid-meeting, riders start going "
                       "wide. Track condition upgrades/downgrades happen during racing.",
        "key_insight": "If rail is out (e.g. 'rail out 6m'), the true running rail "
                       "is 6m from the inside fence. This creates a wider track and "
                       "can negate some inside draw advantage.",
    },
    "synthetic_tracks": {
        "description": "Synthetic surfaces (Pakenham, Geelong) are all-weather and "
                       "generally run consistently. Less affected by rain. Can favour "
                       "on-pace runners due to even surface.",
        "key_insight": "Form on synthetic doesn't always translate to turf and vice versa.",
    },
    "rail_position_impact": {
        "description": "Rail position is set before racing and defines where the running "
                       "rail sits. 'True' = right on the inside fence. 'Out 3m' = rail "
                       "is 3m from fence, giving fresh ground.",
        "key_insight": "When rail is out significantly (6m+), inside barriers lose their "
                       "advantage as there's more room. Mid-to-outside draws become "
                       "competitive.",
    },
    "small_field_patterns": {
        "description": "Fields of 8 or fewer tend to favour on-pace runners. Less "
                       "traffic, fewer excuses for backmarkers.",
        "key_insight": "Small fields + short straight = strong leader bias.",
    },
    "distance_bias": {
        "sprints_1000_1200m": "Speed is king. Barrier draw and early position critical. "
                              "Leaders dominate, especially on small tracks.",
        "middle_1400_1600m": "Balance between speed and stamina. Pace maps become "
                             "important — genuine speed in the race helps closers.",
        "staying_1800_2400m": "Stamina and race fitness matter most. Closers get their "
                              "chance if the pace is genuine. Jockey tactics more important.",
        "extreme_2400m_plus": "Pure stamina test. Front-runners often collapse. "
                               "Patient, well-timed rides win.",
    },
}


def get_track_info(venue: str) -> dict | None:
    """Look up track information by venue name (case-insensitive)."""
    venue_lower = venue.lower().strip()
    for name, info in TRACKS.items():
        if name.lower() == venue_lower:
            return {"name": name, **info}
    # Fuzzy match — check if venue contains track name or vice versa
    for name, info in TRACKS.items():
        if name.lower() in venue_lower or venue_lower in name.lower():
            return {"name": name, **info}
    return None


def get_condition_info(condition: str) -> dict | None:
    """Look up track condition details by name (e.g. 'Good 4')."""
    return TRACK_CONDITIONS.get(condition)


def get_condition_from_penetrometer(reading: float) -> str | None:
    """Infer track condition from penetrometer reading."""
    for name, info in TRACK_CONDITIONS.items():
        low, high = info["penetrometer_range"]
        if low <= reading <= high:
            return name
    return None


def get_straight_bias(venue: str) -> str:
    """Return bias assessment based on straight length."""
    info = get_track_info(venue)
    if not info or not info.get("straight"):
        return "unknown"
    straight = info["straight"]
    if straight <= 250:
        return "strong_leader_bias"
    elif straight <= 350:
        return "moderate_leader_bias"
    elif straight <= 420:
        return "fair"
    else:
        return "suits_closers"
