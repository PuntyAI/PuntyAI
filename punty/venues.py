"""Centralised venue registry — single source of truth for Australian racecourses.

Consolidates venue aliases, normalization, state mapping from across the codebase.
"""
import logging

logger = logging.getLogger(__name__)

# Sponsor prefixes stripped from venue names (shared across all scrapers)
SPONSOR_PREFIXES = [
    "sportsbet", "ladbrokes", "bet365", "tab", "neds", "pointsbet",
    "unibet", "betfair", "palmerbet", "bluebet", "topsport", "aquis",
    "picklebet park", "beaumont",
]

# Venue aliases: map alternative names to canonical form
# canonical form = lowercase, no sponsor, the name used in _STATE_TRACKS
VENUE_ALIASES = {
    # Racing.com URL slug aliases
    "sandown lakeside": "sandown",
    "sandown-lakeside": "sandown",
    "thomas farms rc murray bridge": "murray bridge",
    "thomas-farms-rc-murray-bridge": "murray bridge",
    "park kilmore": "kilmore",
    "park-kilmore": "kilmore",
    # Common display name aliases
    "royal randwick": "randwick",
    "the valley": "moonee valley",
    "rosehill gardens": "rosehill",
    "canterbury park": "canterbury",
    "pinjarra park": "pinjarra",
    "cheltenham park": "morphettville",
    "cannon park": "cairns",
    "ladbrokes cannon park": "cairns",
    # PF-specific aliases
    "beaumont newcastle": "newcastle",
    # Southside = synthetic track at same venue
    "southside pakenham": "pakenham",
    "southside cranbourne": "cranbourne",
    # Yarra Glen = Yarra Valley Racing Club (same track, different names across sources)
    "yarra valley": "yarra glen",
    "yarra-valley": "yarra glen",
}

# Complete Australian racetracks by state (from racingaustralia.horse)
_STATE_TRACKS = {
    "NSW": [
        "adaminaby", "albury", "ardlethan", "armidale", "ballina", "bathurst",
        "bingara", "binnaway", "boorowa", "bourke", "bowraville",
        "braidwood", "brewarrina", "broken hill", "canberra", "canterbury",
        "carinda", "carrathool", "casino", "cessnock", "cobar", "coffs harbour",
        "collarenebri", "come-by-chance", "condobolin", "coolabah", "cooma", "coonabarabran",
        "coonamble", "cootamundra", "corowa", "cowra", "crookwell", "deepwater", "deniliquin",
        "enngonia", "fernhill", "forbes", "geurie", "gilgandra", "glen innes", "gosford",
        "goulburn", "grafton", "grenfell", "griffith", "gulargambone", "gulgong", "gundagai",
        "gunnedah", "harden", "hawkesbury", "hay", "hillston", "holbrook", "jerilderie",
        "kembla grange", "kempsey", "kensington", "lakelands", "leeton", "lightning ridge",
        "lismore", "lockhart", "louth", "mallawa", "mendooran", "merriwa", "moama", "moree",
        "moruya", "moulamein", "mudgee", "mungery", "mungindi", "murwillumbah", "muswellbrook",
        "narrabri", "narrandera", "narromine", "newcastle", "nowra", "nyngan", "orange",
        "parkes", "pooncarie", "quambone", "queanbeyan", "quirindi", "randwick", "rosehill",
        "sapphire coast", "scone", "tabulam", "talmoi",
        "tamworth", "taree", "tocumwal", "tomingley", "tottenham", "trangie", "trundle",
        "tullamore", "tullibigeal", "tumbarumba", "tumut", "tuncurry", "wagga", "wagga riverside",
        "walcha", "walgett", "wallabadah", "wamboyne", "warialda", "warren", "warwick farm",
        "wauchope", "wean", "wellington", "wentworth", "wyong", "yass", "young",
    ],
    "VIC": [
        "alexandra", "ararat", "avoca", "bairnsdale", "ballan", "ballarat", "balnarring",
        "benalla", "bendigo", "burrumbeet", "caulfield", "colac", "coleraine", "cranbourne",
        "donald", "drouin", "dunkeld", "echuca", "edenhope", "flemington", "geelong",
        "great western", "gunbower", "hanging rock", "healesville", "hinnomunjie", "horsham",
        "kerang", "kilmore", "kyneton", "manangatang", "mansfield", "merton", "mildura",
        "moe", "moonee valley", "mornington", "mortlake", "murtoa", "nhill",
        "oak park", "pakenham", "penshurst", "sale", "sandown", "seymour", "st arnaud",
        "stawell", "stony creek", "swan hill", "swifts creek", "tatura", "towong", "traralgon",
        "warracknabeal", "warrnambool", "werribee", "werribee park", "wodonga", "wycheproof",
        "yarra glen", "yarra valley", "yea",
    ],
    "QLD": [
        "almaden", "alpha", "aramac", "augathella", "beaudesert", "bedourie", "bell",
        "betoota", "birdsville", "blackall", "bluff", "boulia", "bowen", "bundaberg",
        "burketown", "burrandowan", "cairns", "calliope", "camooweal", "capella",
        "charleville", "charters towers", "chillagoe", "chinchilla", "clifton", "cloncurry",
        "coen", "cooktown", "corfield", "cunnamulla", "dalby", "deagon", "dingo", "doomben",
        "duaringa", "eagle farm", "eidsvold", "einasleigh", "emerald", "eromanga", "esk",
        "ewan", "flinton", "gatton", "gayndah", "georgetown", "gladstone", "gold coast",
        "goondiwindi", "gordonvale", "gregory downs", "gympie", "hebel", "home hill",
        "hughenden", "ilfracombe", "ingham", "injune", "innisfail", "ipswich", "isisford",
        "jandowae", "jericho", "julia creek", "jundah", "kilcoy", "kumbia", "laura",
        "longreach", "mackay", "mareeba", "maxwelton", "mckinlay", "middlemount", "miles",
        "mingela", "mitchell", "monto", "moranbah", "morven", "mount garnet", "mount isa",
        "mount perry", "muttaburra", "nanango", "noccundra", "normanton", "oakey", "oakley",
        "prairie", "quamby", "quilpie", "richmond", "ridgelands", "rockhampton", "roma",
        "springsure", "stamford", "stanthorpe", "st george", "stonehenge", "sunshine coast",
        "surat", "tambo", "tara", "taroom", "thangool", "theodore", "toowoomba", "townsville",
        "tower hill", "twin hills", "wandoan", "warra", "warwick", "wilpeena", "windorah",
        "winton", "wondai", "wyandra",
    ],
    "SA": [
        "balaklava", "bordertown", "ceduna", "clare", "gawler", "hawker",
        "jamestown", "kingscote", "kimba", "lock", "mindarie-halidon", "morphettville",
        "morphettville parks", "mount gambier", "murray bridge", "naracoorte", "oakbank",
        "penola", "penong", "port augusta", "port lincoln", "port pirie", "quorn",
        "roxby downs", "strathalbyn", "streaky bay", "tumby bay", "victoria park",
    ],
    "WA": [
        "albany", "ascot", "ashburton", "belmont", "beverley", "broome", "bunbury",
        "carnarvon", "collie", "derby", "dongara", "esperance", "exmouth", "fitzroy",
        "geraldton", "junction", "kalgoorlie", "kimberley", "kojonup", "kununurra", "landor",
        "lark hill", "laverton", "leinster", "leonora", "meekatharra", "mingenew", "moora",
        "mount barker", "mount magnet", "narrogin", "newman", "norseman", "northam", "perth",
        "pingrup", "pinjarra", "port hedland", "roebourne", "toodyay",
        "wiluna", "wyndham", "yalgoo", "york",
    ],
    "TAS": [
        "deloraine", "devonport", "hobart", "king island", "launceston", "longford", "spreyton",
    ],
    "NT": [
        "adelaide river", "alice springs", "barrow creek", "darwin", "katherine", "larrimah",
        "mataranka", "pine creek", "pioneer park", "renner", "tennant creek", "timber creek",
    ],
    "ACT": ["canberra", "canberra acton"],
    "HK": ["sha tin", "happy valley"],
    "SGP": ["kranji"],
}

# Build reverse lookup: venue -> state
_VENUE_TO_STATE: dict[str, str] = {}
for _state, _tracks in _STATE_TRACKS.items():
    for _track in _tracks:
        _VENUE_TO_STATE[_track] = _state


def normalize_venue(venue: str) -> str:
    """Normalize venue name: strip sponsors, apply aliases, lowercase.

    Returns canonical lowercase venue name (e.g., "randwick", "moonee valley").
    """
    if not venue:
        return ""
    v = venue.lower().strip()

    # Strip sponsor prefixes
    for prefix in SPONSOR_PREFIXES:
        if v.startswith(prefix + " "):
            v = v[len(prefix) + 1:]
            break

    v = v.strip()

    # Apply aliases
    if v in VENUE_ALIASES:
        return VENUE_ALIASES[v]

    return v


def venue_slug(venue: str) -> str:
    """Convert venue name to URL/ID slug (e.g., 'moonee-valley').

    Used for meeting IDs: f"{venue_slug(venue)}-{date.isoformat()}"
    """
    return normalize_venue(venue).replace(" ", "-").replace("'", "")


def guess_state(venue: str) -> str:
    """Guess state from venue name using the track database."""
    v = normalize_venue(venue)

    # Direct match
    if v in _VENUE_TO_STATE:
        return _VENUE_TO_STATE[v]

    # Try partial match
    for track, state in _VENUE_TO_STATE.items():
        if track in v or v in track:
            return state

    # Default to VIC if unknown
    return "VIC"


def is_known_venue(venue: str) -> bool:
    """Check if venue is in the known registry."""
    v = normalize_venue(venue)
    if v in _VENUE_TO_STATE:
        return True
    # Try partial match
    for track in _VENUE_TO_STATE:
        if track in v or v in track:
            return True
    return False


def get_all_venues() -> dict[str, str]:
    """Return dict of all known venues: {venue_name: state}."""
    return dict(_VENUE_TO_STATE)
