"""Load Flemington 17/01/2026 test data into PuntyAI database."""

import asyncio
import sys
from datetime import date, datetime
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from punty.models.database import async_session, init_db
from punty.models.meeting import Meeting, Race, Runner


# Meeting data
MEETING_DATA = {
    "id": "flemington-2026-01-17",
    "venue": "Flemington",
    "date": date(2026, 1, 17),
    "track_condition": "Good 4",
    "weather": "Fine and Windy 27°C, Wind 32km SE",
    "rail_position": "9 Meters",
}

# Race data with runners
RACES_DATA = [
    {
        "race_number": 1,
        "name": "TAB We're On",
        "distance": 1000,
        "class_": "2YO OPEN",
        "prize_money": 150000,
        "start_time": datetime(2026, 1, 17, 12, 20),
        "runners": [
            {"barrier": 7, "horse_name": "BIG SKY", "jockey": "J J Childs", "trainer": "M Price & M Jnr", "current_odds": 2.25, "opening_odds": 2.10, "speed_map_position": "backmarker", "weight": 57.0},
            {"barrier": 1, "horse_name": "BORIDI", "jockey": "J L Nolen", "trainer": "P Moody & K Coleman", "current_odds": 4.20, "opening_odds": 4.80, "speed_map_position": "backmarker", "weight": 57.0},
            {"barrier": 4, "horse_name": "CIRCUS PRINCE", "jockey": "J R Houston", "trainer": "A Cosgriff", "current_odds": 81.00, "opening_odds": 71.00, "speed_map_position": "backmarker", "weight": 57.0},
            {"barrier": 3, "horse_name": "ISLE OF MONA", "jockey": "J J Mott", "trainer": "N Ryan", "current_odds": 9.00, "opening_odds": 5.00, "speed_map_position": "backmarker", "weight": 57.0},
            {"barrier": 2, "horse_name": "LADY PUISSANCE", "jockey": "J L Bates", "trainer": "A Leek", "current_odds": 23.00, "opening_odds": 34.00, "speed_map_position": "backmarker", "weight": 57.0},
            {"barrier": 5, "horse_name": "MIRADOR", "jockey": "J L Currie", "trainer": "B, W & J Hayes", "current_odds": 3.30, "opening_odds": 3.80, "speed_map_position": "backmarker", "weight": 57.0},
            {"barrier": 8, "horse_name": "NICCTINI", "jockey": "J H Coffey", "trainer": "Tony & C McEvoy", "scratched": True, "scratching_reason": "Vet"},
            {"barrier": 6, "horse_name": "ROAR TALENT", "jockey": "J M Dee", "trainer": "D Sutton", "current_odds": 13.00, "opening_odds": 18.00, "speed_map_position": "backmarker", "weight": 57.0},
        ],
    },
    {
        "race_number": 2,
        "name": "Victorian Bushfire Appeal Plate",
        "distance": 2520,
        "class_": "BM78",
        "prize_money": 130000,
        "start_time": datetime(2026, 1, 17, 12, 50),
        "runners": [
            {"barrier": 1, "horse_name": "ZWEIG", "jockey": "TBC", "trainer": "TBC", "current_odds": 8.00, "weight": 54.0},
            {"barrier": 2, "horse_name": "EAGLE ANGEL", "jockey": "TBC", "trainer": "TBC", "current_odds": 6.00, "weight": 54.0},
            {"barrier": 3, "horse_name": "POLITELY DUN", "jockey": "TBC", "trainer": "TBC", "current_odds": 4.50, "weight": 54.0},
            {"barrier": 4, "horse_name": "DICTIONARY", "jockey": "TBC", "trainer": "TBC", "current_odds": 15.00, "weight": 54.0},
            {"barrier": 5, "horse_name": "SCINTILLANTE", "jockey": "TBC", "trainer": "TBC", "current_odds": 12.00, "weight": 54.0},
            {"barrier": 6, "horse_name": "TARVUE", "jockey": "TBC", "trainer": "TBC", "current_odds": 21.00, "weight": 54.0},
            {"barrier": 7, "horse_name": "LEONCHROI", "jockey": "TBC", "trainer": "TBC", "current_odds": 3.80, "weight": 54.0},
            {"barrier": 8, "horse_name": "HOT TOO GO", "jockey": "TBC", "trainer": "TBC", "current_odds": 5.50, "weight": 54.0},
            {"barrier": 9, "horse_name": "NAVY HEART", "jockey": "TBC", "trainer": "TBC", "current_odds": 26.00, "weight": 54.0},
        ],
    },
    {
        "race_number": 3,
        "name": "Australian Guineas Day, 28 February",
        "distance": 2000,
        "class_": "3YO OPEN",
        "prize_money": 150000,
        "start_time": datetime(2026, 1, 17, 13, 23),
        "runners": [
            {"barrier": 1, "horse_name": "WRIGLEY FIELD", "jockey": "TBC", "trainer": "TBC", "current_odds": 4.00, "weight": 54.0},
            {"barrier": 2, "horse_name": "BLACK FROST", "jockey": "TBC", "trainer": "TBC", "current_odds": 7.00, "weight": 54.0},
            {"barrier": 3, "horse_name": "OUR CHIEF", "jockey": "TBC", "trainer": "TBC", "current_odds": 2.80, "weight": 54.0},
            {"barrier": 4, "horse_name": "THE MEAN FIDDLER", "jockey": "TBC", "trainer": "TBC", "current_odds": 5.00, "weight": 54.0},
            {"barrier": 5, "horse_name": "STOLE HIM", "jockey": "TBC", "trainer": "TBC", "current_odds": 15.00, "weight": 54.0},
            {"barrier": 6, "horse_name": "STOLE HIM", "jockey": "TBC", "trainer": "TBC", "current_odds": 15.00, "weight": 54.0},
            {"barrier": 7, "horse_name": "AGGRESSIVE", "jockey": "TBC", "trainer": "TBC", "current_odds": 21.00, "weight": 54.0},
            {"barrier": 8, "horse_name": "HELIOSON", "jockey": "TBC", "trainer": "TBC", "current_odds": 8.00, "weight": 54.0},
        ],
    },
    {
        "race_number": 4,
        "name": "Black Caviar Lightning Race Day, 14 February",
        "distance": 1800,
        "class_": "BM70",
        "prize_money": 80000,
        "start_time": datetime(2026, 1, 17, 13, 58),
        "runners": [
            {"barrier": 1, "horse_name": "CHAKADO", "jockey": "TBC", "trainer": "TBC", "current_odds": 15.00, "weight": 54.0},
            {"barrier": 2, "horse_name": "FIORENOT", "jockey": "TBC", "trainer": "TBC", "current_odds": 3.50, "weight": 54.0},
            {"barrier": 3, "horse_name": "ALL SO CLEAR", "jockey": "TBC", "trainer": "TBC", "current_odds": 12.00, "weight": 54.0},
            {"barrier": 4, "horse_name": "DARTBOARD", "jockey": "TBC", "trainer": "TBC", "current_odds": 26.00, "weight": 54.0},
            {"barrier": 5, "horse_name": "RAINBOW DELIGHT", "jockey": "TBC", "trainer": "TBC", "current_odds": 21.00, "weight": 54.0},
            {"barrier": 6, "horse_name": "DARK SIMBA", "jockey": "TBC", "trainer": "TBC", "current_odds": 8.00, "weight": 54.0},
            {"barrier": 8, "horse_name": "AMBASSADORIAL", "jockey": "TBC", "trainer": "TBC", "current_odds": 6.00, "weight": 54.0},
            {"barrier": 9, "horse_name": "FRONTLINE ACTION", "jockey": "TBC", "trainer": "TBC", "current_odds": 11.00, "weight": 54.0},
            {"barrier": 14, "horse_name": "YES I KNOW", "jockey": "TBC", "trainer": "TBC", "current_odds": 3.20, "weight": 54.0},
        ],
    },
    {
        "race_number": 5,
        "name": "VRC Super Saturday, 7 March",
        "distance": 1100,
        "class_": "F&M BM78",
        "prize_money": 80000,
        "start_time": datetime(2026, 1, 17, 14, 33),
        "runners": [
            {"barrier": 1, "horse_name": "LUNA CAT", "jockey": "TBC", "trainer": "TBC", "current_odds": 26.00, "weight": 54.0},
            {"barrier": 2, "horse_name": "OUT OF SQUARE", "jockey": "TBC", "trainer": "TBC", "current_odds": 9.00, "weight": 54.0},
            {"barrier": 3, "horse_name": "MYSTIC REIGN", "jockey": "TBC", "trainer": "TBC", "current_odds": 4.00, "weight": 54.0},
            {"barrier": 4, "horse_name": "EGERTON", "jockey": "TBC", "trainer": "TBC", "current_odds": 7.00, "weight": 54.0},
            {"barrier": 5, "horse_name": "VAIN CHAMPAGNE", "jockey": "TBC", "trainer": "TBC", "current_odds": 5.50, "weight": 54.0},
            {"barrier": 6, "horse_name": "CELERITY", "jockey": "TBC", "trainer": "TBC", "current_odds": 6.00, "weight": 54.0},
            {"barrier": 8, "horse_name": "SMART LITTLE MISS", "jockey": "TBC", "trainer": "TBC", "current_odds": 21.00, "weight": 54.0},
            {"barrier": 9, "horse_name": "THANKS GORGEOUS", "jockey": "TBC", "trainer": "TBC", "current_odds": 8.00, "weight": 54.0},
            {"barrier": 10, "horse_name": "MAUNA KEA MISS", "jockey": "TBC", "trainer": "TBC", "current_odds": 15.00, "weight": 54.0},
        ],
    },
    {
        "race_number": 6,
        "name": "VOBIS Gold Reef",
        "distance": 1600,
        "class_": "Quality",
        "prize_money": 175000,
        "start_time": datetime(2026, 1, 17, 15, 8),
        "runners": [
            {"barrier": 1, "horse_name": "JIMMY THE BEAR", "jockey": "J L Nolen", "trainer": "Patrick & M Payne", "current_odds": 5.00, "opening_odds": 5.00, "weight": 54.0},
            {"barrier": 2, "horse_name": "SOMEWHERE", "jockey": "J M Dee", "trainer": "A Alexander", "current_odds": 3.80, "opening_odds": 3.50, "weight": 54.0},
            {"barrier": 3, "horse_name": "MERRIGOLD", "jockey": "J J McNeil", "trainer": "J Sadler", "current_odds": 10.00, "opening_odds": 9.50, "weight": 54.0},
            {"barrier": 4, "horse_name": "MAKE IT SWEET", "jockey": "J J Duffy", "trainer": "C Weeding", "current_odds": 10.00, "opening_odds": 10.00, "weight": 54.0},
            {"barrier": 5, "horse_name": "HARRY'S YACHT", "jockey": "J L Currie", "trainer": "L & T Corstens & W Larkin", "current_odds": 3.00, "opening_odds": 3.00, "weight": 54.0},
            {"barrier": 6, "horse_name": "RESET THE JAZZ", "jockey": "J J Radley", "trainer": "B, W & J Hayes", "current_odds": 5.00, "opening_odds": 6.00, "weight": 54.0},
            {"barrier": 7, "horse_name": "MIXXIT", "jockey": "J E Pozman", "trainer": "E Jusufovic", "current_odds": 23.00, "opening_odds": 21.00, "weight": 54.0},
            {"barrier": 8, "horse_name": "SIRIUS STATEMENT", "jockey": "J B Mertens", "trainer": "G Bedggood", "scratched": True},
        ],
    },
    {
        "race_number": 7,
        "name": "Thank You Emergency Services",
        "distance": 1200,
        "class_": "BM84",
        "prize_money": 130000,
        "start_time": datetime(2026, 1, 17, 15, 43),
        "runners": [
            {"barrier": 2, "horse_name": "CELSIUS STAR", "jockey": "TBC", "trainer": "TBC", "current_odds": 11.00, "weight": 54.0},
            {"barrier": 3, "horse_name": "STEEL MOVE", "jockey": "TBC", "trainer": "TBC", "current_odds": 5.00, "weight": 54.0},
            {"barrier": 4, "horse_name": "RED GALAXY", "jockey": "TBC", "trainer": "TBC", "current_odds": 6.50, "weight": 54.0},
            {"barrier": 6, "horse_name": "DIRTY GRIN", "jockey": "TBC", "trainer": "TBC", "current_odds": 4.50, "weight": 54.0},
            {"barrier": 7, "horse_name": "PERILOUS FIGHTER", "jockey": "TBC", "trainer": "TBC", "current_odds": 9.00, "weight": 54.0},
            {"barrier": 10, "horse_name": "SWEETHEARTED", "jockey": "TBC", "trainer": "TBC", "current_odds": 4.80, "weight": 54.0},
            {"barrier": 11, "horse_name": "HIGHLAND HARLEY", "jockey": "TBC", "trainer": "TBC", "current_odds": 21.00, "weight": 54.0},
            {"barrier": 12, "horse_name": "IMMORTAL STAR", "jockey": "TBC", "trainer": "TBC", "current_odds": 15.00, "weight": 54.0},
        ],
    },
    {
        "race_number": 8,
        "name": "TAB Australian Cup Race Day, 28 March",
        "distance": 2000,
        "class_": "BM100",
        "prize_money": 150000,
        "start_time": datetime(2026, 1, 17, 16, 18),
        "runners": [
            {"barrier": 1, "horse_name": "BERKELEY SQUARE", "jockey": "TBC", "trainer": "TBC", "current_odds": 7.00, "weight": 53.0},
            {"barrier": 2, "horse_name": "SAINT GEORGE", "jockey": "TBC", "trainer": "TBC", "current_odds": 5.00, "weight": 53.0},
            {"barrier": 3, "horse_name": "BRAYDEN STAR", "jockey": "TBC", "trainer": "TBC", "current_odds": 9.00, "weight": 53.0},
            {"barrier": 4, "horse_name": "BOLD SOUL", "jockey": "TBC", "trainer": "TBC", "current_odds": 11.00, "weight": 53.0},
            {"barrier": 5, "horse_name": "PRECIOUS CHARM", "jockey": "TBC", "trainer": "TBC", "current_odds": 15.00, "weight": 53.0},
            {"barrier": 6, "horse_name": "SHAIYHAR", "jockey": "TBC", "trainer": "TBC", "current_odds": 12.00, "weight": 53.0},
            {"barrier": 7, "horse_name": "ARDAKAN", "jockey": "TBC", "trainer": "TBC", "current_odds": 21.00, "weight": 53.0},
            {"barrier": 8, "horse_name": "MAGNASPIN", "jockey": "TBC", "trainer": "TBC", "current_odds": 26.00, "weight": 53.0},
            {"barrier": 10, "horse_name": "JENNI'S MEADOW", "jockey": "TBC", "trainer": "TBC", "current_odds": 6.00, "weight": 53.0},
            {"barrier": 12, "horse_name": "DARKBONEE", "jockey": "TBC", "trainer": "TBC", "current_odds": 4.50, "weight": 53.0},
        ],
    },
    {
        "race_number": 9,
        "name": "Thank You Volunteers Trophy",
        "distance": 1400,
        "class_": "BM84",
        "prize_money": 130000,
        "start_time": datetime(2026, 1, 17, 16, 58),
        "runners": [
            {"barrier": 1, "horse_name": "NDOLA", "jockey": "TBC", "trainer": "TBC", "current_odds": 4.50, "weight": 54.0},
            {"barrier": 4, "horse_name": "INDISPENSABLE", "jockey": "TBC", "trainer": "TBC", "current_odds": 21.00, "weight": 54.0},
            {"barrier": 6, "horse_name": "RECON", "jockey": "TBC", "trainer": "TBC", "current_odds": 8.00, "weight": 54.0},
            {"barrier": 9, "horse_name": "PRANCING SPIRIT", "jockey": "TBC", "trainer": "TBC", "current_odds": 15.00, "weight": 54.0},
            {"barrier": 10, "horse_name": "UNTIL VALHALLA", "jockey": "TBC", "trainer": "TBC", "current_odds": 12.00, "weight": 54.0},
            {"barrier": 11, "horse_name": "CODIGO", "jockey": "TBC", "trainer": "TBC", "current_odds": 7.00, "weight": 54.0},
            {"barrier": 13, "horse_name": "SUPERNIMA", "jockey": "TBC", "trainer": "TBC", "current_odds": 6.50, "weight": 54.0},
            {"barrier": 14, "horse_name": "BOTANICAL BOY", "jockey": "TBC", "trainer": "TBC", "current_odds": 3.80, "weight": 54.0},
        ],
    },
    {
        "race_number": 10,
        "name": "Thank You Firefighters Plate",
        "distance": 1400,
        "class_": "3YO BM70",
        "prize_money": 150000,
        "start_time": datetime(2026, 1, 17, 17, 38),
        "runners": [
            {"barrier": 1, "horse_name": "BUCCLEUCH", "jockey": "TBC", "trainer": "TBC", "current_odds": 3.20, "weight": 54.0},
            {"barrier": 2, "horse_name": "JEWEL BANDIT", "jockey": "TBC", "trainer": "TBC", "current_odds": 5.00, "weight": 54.0},
            {"barrier": 4, "horse_name": "THE VOLTA", "jockey": "TBC", "trainer": "TBC", "current_odds": 6.00, "weight": 54.0},
            {"barrier": 6, "horse_name": "ITAZURA", "jockey": "TBC", "trainer": "TBC", "current_odds": 21.00, "weight": 54.0},
            {"barrier": 7, "horse_name": "YES MAXI", "jockey": "TBC", "trainer": "TBC", "current_odds": 15.00, "weight": 54.0},
            {"barrier": 8, "horse_name": "SASS APPEAL", "jockey": "TBC", "trainer": "TBC", "current_odds": 4.50, "weight": 54.0},
            {"barrier": 9, "horse_name": "EDEN ROSE", "jockey": "TBC", "trainer": "TBC", "current_odds": 9.00, "weight": 54.0},
        ],
    },
]


async def load_data():
    """Load Flemington test data into database."""
    await init_db()

    async with async_session() as db:
        # Create meeting
        meeting = Meeting(**MEETING_DATA)
        db.add(meeting)

        # Create races and runners
        for race_data in RACES_DATA:
            runners_data = race_data.pop("runners")

            race_id = f"{MEETING_DATA['id']}-r{race_data['race_number']}"
            race = Race(
                id=race_id,
                meeting_id=MEETING_DATA["id"],
                **race_data,
            )
            db.add(race)

            # Add runners
            for runner_data in runners_data:
                barrier = runner_data.get("barrier", 1)
                horse_name = runner_data.get("horse_name", "Unknown")
                runner_id = f"{race_id}-{barrier}-{horse_name.lower().replace(' ', '-')[:15]}"

                runner = Runner(
                    id=runner_id,
                    race_id=race_id,
                    **runner_data,
                )
                db.add(runner)

        await db.commit()
        print(f"Loaded Flemington meeting with {len(RACES_DATA)} races")
        print(f"Meeting ID: {MEETING_DATA['id']}")


if __name__ == "__main__":
    asyncio.run(load_data())
