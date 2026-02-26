"""Rotating promotional image selection for social media posts."""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

IMAGES_DIR = Path("data/social_images")
INDEX_FILE = IMAGES_DIR / ".rotation_index"


def get_next_image() -> Path | None:
    """Get the next image in rotation. Returns path or None if no images."""
    images = sorted(IMAGES_DIR.glob("punty_promo_*.png"))
    if not images:
        logger.debug("No promotional images found in %s", IMAGES_DIR)
        return None

    # Read current index
    idx = 0
    if INDEX_FILE.exists():
        try:
            idx = json.loads(INDEX_FILE.read_text()).get("index", 0)
        except Exception:
            idx = 0

    # Wrap around
    idx = idx % len(images)
    image_path = images[idx]

    # Advance index
    try:
        INDEX_FILE.write_text(json.dumps({"index": idx + 1}))
    except Exception as e:
        logger.warning("Could not update rotation index: %s", e)

    logger.info("Selected promo image: %s (index %d/%d)", image_path.name, idx + 1, len(images))
    return image_path
