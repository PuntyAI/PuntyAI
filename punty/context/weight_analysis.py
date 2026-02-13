"""Weight-specific form analysis from race history.

Analyses how a horse performs at different weight ranges by parsing
form_history data. Identifies optimal weight bands and flags warnings
when a horse carries significantly more weight than its successful range.
"""

from __future__ import annotations


def _parse_weight(value) -> float | None:
    """Robustly parse a weight value that may be str, int, float, or None."""
    if value is None:
        return None
    try:
        w = float(str(value))
        return w if 45 <= w <= 75 else None  # Sanity range for horse racing
    except (ValueError, TypeError):
        return None


def _parse_position(value) -> int | None:
    """Parse a finishing position."""
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def analyse_weight_form(form_history: list[dict], current_weight: float) -> dict:
    """Analyse a horse's form at different weight ranges.

    Groups past starts into weight bands relative to current weight:
    - lighter: >= 2kg less than current
    - similar: within 2kg of current
    - heavier: >= 2kg more than current

    Args:
        form_history: List of past start dicts (most recent first).
        current_weight: Weight carried in today's race.

    Returns:
        Dict with bands, weight_change, optimal_band, and warning.
        Empty dict if insufficient data.
    """
    if not form_history or not isinstance(form_history, list):
        return {}

    current = _parse_weight(current_weight)
    if not current:
        return {}

    # Parse starts with valid weight and position
    starts = []
    for entry in form_history:
        w = _parse_weight(entry.get("weight"))
        pos = _parse_position(entry.get("position") or entry.get("pos"))
        if w is not None and pos is not None:
            starts.append({"weight": w, "position": pos})

    if not starts:
        return {}

    # Group into bands relative to current weight
    bands = {
        "lighter": {"starts": 0, "wins": 0, "places": 0, "weights": []},
        "similar": {"starts": 0, "wins": 0, "places": 0, "weights": []},
        "heavier": {"starts": 0, "wins": 0, "places": 0, "weights": []},
    }

    for s in starts:
        diff = s["weight"] - current
        if diff <= -2:
            band = "lighter"
        elif diff >= 2:
            band = "heavier"
        else:
            band = "similar"

        bands[band]["starts"] += 1
        bands[band]["weights"].append(s["weight"])
        if s["position"] == 1:
            bands[band]["wins"] += 1
        if s["position"] <= 3:
            bands[band]["places"] += 1

    # Build weight ranges for display
    for band_data in bands.values():
        weights = band_data["weights"]
        if weights:
            band_data["weight_range"] = f"{min(weights):.0f}-{max(weights):.0f}kg"
        del band_data["weights"]  # Don't need raw weights in output

    # Weight change from last start
    last_weight = _parse_weight(form_history[0].get("weight"))
    weight_change = None
    if last_weight is not None:
        weight_change = round(current - last_weight, 1)

    # Find optimal band (highest win rate, min 2 starts)
    optimal_band = None
    best_rate = -1
    for band_name, bd in bands.items():
        if bd["starts"] >= 2:
            rate = bd["wins"] / bd["starts"]
            if rate > best_rate:
                best_rate = rate
                optimal_band = band_name

    # Generate warning
    warning = None

    # Check if carrying heavier than successful range
    lighter = bands["lighter"]
    similar = bands["similar"]
    heavier = bands["heavier"]

    # Warning: 0 wins at heavier band with enough starts
    if heavier["starts"] >= 2 and heavier["wins"] == 0:
        warning = f"0/{heavier['starts']} winning at {heavier.get('weight_range', 'heavier weights')}"
        if current >= (min(float(heavier.get('weight_range', '0').split('-')[0].rstrip('kg')) if heavier.get('weight_range') else 99, 99)):
            warning += f" — today {current:.0f}kg"

    # Warning: win rate drops dramatically at similar/heavier vs lighter
    if not warning and lighter["starts"] >= 2 and lighter["wins"] >= 1:
        lighter_rate = lighter["wins"] / lighter["starts"]
        current_band = similar if abs(weight_change or 0) < 2 else heavier
        if current_band["starts"] >= 2:
            current_rate = current_band["wins"] / current_band["starts"]
            if current_rate < lighter_rate * 0.5:
                warning = (
                    f"{lighter['wins']}/{lighter['starts']} at {lighter.get('weight_range', 'lighter')}"
                    f" vs {current_band['wins']}/{current_band['starts']} at heavier"
                )

    return {
        "bands": bands,
        "weight_change": weight_change,
        "optimal_band": optimal_band,
        "warning": warning,
    }
