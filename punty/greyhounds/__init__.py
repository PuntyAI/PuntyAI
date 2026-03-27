"""Greyhound racing module — VIC, NSW, QLD.

Separate from thoroughbred code. Uses its own models, scrapers, and data pipeline.
Greyhound-specific considerations:
  - Box numbers 1-8 (reserves 9-10)
  - No jockey — trainer is primary connection
  - Grades: C0-C6, Gr1-5, Maiden, FFA, Masters, Novice
  - Distances: 295m-700m+ (track-dependent)
  - Track types: oval (most), straight (Healesville etc.)
  - States: VIC, NSW, QLD only
"""
