"""A module containing 'Replacement' model."""

from dataclasses import dataclass


@dataclass
class Replacement:
    """Replacement class definition."""

    pattern: str
    replacement: str
