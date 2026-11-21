"""Floor rules. They live in data/maps.json, NOT in the solver.

To add a new floor you add one entry to "tiles" in maps.json (and use its letter
in a map). Nothing in solver.py has to change, because the solver only sees the
transition table that mdp.py builds from these rules.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "maps.json"

# Order matters: it is also the order used to break ties (first listed wins).
ACTIONS = ("up", "right", "down", "left")
DELTA = {"up": (-1, 0), "right": (0, 1), "down": (1, 0), "left": (0, -1)}
# the two sideways directions of each action (used only on ice)
PERPENDICULAR = {
    "up": ("left", "right"), "down": ("left", "right"),
    "left": ("up", "down"), "right": ("up", "down"),
}


@dataclass(frozen=True)
class TileRule:
    letter: str
    traversable: bool = True
    reward: float = 0.0           # reward for ENDING a move on this tile
    terminal: bool = False
    slippery: bool = False
    intended: float = 1.0         # probability of the intended direction
    perpendicular_each: float = 0.0


@dataclass(frozen=True)
class Config:
    discount: float
    tolerance: float
    max_iterations: int
    batch_seeds: tuple
    episodes_per_batch: int
    max_steps: int
    tiles: dict           # letter -> TileRule
    maps: dict            # name -> list[str]
    paper_v2_positions: tuple


def load_config(path: Path = DATA_FILE) -> Config:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    tiles = {}
    for letter, spec in raw["tiles"].items():
        slippery = spec.get("movement") == "slippery"
        tiles[letter] = TileRule(
            letter=letter,
            traversable=spec.get("traversable", True),
            reward=float(spec.get("reward", 0.0)),
            terminal=bool(spec.get("terminal", False)),
            slippery=slippery,
            intended=float(spec.get("intended", 1.0)) if slippery else 1.0,
            perpendicular_each=float(spec.get("perpendicular_each", 0.0)) if slippery else 0.0,
        )
    return Config(
        discount=float(raw["discount"]),
        tolerance=float(raw["solver"]["tolerance"]),
        max_iterations=int(raw["solver"]["max_iterations"]),
        batch_seeds=tuple(raw["simulation"]["batch_seeds"]),
        episodes_per_batch=int(raw["simulation"]["episodes_per_batch"]),
        max_steps=int(raw["simulation"]["max_steps"]),
        tiles=tiles,
        maps={"paper": raw["paper_map"], "extended": raw["extended_map"]},
        paper_v2_positions=tuple(tuple(p) for p in raw["paper_v2_positions"]),
    )
