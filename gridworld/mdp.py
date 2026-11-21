"""Builds the known transition model from a map and the floor rules.

State = (row, col), 1-indexed, row 1 is the top row.
transitions(state, action) -> list of (probability, next_state, reward)

Rules implemented here (and nowhere else):
* Terminal tiles (G, H) have no transitions: value 0, no reward after termination.
* A move is resolved per outcome: the slip direction is chosen with the
  probabilities of the tile the move STARTS from (ice: 0.8 / 0.1 / 0.1).
* An outcome that would leave the grid or enter a wall '#' keeps the agent in place.
* The reward is the reward of the tile where that outcome ends (the same tile
  as the start if the move was blocked). Blocked outcomes are handled separately
  for every slip direction.
"""
from __future__ import annotations

from .rules import ACTIONS, DELTA, PERPENDICULAR, Config, TileRule


class GridMDP:
    def __init__(self, rows: list[str], tiles: dict[str, TileRule], discount: float):
        self.rows = list(rows)
        self.tiles = tiles
        self.discount = discount
        self.n_rows = len(rows)
        self.n_cols = len(rows[0])
        assert all(len(r) == self.n_cols for r in rows), "map rows must have equal length"
        for r in rows:
            for ch in r:
                assert ch in tiles, f"unknown tile letter {ch!r} (add it to data/maps.json)"
        self.states = [(r, c) for r in range(1, self.n_rows + 1) for c in range(1, self.n_cols + 1)
                       if tiles[self.letter((r, c))].traversable]
        self.start = next(s for s in self.states if self.letter(s) == "S")
        self._table = {
            (s, a): self._build(s, a)
            for s in self.states if not self.is_terminal(s) for a in ACTIONS
        }

    # ---- map access -------------------------------------------------------
    def letter(self, s) -> str:
        return self.rows[s[0] - 1][s[1] - 1]

    def rule(self, s) -> TileRule:
        return self.tiles[self.letter(s)]

    def is_terminal(self, s) -> bool:
        return self.rule(s).terminal

    def nonterminal_states(self):
        return [s for s in self.states if not self.is_terminal(s)]

    def _step_from(self, s, direction):
        dr, dc = DELTA[direction]
        t = (s[0] + dr, s[1] + dc)
        inside = 1 <= t[0] <= self.n_rows and 1 <= t[1] <= self.n_cols
        if not inside or not self.rule(t).traversable:
            return s                      # boundary or wall: stay in place
        return t

    def _build(self, s, action):
        rule = self.rule(s)              # slipperiness is decided by the START tile
        if rule.slippery:
            side1, side2 = PERPENDICULAR[action]
            options = [(action, rule.intended), (side1, rule.perpendicular_each),
                       (side2, rule.perpendicular_each)]
        else:
            options = [(action, 1.0)]
        out = []
        for direction, p in options:
            t = self._step_from(s, direction)
            out.append((p, t, self.rule(t).reward))
        return out

    # ---- public model -----------------------------------------------------
    def transitions(self, s, a):
        return self._table[(s, a)]

    def q_value(self, V, s, a):
        g = self.discount
        return sum(p * (r + g * V[t]) for p, t, r in self.transitions(s, a))
