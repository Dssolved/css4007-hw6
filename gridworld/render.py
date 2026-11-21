"""Text printing of value maps and policy arrows."""
from __future__ import annotations

from .mdp import GridMDP

ARROW = {"up": "^", "right": ">", "down": "v", "left": "<"}


def value_map(mdp: GridMDP, V, decimals: int = 3) -> str:
    width = decimals + 5
    lines = []
    for r in range(1, mdp.n_rows + 1):
        cells = []
        for c in range(1, mdp.n_cols + 1):
            s = (r, c)
            if not mdp.rule(s).traversable:
                cells.append("#".center(width))
            else:
                cells.append(f"{V[s]:.{decimals}f}".rjust(width))
        lines.append(" ".join(cells))
    return "\n".join(lines)


def policy_map(mdp: GridMDP, policy) -> str:
    lines = []
    for r in range(1, mdp.n_rows + 1):
        cells = []
        for c in range(1, mdp.n_cols + 1):
            s = (r, c)
            ch = mdp.letter(s)
            if s in policy:
                cells.append(ARROW[policy[s]])
            else:
                cells.append(ch)          # G, H or '#'
        lines.append("  ".join(cells))
    return "\n".join(lines)


def tile_map(mdp: GridMDP) -> str:
    return "\n".join("  ".join(row) for row in mdp.rows)
