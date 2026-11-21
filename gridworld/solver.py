"""Value iteration written from scratch. It knows nothing about floor letters:
it only calls mdp.transitions / mdp.q_value / mdp.nonterminal_states."""
from __future__ import annotations

from dataclasses import dataclass, field

from .mdp import GridMDP
from .rules import ACTIONS

TIE_TOLERANCE = 1e-6   # two actions whose Q-values differ by less than this count as tied


@dataclass
class VIResult:
    V: dict
    iterations: int            # number of full updates performed (the last one had change < tolerance)
    final_change: float
    converged: bool
    history: dict = field(default_factory=dict)   # k -> copy of V_k, kept for k = 0, 1, 2 only


def value_iteration(mdp: GridMDP, tolerance: float, max_iterations: int) -> VIResult:
    V = {s: 0.0 for s in mdp.states}              # V_0 = 0 everywhere, terminal tiles stay 0
    history = {0: dict(V)}
    change = float("inf")
    k = 0
    while k < max_iterations:
        new_V = dict(V)                            # synchronous: every Q uses the PREVIOUS V
        change = 0.0
        for s in mdp.nonterminal_states():
            new_V[s] = max(mdp.q_value(V, s, a) for a in ACTIONS)
            change = max(change, abs(new_V[s] - V[s]))
        V = new_V
        k += 1
        if k <= 2:
            history[k] = dict(V)
        if change < tolerance:
            return VIResult(V, k, change, True, history)
    return VIResult(V, k, change, False, history)


def action_values(mdp: GridMDP, V, s) -> dict:
    return {a: mdp.q_value(V, s, a) for a in ACTIONS}


def best_actions(mdp: GridMDP, V, s, tie_tolerance: float = TIE_TOLERANCE):
    """Returns (chosen_action, tied_actions). Tie-break: among the actions within
    tie_tolerance of the best Q-value, the FIRST in the order up, right, down, left."""
    q = action_values(mdp, V, s)
    best = max(q.values())
    tied = [a for a in ACTIONS if best - q[a] < tie_tolerance]
    return tied[0], tied


def extract_policy(mdp: GridMDP, V, tie_tolerance: float = TIE_TOLERANCE):
    policy, ties = {}, {}
    for s in mdp.nonterminal_states():
        chosen, tied = best_actions(mdp, V, s, tie_tolerance)
        policy[s] = chosen
        if len(tied) > 1:
            ties[s] = tied
    return policy, ties


def smallest_gap(mdp: GridMDP, V) -> tuple:
    """Smallest positive gap between the best and the second-best Q-value over all
    states, to show how close any non-tied decision is (state, gap)."""
    worst = (None, float("inf"))
    for s in mdp.nonterminal_states():
        qs = sorted(action_values(mdp, V, s).values(), reverse=True)
        gap = qs[0] - qs[1]
        if gap >= TIE_TOLERANCE and gap < worst[1]:
            worst = (s, gap)
    return worst
