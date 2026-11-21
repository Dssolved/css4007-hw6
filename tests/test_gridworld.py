import dataclasses

import pytest

from gridworld.checks import run_checks
from gridworld.mdp import GridMDP
from gridworld.rules import ACTIONS, TileRule, load_config
from gridworld.simulate import run_episode
from gridworld.solver import extract_policy, value_iteration

CFG = load_config()


def make(name):
    return GridMDP(CFG.maps[name], CFG.tiles, CFG.discount)


def test_all_rule_checks_pass_on_both_maps():
    for name in ("paper", "extended"):
        results = run_checks(make(name), name, CFG.max_steps, CFG.discount)
        assert results and all(r["passed"] for r in results), [r for r in results if not r["passed"]]


def test_paper_v1_and_v2_values():
    vi = value_iteration(make("paper"), CFG.tolerance, CFG.max_iterations)
    v1, v2 = vi.history[1], vi.history[2]
    assert v1[(1, 3)] == 10.0 and v1[(1, 1)] == -1.0 and v1[(2, 3)] == -1.0
    assert v2[(1, 2)] == pytest.approx(8.0)
    assert v2[(1, 3)] == pytest.approx(10.0)
    assert v2[(2, 3)] == pytest.approx(5.21)


def test_convergence_rule_and_safeguard():
    mdp = make("paper")
    vi = value_iteration(mdp, CFG.tolerance, CFG.max_iterations)
    assert vi.converged and vi.final_change < CFG.tolerance and vi.iterations < CFG.max_iterations
    short = value_iteration(mdp, CFG.tolerance, 3)
    assert not short.converged and short.iterations == 3


def test_solution_is_a_fixed_point_of_the_bellman_update():
    mdp = make("extended")
    vi = value_iteration(mdp, 1e-12, CFG.max_iterations)
    for s in mdp.nonterminal_states():
        assert vi.V[s] == pytest.approx(max(mdp.q_value(vi.V, s, a) for a in ACTIONS), abs=1e-9)
    for s in mdp.states:
        if mdp.is_terminal(s):
            assert vi.V[s] == 0.0


def test_extended_map_changes_the_start_decision():
    p = extract_policy(make("paper"), value_iteration(make("paper"), CFG.tolerance, CFG.max_iterations).V)[0]
    e = extract_policy(make("extended"), value_iteration(make("extended"), CFG.tolerance, CFG.max_iterations).V)[0]
    assert p[(3, 1)] == "right" and e[(3, 1)] == "up"


def test_tie_break_takes_first_action_in_order():
    # one cell next to the goal: 'right' is the only good action
    tiles = {**CFG.tiles}
    mdp = GridMDP(["SG"], tiles, CFG.discount)
    vi = value_iteration(mdp, CFG.tolerance, CFG.max_iterations)
    policy, ties = extract_policy(mdp, vi.V)
    assert policy[(1, 1)] == "right"
    mdp2 = GridMDP(["SH"], tiles, CFG.discount)         # up, down, left all stay in place (same value): a real tie
    vi2 = value_iteration(mdp2, CFG.tolerance, CFG.max_iterations)
    policy2, ties2 = extract_policy(mdp2, vi2.V)
    assert ties2[(1, 1)] == ["up", "down", "left"] and policy2[(1, 1)] == "up"


def test_new_floor_needs_only_a_rule_not_a_solver_change():
    tiles = dict(CFG.tiles)
    tiles["X"] = TileRule(letter="X", reward=-4.0)
    mdp = GridMDP(["SXG"], tiles, CFG.discount)
    vi = value_iteration(mdp, CFG.tolerance, CFG.max_iterations)
    # entering X costs 4, entering G pays 10 from X: V(S) = -4 + 0.9 * 10
    assert vi.V[(1, 1)] == pytest.approx(-4 + 0.9 * 10)


def test_simulation_is_reproducible_and_stops_at_terminal():
    mdp = make("paper")
    vi = value_iteration(mdp, CFG.tolerance, CFG.max_iterations)
    policy, _ = extract_policy(mdp, vi.V)
    a = run_episode(mdp, policy, 11000, CFG.max_steps, CFG.discount)
    b = run_episode(mdp, policy, 11000, CFG.max_steps, CFG.discount)
    assert a == b
    if a["outcome"] != "time limit":
        assert mdp.is_terminal(a["path"][-1]) and len(a["rewards"]) == a["steps"]


def test_time_limit_when_policy_never_ends():
    mdp = make("paper")
    stuck = {s: "up" for s in mdp.nonterminal_states()}
    ep = run_episode(mdp, stuck, 1, 7, CFG.discount)
    assert ep["outcome"] == "time limit" and ep["steps"] == 7
