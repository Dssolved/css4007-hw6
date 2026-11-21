import gymnasium as gym
import pytest

import run_part_b as pb


def policy_for(env):
    fixed = pb.load_fixed_policy(env)
    return lambda obs: fixed[int(obs)]


def test_fixed_policy_covers_exactly_the_nonterminal_states():
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="ansi")
    fixed = pb.load_fixed_policy(env)
    assert sorted(fixed) == [0, 1, 2, 3, 4, 6, 8, 9, 10, 13, 14]


def test_deterministic_fixed_policy_reaches_goal_in_six_steps():
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="ansi")
    ep = pb.run_episode(env, policy_for(env), seed=11000)
    assert ep["outcome"] == "goal" and ep["steps"] == 6 and ep["total_reward"] == 1.0


def test_terminal_on_the_last_allowed_step_counts_as_terminal_outcome():
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="ansi")
    assert pb.run_episode(env, policy_for(env), seed=1, max_steps=6)["outcome"] == "goal"
    assert pb.run_episode(env, policy_for(env), seed=1, max_steps=5)["outcome"] == "time limit"


def test_hole_is_recorded_as_hole():
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="ansi")
    ep = pb.run_episode(env, lambda obs: 2 if obs == 0 else 1, seed=1)    # 0 -> right(1) -> down(5 = hole)
    assert ep["outcome"] == "hole" and ep["total_reward"] == 0.0 and ep["final_state"] == 5


def test_same_seed_gives_same_slippery_episode():
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True, render_mode="ansi")
    a = pb.run_episode(env, policy_for(env), seed=22003)
    b = pb.run_episode(env, policy_for(env), seed=22003)
    assert a["log"] == b["log"] and a["outcome"] == b["outcome"]


def test_env_time_limit_is_100():
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    assert env.spec.max_episode_steps == 100
