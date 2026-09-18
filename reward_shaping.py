# pyright: reportMissingTypeStubs=false, reportMissingParameterType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
"""
Tunable reward-shaping framework.

This module is deliberately dumb: it provides building blocks (dense reward
terms, a linear-annealing wrapper, and an assembler) and holds no opinion on
what the "right" weights or schedules are. Do that tuning in quick_start.py's
REWARD_TERMS list, which is the one place you should need to touch to change
what the agent is rewarded for.

Two kinds of term:
  - constant weight:  RewardTerm(name, factory, weight=W)
  - annealed weight:  RewardTerm(name, factory, anneal=AnnealSpec(start, end, steps))
                       (the `weight` field is ignored when `anneal` is set)

"steps" in AnnealSpec are *this process's* env.step() calls (one call =
one action_repeat window, e.g. 8 physics ticks), not global training
timesteps. Each of the n_proc worker processes builds its own env and its
own RewardTerm/AnnealedRewardFunction instances independently, so there is
no cross-process synchronization — see the AnnealSpec docstring for how to
convert a global-timestep target into a per-process step count.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from rlgym.api import AgentID, RewardFunction
from rlgym.rocket_league import common_values
from rlgym.rocket_league.api import GameState

Term = RewardFunction[AgentID, GameState, float]


# ---------------------------------------------------------------------------
# Dense shaping terms (rlgym.rocket_league.reward_functions only has
# GoalReward / TouchReward / CombinedReward; everything else here is custom).
# ---------------------------------------------------------------------------


class PlayerToBallDistanceReward(RewardFunction[AgentID, GameState, float]):
    """exp(-dist / (CAR_MAX_SPEED * dispersion)): 1.0 when touching the ball,
    decaying toward 0 as the car gets farther away. Smaller `dispersion`
    falls off faster (rewards only being very close)."""

    def __init__(self, dispersion: float = 1.0):
        self.dispersion = dispersion

    def reset(self, agents, initial_state, shared_info) -> None:
        pass

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            dist = float(np.linalg.norm(car.physics.position - state.ball.position))
            rewards[agent] = float(
                np.exp(-dist / (common_values.CAR_MAX_SPEED * self.dispersion))
            )
        return rewards


class VelocityPlayerToBallReward(RewardFunction[AgentID, GameState, float]):
    """Component of the car's velocity pointing straight at the ball,
    normalized by CAR_MAX_SPEED. Range is roughly [-1, 1]."""

    def reset(self, agents, initial_state, shared_info) -> None:
        pass

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            to_ball = state.ball.position - car.physics.position
            dist = float(np.linalg.norm(to_ball))
            if dist < 1e-8:
                rewards[agent] = 0.0
                continue
            direction = to_ball / dist
            speed_toward_ball = float(np.dot(car.physics.linear_velocity, direction))
            rewards[agent] = speed_toward_ball / common_values.CAR_MAX_SPEED
        return rewards


class VelocityBallToGoalReward(RewardFunction[AgentID, GameState, float]):
    """Component of the ball's velocity pointing at the agent's *attacking*
    goal (opponent's net), normalized by BALL_MAX_SPEED. Range is roughly
    [-1, 1]. This is team-relative: blue cars are scored against orange's
    net and vice versa."""

    def reset(self, agents, initial_state, shared_info) -> None:
        pass

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {}
        ball_pos = state.ball.position
        ball_vel = state.ball.linear_velocity
        blue_target = np.asarray(common_values.ORANGE_GOAL_BACK, dtype=np.float32)
        orange_target = np.asarray(common_values.BLUE_GOAL_BACK, dtype=np.float32)
        for agent in agents:
            car = state.cars[agent]
            target = blue_target if car.is_blue else orange_target
            to_goal = target - ball_pos
            dist = float(np.linalg.norm(to_goal))
            if dist < 1e-8:
                rewards[agent] = 0.0
                continue
            direction = to_goal / dist
            speed_toward_goal = float(np.dot(ball_vel, direction))
            rewards[agent] = speed_toward_goal / common_values.BALL_MAX_SPEED
        return rewards


class BoostPickupReward(RewardFunction[AgentID, GameState, float]):
    """Fraction of boost (0-100 scale) gained since the previous step.
    Only positive deltas count, so using boost doesn't get punished here
    (pair with a separate SaveBoost-style term if you want that)."""

    def __init__(self):
        self._prev_boost: Dict[AgentID, float] = {}

    def reset(self, agents, initial_state, shared_info) -> None:
        self._prev_boost = {
            agent: initial_state.cars[agent].boost_amount for agent in agents
        }

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {}
        for agent in agents:
            boost = state.cars[agent].boost_amount
            prev = self._prev_boost.get(agent, boost)
            rewards[agent] = max(0.0, boost - prev) / 100.0
            self._prev_boost[agent] = boost
        return rewards


# ---------------------------------------------------------------------------
# Annealing
# ---------------------------------------------------------------------------


@dataclass
class AnnealSpec:
    """Linearly interpolate a term's weight from start_weight to end_weight
    over `steps` agent-steps in *this* env process (env.step() calls, not
    global timesteps across all n_proc processes).

    To target a global-timestep fraction: with `n_proc` parallel envs and a
    `timestep_limit` of T total agent-steps, each process sees roughly
    T / n_proc of them, so to anneal out over the first X% of training set
        steps = X * T / n_proc
    """

    start_weight: float
    end_weight: float
    steps: int


class AnnealedRewardFunction(RewardFunction[AgentID, GameState, float]):
    """Wraps a reward term and linearly ramps its weight over time."""

    def __init__(self, inner: Term, anneal: AnnealSpec):
        self.inner = inner
        self.anneal = anneal
        self._step = 0

    def reset(self, agents, initial_state, shared_info) -> None:
        self.inner.reset(agents, initial_state, shared_info)

    def current_weight(self) -> float:
        if self.anneal.steps <= 0:
            return self.anneal.end_weight
        frac = min(1.0, self._step / self.anneal.steps)
        return self.anneal.start_weight + frac * (
            self.anneal.end_weight - self.anneal.start_weight
        )

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        weight = self.current_weight()
        self._step += 1
        rewards = self.inner.get_rewards(
            agents, state, is_terminated, is_truncated, shared_info
        )
        return {agent: reward * weight for agent, reward in rewards.items()}


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


@dataclass
class RewardTerm:
    """One line of your reward design.

    `factory` builds a fresh reward-function instance (CombinedReward /
    AnnealedRewardFunction hold state, so each env process needs its own).
    `weight` is a plain multiplier and is ignored if `anneal` is set.
    """

    name: str
    factory: Callable[[], Term]
    weight: float = 1.0
    anneal: Optional[AnnealSpec] = None


def build_combined_reward(terms: List[RewardTerm]):
    from rlgym.rocket_league.reward_functions import CombinedReward

    rewards_and_weights = []
    for term in terms:
        reward_fn = term.factory()
        if term.anneal is not None:
            rewards_and_weights.append(
                (AnnealedRewardFunction(reward_fn, term.anneal), 1.0)
            )
        else:
            rewards_and_weights.append((reward_fn, term.weight))
    return CombinedReward(*rewards_and_weights)
