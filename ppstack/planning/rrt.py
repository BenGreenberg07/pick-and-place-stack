"""RRT* in configuration space.

The grid planner in `astar.py` plans for the end effector in the table frame.
That is fast, optimal on its grid, and structurally unable to say anything
about where the elbow goes: it plans in task space, and the arm's collisions
happen in configuration space. For an arm with redundancy the two are not the
same problem, and a tip path can be perfectly clear while every joint solution
along it buries a link in an obstacle.

So this is the other half. RRT* samples joint vectors directly, connects them
with edges that are collision-checked along their whole length by
`ArmCollisionChecker`, and rewires to drive the path cost down as it goes. It
is slower and its answer is only asymptotically optimal, but every
configuration it returns is one the entire arm can actually be in.

The two planners are benchmarked against each other in `benchmark.py`.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from ..kinematics.arm import PlanarArm
from .collision import ArmCollisionChecker


class RRTFailed(RuntimeError):
    """Raised when no collision-free path was found within the budget."""


@dataclass
class RRTResult:
    path: list[np.ndarray]
    cost: float
    nodes: int
    collision_checks: int
    iterations: int
    elapsed_s: float
    cost_history: list[tuple[int, float]] = field(default_factory=list)

    @property
    def waypoints(self) -> np.ndarray:
        return np.vstack(self.path)


class RRTStar:
    """Asymptotically optimal sampling planner over joint space.

    Cost is joint-space path length, which is a reasonable proxy for how much
    the motors actually have to turn. The neighbourhood radius follows the
    standard RRT* shrinking schedule

        r(n) = min(gamma * (log n / n)^(1/d), max_step)

    with d the number of joints. The schedule is what makes the algorithm
    asymptotically optimal rather than merely complete: it has to shrink slowly
    enough that the graph stays connected and fast enough that rewiring stays
    cheap.
    """

    def __init__(
        self,
        arm: PlanarArm,
        checker: ArmCollisionChecker,
        *,
        max_step: float = 0.35,
        goal_bias: float = 0.12,
        gamma: float = 3.0,
        seed: int = 0,
    ) -> None:
        self.arm = arm
        self.checker = checker
        self.max_step = max_step
        self.goal_bias = goal_bias
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)
        self.limits = arm._limits

    # ------------------------------------------------------------- internals

    def _sample(self, goals: np.ndarray) -> np.ndarray:
        """Uniform over the joint box, biased towards a goal now and then.

        Without the bias, a narrow passage in configuration space is found only
        by luck. With too much of it the tree stops exploring and drives
        repeatedly into whatever wall lies between it and the goal.
        """
        if self.rng.random() < self.goal_bias:
            return goals[self.rng.integers(len(goals))].copy()
        return self.rng.uniform(self.limits[:, 0], self.limits[:, 1])

    def _steer(self, frm: np.ndarray, to: np.ndarray) -> np.ndarray:
        d = to - frm
        n = float(np.linalg.norm(d))
        if n <= self.max_step:
            return to.copy()
        return frm + d * (self.max_step / n)

    # ------------------------------------------------------------------ plan

    def plan(
        self,
        q_start,
        goal_configs,
        *,
        max_iterations: int = 3000,
        goal_tolerance: float = 0.08,
        time_budget_s: float = 5.0,
        stop_on_first: bool = False,
    ) -> RRTResult:
        """Plan from `q_start` to any of `goal_configs`.

        Goals are supplied as configurations rather than as a tip position on
        purpose: choosing which IK solution to aim at is a decision the caller
        should own, and handing over several lets the planner pick whichever
        elbow branch it can actually reach.

        `stop_on_first` turns the planner back into plain RRT, which is what the
        benchmark uses to show what the star is buying.
        """
        q_start = np.asarray(q_start, dtype=float)
        goals = np.atleast_2d(np.asarray(goal_configs, dtype=float))
        if self.checker.collides(q_start):
            raise RRTFailed("the start configuration is already in collision")
        reachable = [g for g in goals if not self.checker.collides(g)]
        if not reachable:
            raise RRTFailed("every goal configuration is in collision")
        goals = np.vstack(reachable)

        t0 = time.perf_counter()
        checks0 = self.checker.checks
        nodes = [q_start]
        parent = [-1]
        cost = [0.0]
        best_goal: int | None = None
        best_cost = math.inf
        history: list[tuple[int, float]] = []
        d = self.arm.n_joints
        it = 0

        for it in range(1, max_iterations + 1):
            if time.perf_counter() - t0 > time_budget_s:
                break

            q_rand = self._sample(goals)
            arr = np.vstack(nodes)
            nearest = int(np.argmin(np.linalg.norm(arr - q_rand, axis=1)))
            q_new = self._steer(nodes[nearest], q_rand)
            q_new = np.clip(q_new, self.limits[:, 0], self.limits[:, 1])

            if not self.checker.path_clear(nodes[nearest], q_new):
                continue

            # Choose the cheapest feasible parent inside the shrinking radius,
            # not merely the nearest node. This is the first half of what
            # separates RRT* from RRT.
            n = len(nodes)
            radius = min(self.gamma * (math.log(n + 1) / (n + 1)) ** (1.0 / d), self.max_step)
            dists = np.linalg.norm(arr - q_new, axis=1)
            near = np.flatnonzero(dists <= radius)

            best_parent, best_new_cost = nearest, cost[nearest] + float(dists[nearest])
            for j in near:
                c = cost[j] + float(dists[j])
                if c < best_new_cost and self.checker.path_clear(nodes[j], q_new):
                    best_parent, best_new_cost = int(j), c

            nodes.append(q_new)
            parent.append(best_parent)
            cost.append(best_new_cost)
            new_index = len(nodes) - 1

            # Rewire: any neighbour that would be cheaper reached through the
            # new node gets re-parented onto it. This is the second half, and
            # the reason the path keeps straightening as the tree grows.
            for j in near:
                j = int(j)
                through = best_new_cost + float(dists[j])
                if through < cost[j] and self.checker.path_clear(q_new, nodes[j]):
                    parent[j] = new_index
                    delta = through - cost[j]
                    cost[j] = through
                    # Push the saving down the subtree so descendant costs stay
                    # consistent; a stale cost silently disables later rewiring.
                    stack = [j]
                    while stack:
                        cur = stack.pop()
                        for k in range(len(nodes)):
                            if parent[k] == cur and k != j:
                                cost[k] += delta
                                stack.append(k)

            goal_gap = float(np.linalg.norm(goals - q_new, axis=1).min())
            if goal_gap <= goal_tolerance and best_new_cost < best_cost:
                best_goal, best_cost = new_index, best_new_cost
                history.append((it, best_cost))
                if stop_on_first:
                    break

        if best_goal is None:
            raise RRTFailed(
                f"no collision-free path after {it} iterations and "
                f"{self.checker.checks - checks0} collision checks"
            )

        path = []
        node = best_goal
        while node != -1:
            path.append(nodes[node])
            node = parent[node]
        path.reverse()

        return RRTResult(
            path=path,
            cost=best_cost,
            nodes=len(nodes),
            collision_checks=self.checker.checks - checks0,
            iterations=it,
            elapsed_s=time.perf_counter() - t0,
            cost_history=history,
        )


def shortcut_configs(
    path: list[np.ndarray], checker: ArmCollisionChecker, *, rounds: int = 60, seed: int = 0
) -> list[np.ndarray]:
    """Randomised shortcutting of a configuration-space path.

    RRT* paths are asymptotically optimal and, at any finite number of samples,
    visibly kinked. Repeatedly trying to replace a random sub-path with the
    straight line between its endpoints removes most of that for a tiny
    fraction of the planning cost.
    """
    rng = np.random.default_rng(seed)
    out = [np.asarray(p, dtype=float) for p in path]
    for _ in range(rounds):
        if len(out) <= 2:
            break
        i, j = sorted(rng.integers(0, len(out), 2))
        if j - i < 2:
            continue
        if checker.path_clear(out[i], out[j]):
            out = out[: i + 1] + out[j:]
    return out


def path_cost(path) -> float:
    arr = np.vstack(path)
    return float(np.linalg.norm(np.diff(arr, axis=0), axis=1).sum())
