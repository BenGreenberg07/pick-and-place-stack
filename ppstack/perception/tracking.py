"""Multi-object tracking: constant-velocity Kalman filters plus optimal
assignment.

A single-frame detector has no memory. It cannot tell you that the red block on
this frame is the same red block as on the last one, which means it cannot give
you a velocity, cannot bridge a frame where the block was occluded, and cannot
stop a mis-detection from being treated as a brand new object. Everything above
this layer that wants to reason about a moving world needs those three things.

Two pieces do the work:

* A **Kalman filter** per track over the state [x, y, vx, vy] in table
  millimetres, with a constant-velocity motion model. It supplies the predicted
  position that association matches against, and its innovation covariance
  supplies the *shape* of the gate: an uncertain track is allowed to match
  further away than a confident one, which a fixed distance threshold cannot
  express.
* The **Hungarian algorithm** for association, over Mahalanobis distance rather
  than Euclidean. See `assignment.py` for why greedy matching is not good
  enough here.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from .assignment import hungarian

# Chi-squared 99% quantile with 2 degrees of freedom. A track accepts a
# detection only if the squared Mahalanobis distance falls inside this, so the
# gate automatically widens as the filter's own uncertainty grows.
CHI2_2DOF_99 = 9.21


@dataclass
class Track:
    """One tracked object."""

    id: int
    color: str
    x: np.ndarray                     # [x, y, vx, vy], table mm and mm/s
    P: np.ndarray                     # 4x4 covariance
    hits: int = 1
    misses: int = 0
    age: int = 1
    confirmed: bool = False
    history: list[tuple[float, float]] = field(default_factory=list)

    @property
    def position(self) -> tuple[float, float]:
        return float(self.x[0]), float(self.x[1])

    @property
    def velocity(self) -> tuple[float, float]:
        return float(self.x[2]), float(self.x[3])

    @property
    def speed(self) -> float:
        return float(np.hypot(self.x[2], self.x[3]))


class MultiObjectTracker:
    """Track blocks across frames, in table millimetres.

    Tracks are confirmed after `min_hits` associations and deleted after
    `max_misses` consecutive misses. The delay before confirmation is what keeps
    a one-frame speck from being published as an object, and the tolerance for
    misses is what lets a track survive an occlusion instead of being reborn
    with a new identity on the far side of it.
    """

    def __init__(
        self,
        *,
        dt: float = 1 / 30,
        process_accel_sigma: float = 400.0,   # mm/s^2, how hard a block may accelerate
        measurement_sigma: float = 2.0,       # mm, roughly the detector's error
        min_hits: int = 3,
        max_misses: int = 5,
        match_colors: bool = True,
        associate: str = "hungarian",
    ) -> None:
        self.dt = dt
        self.q_sigma = process_accel_sigma
        self.r_sigma = measurement_sigma
        self.min_hits = min_hits
        self.max_misses = max_misses
        self.match_colors = match_colors
        if associate not in ("hungarian", "greedy"):
            raise ValueError("associate must be 'hungarian' or 'greedy'")
        # "greedy" exists only so the benchmark can show what it costs. It is
        # not an option anyone should choose.
        self.associate = associate
        self.tracks: list[Track] = []
        self._ids = itertools.count(1)
        self.frames = 0

    # ------------------------------------------------------------ the model

    def _F(self, dt: float) -> np.ndarray:
        return np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], float)

    def _Q(self, dt: float) -> np.ndarray:
        """Process noise from a piecewise-constant white acceleration model.

        Writing it out rather than picking diagonal numbers matters: an
        unmodelled acceleration corrupts position and velocity together, so Q
        has real off-diagonal terms. A diagonal Q makes the filter overconfident
        about exactly the thing it is worst at, which is a manoeuvre.
        """
        g = np.array([[dt**2 / 2, 0], [0, dt**2 / 2], [dt, 0], [0, dt]])
        return g @ g.T * self.q_sigma**2

    @property
    def _H(self) -> np.ndarray:
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)

    @property
    def _R(self) -> np.ndarray:
        return np.eye(2) * self.r_sigma**2

    # -------------------------------------------------------------- stepping

    def predict(self, dt: float | None = None) -> None:
        dt = self.dt if dt is None else dt
        F, Q = self._F(dt), self._Q(dt)
        for t in self.tracks:
            t.x = F @ t.x
            t.P = F @ t.P @ F.T + Q
            t.age += 1

    def _gate_cost(self, observations) -> np.ndarray:
        """Squared Mahalanobis distance from every track to every detection.

        Mahalanobis rather than Euclidean is the point. It measures the
        residual in units of the filter's own predicted uncertainty, so a track
        that has just coasted through three missed frames is correctly willing
        to accept a detection 40 mm away, while a well-observed one is not.
        """
        H, R = self._H, self._R
        cost = np.full((len(self.tracks), len(observations)), np.inf)
        for i, t in enumerate(self.tracks):
            S = H @ t.P @ H.T + R
            S_inv = np.linalg.inv(S)
            for j, (colour, pos) in enumerate(observations):
                if self.match_colors and colour != t.color:
                    continue
                y = np.asarray(pos, float) - H @ t.x
                d2 = float(y @ S_inv @ y)
                if d2 <= CHI2_2DOF_99:
                    cost[i, j] = d2
        return cost

    def update(self, observations, dt: float | None = None) -> list[Track]:
        """Advance one frame. `observations` is a list of (colour, (x, y)) in mm.

        Returns the confirmed tracks.
        """
        self.frames += 1
        self.predict(dt)
        observations = list(observations)

        matched_tracks: set[int] = set()
        matched_obs: set[int] = set()
        if self.tracks and observations:
            cost = self._gate_cost(observations)
            pairs = hungarian(cost) if self.associate == "hungarian" else _greedy(cost)
            for i, j in pairs:
                self._correct(self.tracks[i], np.asarray(observations[j][1], float))
                matched_tracks.add(i)
                matched_obs.add(j)

        for i, t in enumerate(self.tracks):
            if i not in matched_tracks:
                t.misses += 1
            t.history.append(t.position)

        for j, (colour, pos) in enumerate(observations):
            if j not in matched_obs:
                self.tracks.append(self._spawn(colour, np.asarray(pos, float)))

        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]
        return self.confirmed_tracks()

    def _correct(self, t: Track, z: np.ndarray) -> None:
        H, R = self._H, self._R
        y = z - H @ t.x
        S = H @ t.P @ H.T + R
        K = t.P @ H.T @ np.linalg.inv(S)
        t.x = t.x + K @ y
        # Joseph form: stays symmetric and positive definite under round-off,
        # where the textbook (I - KH)P does not and eventually breaks the gate.
        I_KH = np.eye(4) - K @ H
        t.P = I_KH @ t.P @ I_KH.T + K @ R @ K.T
        t.hits += 1
        t.misses = 0
        if t.hits >= self.min_hits:
            t.confirmed = True

    def _spawn(self, colour: str, pos: np.ndarray) -> Track:
        P = np.diag([self.r_sigma**2, self.r_sigma**2, 500.0**2, 500.0**2])
        return Track(
            id=next(self._ids),
            color=colour,
            x=np.array([pos[0], pos[1], 0.0, 0.0]),
            P=P,
        )

    def confirmed_tracks(self) -> list[Track]:
        """Confirmed tracks, including ones currently coasting through a miss.

        Publishing a coasting track is the entire point of having a motion
        model. Dropping it the moment the detector blinks would hand the
        consumer a fresh identity on the far side of every occlusion, which is
        the failure the tracker exists to prevent.
        """
        return [t for t in self.tracks if t.confirmed]

    def visible_tracks(self) -> list[Track]:
        """Only the tracks backed by a detection on this frame."""
        return [t for t in self.tracks if t.confirmed and t.misses == 0]

    def all_tracks(self) -> list[Track]:
        return list(self.tracks)


def _greedy(cost: np.ndarray) -> list[tuple[int, int]]:
    """Repeatedly take the globally cheapest remaining pair.

    The failure mode is structural, not a tuning problem: committing to the
    single best pair first can force a later track onto a match that the
    optimal joint assignment would have avoided. Two objects passing close to
    each other is exactly that situation, and the result is an identity swap.
    """
    pairs: list[tuple[int, int]] = []
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    order = np.dstack(np.unravel_index(np.argsort(cost, axis=None), cost.shape))[0]
    for i, j in order:
        i, j = int(i), int(j)
        if not np.isfinite(cost[i, j]):
            break
        if i in used_rows or j in used_cols:
            continue
        pairs.append((i, j))
        used_rows.add(i)
        used_cols.add(j)
    return sorted(pairs)


def count_id_switches(assignments: list[dict[str, int]]) -> int:
    """Identity switches across a sequence.

    `assignments` is one dict per frame mapping a ground-truth object name to
    the track id that claimed it. A switch is any frame where a truth object
    that already had an id is now carried by a different one. This is the
    headline number in the MOT literature for a reason: a tracker can have
    excellent position error and still be useless if it keeps swapping which
    object is which.
    """
    last: dict[str, int] = {}
    switches = 0
    for frame in assignments:
        for name, tid in frame.items():
            if name in last and last[name] != tid:
                switches += 1
            last[name] = tid
    return switches
