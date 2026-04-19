import numpy as np
from mesa import Model
from mesa.datacollection import DataCollector

from .PigeonAgent import PigeonAgent

# ---------------------------------------------------------------------------
# Empirical hierarchy data from hf4 leader-follower network
# ---------------------------------------------------------------------------
EMPIRICAL_TAU = {
    "A": +0.30,
    "B": +0.22,
    "H": +0.25,
    "D": +0.10,
    "J": +0.02,
    "L": -0.05,
    "C": -0.23,
    "I": -0.28,
    "G": -0.38,
}

HIERARCHY_ORDER = sorted(EMPIRICAL_TAU, key=lambda k: EMPIRICAL_TAU[k], reverse=True)

# Max centroid trail length (number of steps to show)
MAX_TRAIL = 300


class PigeonFlockModel(Model):
    """
    ABM of a pigeon flock with empirical leader-follower hierarchy.

    Optional homing mode: set home_pos to a (x, y) target and birds will
    be attracted towards it, with leaders feeling the pull more strongly.
    """

    def __init__(
        self,
        space_width=400.0,
        space_height=400.0,
        speed=2.0,
        perception_radius=50.0,
        separation_radius=12.0,
        w_separation=1.8,
        w_alignment=1.0,
        w_cohesion=0.8,
        w_homing=0.6,
        # Start position: top-left area; home: bottom-right area
        start_x=80.0,
        start_y=320.0,
        home_x=320.0,
        home_y=80.0,
        dt=0.2,
        seed=None,
    ):
        super().__init__(seed=seed)

        self.space_width = space_width
        self.space_height = space_height
        self.dt = dt
        self.w_homing = w_homing
        self.num_birds = len(EMPIRICAL_TAU)

        # Home target position
        self.home_pos = np.array([home_x, home_y])
        self.start_pos = np.array([start_x, start_y])

        # Centroid trail: list of (x, y) tuples
        self.centroid_trail = []

        # ------------------------------------------------------------------
        # Initial heading: from start towards home
        # ------------------------------------------------------------------
        to_home = self.home_pos - self.start_pos
        dist = np.linalg.norm(to_home)
        flock_heading = to_home / dist if dist > 0 else np.array([1.0, 0.0])

        for rank, bird_id in enumerate(HIERARCHY_ORDER):
            tau_i = EMPIRICAL_TAU[bird_id]

            forward_offset = (len(HIERARCHY_ORDER) / 2 - rank) * 6.0
            lateral_offset = self.rng.uniform(-10.0, 10.0)

            # Perpendicular to heading for lateral spread
            perp = np.array([-flock_heading[1], flock_heading[0]])

            pos = (
                self.start_pos
                + forward_offset * flock_heading
                + lateral_offset * perp
            )

            angle_noise = self.rng.uniform(-0.15, 0.15)
            cos_a, sin_a = np.cos(angle_noise), np.sin(angle_noise)
            vx = speed * (flock_heading[0] * cos_a - flock_heading[1] * sin_a)
            vy = speed * (flock_heading[0] * sin_a + flock_heading[1] * cos_a)

            PigeonAgent(
                model=self,
                pos=pos.tolist(),
                velocity=[vx, vy],
                bird_id=bird_id,
                tau_i=tau_i,
                speed=speed,
                perception_radius=perception_radius,
                separation_radius=separation_radius,
                w_separation=w_separation,
                w_alignment=w_alignment,
                w_cohesion=w_cohesion,
            )

        # ------------------------------------------------------------------
        # Data collection
        # ------------------------------------------------------------------
        self.datacollector = DataCollector(
            model_reporters={
                "flock_radius": self._flock_radius,
                "centroid_x":   lambda m: float(np.mean([a.pos[0] for a in m.agents])),
                "centroid_y":   lambda m: float(np.mean([a.pos[1] for a in m.agents])),
                "mean_speed":   lambda m: float(np.mean([np.linalg.norm(a.velocity) for a in m.agents])),
            }
        )
        self.datacollector.collect(self)

        # Record initial centroid
        self.centroid_trail.append(tuple(self.get_centroid()))

    # ------------------------------------------------------------------
    # Metric helpers
    # ------------------------------------------------------------------

    def _flock_radius(self):
        positions = np.array([a.pos for a in self.agents])
        centroid = positions.mean(axis=0)
        return float(np.mean(np.linalg.norm(positions - centroid, axis=1)))

    def get_centroid(self):
        positions = np.array([a.pos for a in self.agents])
        return positions.mean(axis=0)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self):
        self.agents.do("compute_new_velocity")
        self.agents.do("apply_move")

        # Record centroid trail
        cx, cy = self.get_centroid()
        self.centroid_trail.append((cx, cy))
        if len(self.centroid_trail) > MAX_TRAIL:
            self.centroid_trail.pop(0)

        self.datacollector.collect(self)
