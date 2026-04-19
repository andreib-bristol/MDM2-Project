import numpy as np
from mesa import Agent


class PigeonAgent(Agent):
    """
    A single pigeon agent in the flocking model.

    Movement is based on Reynolds' three rules (separation, alignment, cohesion)
    with a hierarchy layer derived from empirical tau_ij delay data, plus an
    optional homing force scaled by each bird's rank in the hierarchy.
    """

    def __init__(
        self,
        model,
        pos,
        velocity,
        bird_id,
        tau_i,
        speed=2.0,
        perception_radius=50.0,
        separation_radius=10.0,
        w_separation=1.8,
        w_alignment=1.0,
        w_cohesion=0.8,
        w_inertia=1.2,
    ):
        super().__init__(model)
        self.pos = np.array(pos, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.bird_id = bird_id
        self.tau_i = tau_i
        self.speed = speed
        self.perception_radius = perception_radius
        self.separation_radius = separation_radius
        self.w_separation = w_separation
        self.w_alignment = w_alignment
        self.w_cohesion = w_cohesion
        # Leaders are harder to steer
        self.w_inertia = w_inertia + max(0.0, tau_i) * 3.0

        self._velocity_history = [np.array(velocity, dtype=float)]
        self._new_velocity = np.array(velocity, dtype=float)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _wrap(self, pos):
        W, H = self.model.space_width, self.model.space_height
        return np.array([pos[0] % W, pos[1] % H])

    def _neighbours(self):
        nearby = []
        for other in self.model.agents:
            if other is self:
                continue
            diff = other.pos - self.pos
            W, H = self.model.space_width, self.model.space_height
            diff[0] = (diff[0] + W / 2) % W - W / 2
            diff[1] = (diff[1] + H / 2) % H - H / 2
            if np.linalg.norm(diff) < self.perception_radius:
                nearby.append((other, diff))
        return nearby

    def _delayed_velocity(self, agent):
        delay_steps = max(0, round((agent.tau_i - self.tau_i) / self.model.dt))
        hist = agent._velocity_history
        idx = max(0, len(hist) - 1 - delay_steps)
        return hist[idx]

    # ------------------------------------------------------------------
    # Reynolds rules
    # ------------------------------------------------------------------

    def _separation(self, neighbours):
        steer = np.zeros(2)
        for other, diff in neighbours:
            dist = np.linalg.norm(diff)
            if dist < self.separation_radius and dist > 0:
                steer -= diff / dist
        return steer

    def _alignment(self, neighbours):
        if not neighbours:
            return np.zeros(2)
        weighted_vel = np.zeros(2)
        total_weight = 0.0
        for other, _ in neighbours:
            w = max(0.1, other.tau_i - self.tau_i + 0.5)
            delayed_v = self._delayed_velocity(other)
            norm = np.linalg.norm(delayed_v)
            if norm > 0:
                weighted_vel += w * delayed_v / norm
            total_weight += w
        if total_weight > 0:
            weighted_vel /= total_weight
        return weighted_vel

    def _cohesion(self, neighbours):
        if not neighbours:
            return np.zeros(2)
        centre = np.mean([self.pos + diff for _, diff in neighbours], axis=0)
        steer = centre - self.pos
        norm = np.linalg.norm(steer)
        return steer / norm if norm > 0 else steer

    def _homing(self):
        """
        Pull towards home_pos, scaled by hierarchy rank.
        Leaders (tau_i ~ +0.3) get full weight.
        Followers (tau_i ~ -0.4) get ~20% — they navigate via the flock.
        """
        if self.model.home_pos is None:
            return np.zeros(2)
        to_home = self.model.home_pos - self.pos
        dist = np.linalg.norm(to_home)
        if dist < 5.0:
            return np.zeros(2)
        direction = to_home / dist
        # Map tau_i [-0.4, 0.35] -> scale [0.2, 1.0]
        scale = 0.2 + 0.8 * np.clip((self.tau_i + 0.4) / 0.75, 0.0, 1.0)
        return scale * direction

    # ------------------------------------------------------------------
    # Mesa step interface
    # ------------------------------------------------------------------

    def compute_new_velocity(self):
        """Phase 1: compute desired velocity without moving yet."""
        neighbours = self._neighbours()

        sep = self._separation(neighbours)
        ali = self._alignment(neighbours)
        coh = self._cohesion(neighbours)
        hom = self._homing()
        inertia = self.velocity / (np.linalg.norm(self.velocity) + 1e-9)

        desired = (
            self.w_separation * sep
            + self.w_alignment * ali
            + self.w_cohesion * coh
            + self.model.w_homing * hom
            + self.w_inertia * inertia
        )

        norm = np.linalg.norm(desired)
        if norm > 0:
            self._new_velocity = self.speed * desired / norm
        else:
            self._new_velocity = self.velocity.copy()

    def apply_move(self):
        """Phase 2: commit velocity, update position and history buffer."""
        self.velocity = self._new_velocity.copy()
        self.pos = self._wrap(self.pos + self.velocity * self.model.dt)

        self._velocity_history.append(self.velocity.copy())
        max_hist = int(3.0 / self.model.dt) + 1
        if len(self._velocity_history) > max_hist:
            self._velocity_history.pop(0)
