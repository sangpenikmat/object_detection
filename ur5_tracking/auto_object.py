"""Automatic object driver (used by --auto mode).

Kinematically moves the object across the work surface to a random target, then
leaves it stationary, which makes the FSM detect "left" and retrieve it. Only
active while the FSM is in TRACK (i.e. the object is not being held). After the
object is returned home and tracking resumes, the next cycle can start.
"""
from __future__ import annotations

import numpy as np


class AutoObjectDriver:
    def __init__(self, ctl, cfg, rng=None):
        self.ctl = ctl
        self.rng = rng or np.random.default_rng(0)
        ws = cfg.get("work_surface")
        top_z = ws["center"][2] + ws["size"][2] / 2.0
        obj_half = float(np.asarray(cfg.get("object", "size"), float)[2]) / 2.0
        self.rest_z = top_z + obj_half
        c = np.asarray(ws["center"], float)[:2]
        h = np.asarray(ws["size"], float)[:2] / 2.0 - 0.06   # margin from the edge
        self.lo, self.hi = c - h, c + h
        self.home_xy = cfg.arr("home_return", "position")[:2]
        self.speed = float(cfg.get("tracking", "move_speed", default=0.05)) * 2.5

        self._target = None
        self._moving = False
        self._cooldown_until = 0.0

    def _new_target(self):
        return self.rng.uniform(self.lo, self.hi)

    def update(self, d, fsm_state, t, dt):
        # hand control back to the FSM while the object is being retrieved
        if fsm_state != "TRACK":
            self._moving = False
            self._cooldown_until = t + 1.0
            return

        if not self._moving and t > self._cooldown_until:
            tgt = self._new_target()
            while np.linalg.norm(tgt - self.home_xy) < 0.12:   # ensure it triggers a retrieve
                tgt = self._new_target()
            self._target, self._moving = tgt, True

        if not self._moving:
            return

        ctl = self.ctl
        cur = ctl.object_pos(d)[:2]
        delta = self._target - cur
        dist = np.linalg.norm(delta)
        if dist < 0.01:
            # arrived -> leave it (stationary) so the FSM detects "left"
            self._moving = False
            self._cooldown_until = t + 8.0
            ctl.set_object_pose(d, [self._target[0], self._target[1], self.rest_z], vel=np.zeros(3))
            return
        direction = delta / dist
        step = min(self.speed * dt, dist)
        nxt = cur + direction * step
        ctl.set_object_pose(d, [nxt[0], nxt[1], self.rest_z],
                            vel=[direction[0] * self.speed, direction[1] * self.speed, 0.0])
