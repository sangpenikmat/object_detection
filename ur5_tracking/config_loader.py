"""Load and validate config.yaml (the single source of truth)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import yaml


class ConfigError(Exception):
    pass


@dataclass
class Config:
    raw: Dict[str, Any]
    root_dir: str  # directory of config.yaml, for resolving relative paths

    # -- safe nested access ---------------------------------------------------
    def get(self, *keys, default=None):
        node: Any = self.raw
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node

    def require(self, *keys):
        sentinel = object()
        val = self.get(*keys, default=sentinel)
        if val is sentinel:
            raise ConfigError(f"Missing required config key: {'/'.join(keys)}")
        return val

    def arr(self, *keys) -> np.ndarray:
        return np.asarray(self.require(*keys), dtype=float)

    # -- helpers --------------------------------------------------------------
    def abspath(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.normpath(os.path.join(self.root_dir, path))

    @property
    def menagerie_ur5e_dir(self) -> str:
        return self.abspath(self.require("robot", "menagerie_ur5e_dir"))

    @property
    def menagerie_2f85_dir(self) -> str:
        return self.abspath(self.require("gripper", "menagerie_2f85_dir"))

    @property
    def menagerie_d435i_dir(self) -> str | None:
        path = self.get("camera", "menagerie_d435i_dir", default=None)
        return self.abspath(path) if path is not None else None

    @property
    def menagerie_adapter_dir(self) -> str | None:
        path = self.get("camera", "menagerie_adapter_dir", default=None)
        return self.abspath(path) if path is not None else None

    @property
    def joint_names(self) -> List[str]:
        return list(self.require("robot", "joint_names"))

    @property
    def n_joints(self) -> int:
        return len(self.joint_names)


def load_config(path: str) -> Config:
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise ConfigError(f"config.yaml not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ConfigError("config.yaml is empty or malformed.")
    cfg = Config(raw=raw, root_dir=os.path.dirname(path))
    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    if cfg.n_joints != 6:
        raise ConfigError(f"Expected 6 joints, got {cfg.n_joints}.")

    for key in ("joint_limits_lower", "joint_limits_upper"):
        v = cfg.arr("robot", key)
        if v.shape != (6,):
            raise ConfigError(f"robot.{key} must have length 6, got {v.shape}.")
    if np.any(cfg.arr("robot", "joint_limits_lower") >= cfg.arr("robot", "joint_limits_upper")):
        raise ConfigError("joint_limits_lower must be < upper for every joint.")

    if cfg.arr("platform", "size").shape != (3,):
        raise ConfigError("platform.size must have 3 values.")

    for sect in ("object", "work_surface", "tracking", "pick", "home_return"):
        if cfg.get(sect) is None:
            raise ConfigError(f"Missing required config section: {sect}")

    if cfg.arr("object", "spawn").shape != (3,) or cfg.arr("object", "size").shape != (3,):
        raise ConfigError("object.spawn and object.size must each have 3 values.")
    if len(cfg.arr("tracking", "perch_joints")) != 6:
        raise ConfigError("tracking.perch_joints must have 6 values.")


if __name__ == "__main__":
    import sys
    c = load_config(sys.argv[1] if len(sys.argv) > 1 else "config.yaml")
    print("Config OK.")
    print("  ur5e dir :", c.menagerie_ur5e_dir)
    print("  2f85 dir :", c.menagerie_2f85_dir)
    print("  obstacles:", [o["name"] for o in c.get("obstacles", default=[])])
