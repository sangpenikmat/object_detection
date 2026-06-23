# UR5e Object Tracking & Retrieval (MuJoCo)

A UR5e arm with a Robotiq 2F-85 gripper **visually tracks a movable object** —
the gripper-mounted camera follows it like a snake's head. When the object is
**left stationary**, the arm **picks it up and returns it to its original
position**, then resumes tracking.

This is a **closed-loop simulation**: every control step reads the object pose
from the simulator's ground truth. (It is not an open-loop trajectory for a real
robot — that would require real perception, e.g. a camera + object detector.)

## What you get

```
object detection/
├─ config.yaml                 # single source of truth (robot, env, object, behaviour)
├─ track_and_retrieve.py       # main entry point (interactive / --auto / --headless)
├─ requirements.txt
├─ mujoco_menagerie/
│  ├─ universal_robots_ur5e/   # UR5e model (bundled)
│  └─ robotiq_2f85/            # gripper model (bundled)
└─ ur5_tracking/               # the package
   ├─ config_loader.py         # load + validate config.yaml
   ├─ object_scene.py          # build scene: UR5e + gripper + object + camera + grasp weld
   ├─ track_control.py         # state readouts, aim/reach IK, grasp (weld)
   ├─ track_states.py          # finite state machine: track -> pick -> return
   └─ auto_object.py           # scripted object motion for --auto mode
```

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

(Only `mujoco`, `numpy`, `pyyaml` are required; `imageio` is optional, for `--record`.)

## Run

```powershell
# Interactive: drag the blue object with Ctrl + right-mouse-drag in the viewer.
python track_and_retrieve.py

# The object moves on its own, then is left; the arm retrieves it. Repeats.
python track_and_retrieve.py --auto

# No display (quick verification): prints state transitions + placement accuracy.
python track_and_retrieve.py --headless --seconds 30

# Record the gripper camera while running headless.
python track_and_retrieve.py --headless --seconds 30 --record run.mp4
```

In the interactive viewer, drag the object around — the arm aims at it. Let go;
once it is stationary for ~1.5 s and away from home, the arm picks it up and
places it back on the green marker.

## Cameras & viewing

The scene ships with a skybox, a checkered floor, a wooden work table, metal /
glossy materials, and soft shadows, with a sensible default camera angle.

Two named cameras are available; press **Tab** (or `[` / `]`) in the viewer to
cycle:
- **overview** — auto-frames the object (stays pointed at it as it moves).
- **gripper_cam** — the gripper's-eye view, looking along the tool approach axis.

You can still orbit/pan/zoom the free camera with the mouse at any time.

## How it works

State machine (`ur5_tracking/track_states.py`):

```
TRACK -> APPROACH -> DESCEND -> GRASP -> LIFT -> CARRY -> PLACE -> RELEASE -> RETREAT -> TRACK
```

- **TRACK (gaze):** the arm holds the "perch" pose and only re-orients so the
  tool +z axis (and the camera) points at the object. It watches the object's
  speed; if it stays below `tracking.idle_speed` for `tracking.idle_time` and is
  not already home, it starts a retrieve.
- **GRASP / RELEASE:** the gripper closes/opens and a weld constraint is
  enabled/disabled.
- Each motion phase advances when the target is reached, when progress stalls,
  or after an 8 s safety timeout — so it never deadlocks.

All tunable numbers live in `config.yaml`: `robot`, `gripper`, environment
(`platform`, `obstacles`, `work_surface`, `floor_z`), `object`, `home_return`,
`tracking`, and `pick`.

## Design notes (honest)

1. **The grasp is a weld, not pure friction.** The 2F-85 visibly closes, but the
   hold is guaranteed by a MuJoCo weld equality enabled at grasp time. Pure
   contact grasping in MuJoCo is fragile (slip, closing force, friction tuning);
   the weld keeps the demo reliable. For realistic contact grasping, disable the
   weld in `track_control.set_grasp` and tune the gripper friction/force.

2. **A "work surface" was added.** The main platform sits under the base and is
   too narrow in +y, so an object placed in the workspace would fall. A static
   surface in front of the robot (within reach) gives the object somewhere to
   rest. Edit `work_surface` in the config for your real setup.

3. **Snake-like tracking = gaze control.** The arm keeps its perch position and
   only changes orientation to keep the camera pointed at the object, so the
   "head" follows while the body stays roughly put.

## Verification

`--headless` reports the state transitions, the number of completed retrieve
cycles, and the placement error at each release. In a 30 s `--auto` run, four
cycles complete with placement errors of ~2-5 mm (mean ~3 mm). The
`preview_*.png` images show the rendered scene.
