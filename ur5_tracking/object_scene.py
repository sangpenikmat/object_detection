"""Build the MuJoCo scene for the track / pick / return task.

Starts from the UR5e model and adds:
  - a Robotiq 2F-85 gripper (attached to the flange),
  - a RealSense D435i depth camera (attached to the gripper, visual only),
  - a free-floating object the user can drag (or a script can move),
  - a camera mounted on the gripper, aligned with the tool approach axis,
  - a WELD equality (object <-> gripper) enabled while grasping (stable hold),
  - visual polish: skybox, checkered floor, materials, lights, shadows, and a
    sensible default viewing angle plus an "overview" camera.

Built with MjSpec so the UR5e, 2F-85, and D435i assets resolve automatically.

Note: box sizes in config are FULL extents; MuJoCo uses half-extents, so we
divide by 2 here.
"""
from __future__ import annotations

import os
from typing import Tuple

import mujoco
import numpy as np

from .config_loader import Config, ConfigError

_RGB_ROLE = int(mujoco.mjtTextureRole.mjTEXROLE_RGB)
_NROLE = int(mujoco.mjtTextureRole.mjNTEXROLE)


def _link_texture(material, tex_name: str) -> None:
    roles = [""] * _NROLE
    roles[_RGB_ROLE] = tex_name
    material.textures = roles


def _add_visuals(spec, cfg: Config) -> None:
    """Skybox, ground, materials, lights and default camera framing."""
    # --- skybox + checkered ground -------------------------------------------
    spec.add_texture(name="skybox", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.32, 0.45, 0.62], rgb2=[0.04, 0.07, 0.12],
                     width=512, height=512)
    spec.add_texture(name="grid_tex", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.24, 0.27, 0.31], rgb2=[0.17, 0.19, 0.23],
                     mark=mujoco.mjtMark.mjMARK_EDGE, markrgb=[0.45, 0.48, 0.52],
                     width=300, height=300)
    grid = spec.add_material(name="grid", texrepeat=[8, 8], reflectance=0.12, shininess=0.1)
    _link_texture(grid, "grid_tex")

    # --- surface / object materials ------------------------------------------
    table = spec.add_material(name="table_mat", rgba=[0.62, 0.46, 0.32, 1.0],
                              reflectance=0.05, shininess=0.2, specular=0.2)
    spec.add_material(name="platform_mat", rgba=[0.55, 0.57, 0.60, 1.0],
                      reflectance=0.35, shininess=0.5, specular=0.5, metallic=0.6)
    spec.add_material(name="object_mat", rgba=cfg.get("object", "rgba",
                      default=[0.15, 0.55, 0.95, 1.0]),
                      reflectance=0.2, shininess=0.6, specular=0.6)
    spec.add_material(name="obstacle_mat", rgba=[0.80, 0.42, 0.30, 0.55],
                      reflectance=0.05, shininess=0.2)
    _ = table

    # --- lighting ------------------------------------------------------------
    hl = spec.visual.headlight
    hl.ambient = [0.42, 0.42, 0.42]
    hl.diffuse = [0.45, 0.45, 0.45]
    hl.specular = [0.12, 0.12, 0.12]

    spec.worldbody.add_light(pos=[0.6, -0.8, 2.2], dir=[-0.3, 0.4, -1.0],
                             type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                             castshadow=True, diffuse=[0.7, 0.7, 0.68],
                             specular=[0.3, 0.3, 0.3])
    spec.worldbody.add_light(pos=[-0.8, 0.6, 1.8], dir=[0.4, -0.3, -1.0],
                             type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                             castshadow=False, diffuse=[0.3, 0.3, 0.35])

    # --- global look + default framing ---------------------------------------
    g = spec.visual.global_
    g.azimuth = 140
    g.elevation = -22
    g.offwidth = 1280
    g.offheight = 960
    spec.visual.map.haze = 0.0
    spec.stat.center = [-0.1, 0.25, 0.25]
    spec.stat.extent = 1.15

    # overview camera that keeps the object framed (press Tab / [ ] to switch).
    # Defined later, after the object body exists.


def _add_environment(spec, cfg: Config) -> None:
    """Add floor, main platform and the static obstacles."""
    wb = spec.worldbody
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                size=[0, 0, 0.05], pos=[0, 0, float(cfg.get("floor_z", default=-0.80))],
                material="grid", contype=1, conaffinity=1, group=0)
    wb.add_geom(name="platform", type=mujoco.mjtGeom.mjGEOM_BOX,
                size=(cfg.arr("platform", "size") / 2.0).tolist(),
                pos=cfg.arr("platform", "center").tolist(),
                material="platform_mat", contype=1, conaffinity=1, group=0)
    for o in cfg.get("obstacles", default=[]):
        wb.add_geom(name=f"obstacle_{o['name']}", type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=(np.asarray(o["size"], float) / 2.0).tolist(),
                    pos=np.asarray(o["center"], float).tolist(),
                    material="obstacle_mat", contype=1, conaffinity=1, group=0)

    ws = cfg.get("work_surface")
    if ws is not None:
        wb.add_geom(name="work_surface", type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=(np.asarray(ws["size"], float) / 2.0).tolist(),
                    pos=np.asarray(ws["center"], float).tolist(),
                    material="table_mat", contype=1, conaffinity=1, group=0)


def build_scene(cfg: Config) -> "mujoco.MjSpec":
    """Assemble and return the full MjSpec (uncompiled)."""
    ur5e_xml = os.path.join(cfg.menagerie_ur5e_dir, "ur5e.xml")
    grip_xml = os.path.join(cfg.menagerie_2f85_dir, "2f85.xml")
    for p in (ur5e_xml, grip_xml):
        if not os.path.exists(p):
            raise ConfigError(f"Model file not found: {p}")

    spec = mujoco.MjSpec.from_file(ur5e_xml)
    grip = mujoco.MjSpec.from_file(grip_xml)

    spec.body("base").pos = cfg.arr("base", "position")

    _add_visuals(spec, cfg)
    _add_environment(spec, cfg)

    # --- free-floating object ------------------------------------------------
    osz = cfg.arr("object", "size")
    obj = spec.worldbody.add_body(name="object", pos=cfg.arr("object", "spawn").tolist())
    obj.add_freejoint(name="object_free")
    obj.add_geom(name="object_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
                 size=(osz / 2.0).tolist(), material="object_mat",
                 mass=float(cfg.get("object", "mass", default=0.2)),
                 friction=[1.0, 0.02, 0.001], contype=1, conaffinity=1, group=1)

    # --- visual marker for the home position (flat pad on the table top) ------
    home = cfg.arr("home_return", "position")
    ws = cfg.get("work_surface")
    pad_z = (ws["center"][2] + ws["size"][2] / 2.0 + 0.002) if ws else float(home[2])
    spec.worldbody.add_site(name="home_marker", pos=[float(home[0]), float(home[1]), pad_z],
                            type=mujoco.mjtGeom.mjGEOM_BOX,
                            size=[float(osz[0] * 0.7), float(osz[1] * 0.7), 0.0015],
                            rgba=[0.2, 0.95, 0.4, 0.55], group=2)

    # --- PickNik adapter bracket + gripper + D435i (eye-in-hand) -------------
    # Kinematic chain: UR5e flange → adapter bracket → gripper  (at bracket tool0, +7 mm)
    #                                                → D435i    (at bracket camera_mount)
    # Falls back to direct flange attach if the adapter dir is absent.
    adapter_dir = cfg.menagerie_adapter_dir
    adapter_xml = os.path.join(adapter_dir, "adapter.xml") if adapter_dir else None
    d435i_dir = cfg.menagerie_d435i_dir
    d435i_xml = os.path.join(d435i_dir, "d435i.xml") if d435i_dir else None
    ee_site_name = cfg.get("robot", "ee_site", default="attachment_site")

    if adapter_xml and os.path.exists(adapter_xml):
        adapter_spec = mujoco.MjSpec.from_file(adapter_xml)
        spec.site(ee_site_name).attach_body(adapter_spec.body("adapter"), "adapter_", "")
        # gripper hangs off bracket's tool0 site (7 mm further along tool axis)
        spec.site("adapter_tool0").attach_body(grip.body("base_mount"), "2f85_", "")
        # camera hangs off bracket's camera_mount site (side-mounted at URDF angles)
        if d435i_xml and os.path.exists(d435i_xml):
            d435i_spec = mujoco.MjSpec.from_file(d435i_xml)
            spec.site("adapter_camera_mount").attach_body(d435i_spec.body("d435i"), "d435i_", "")
            # D435i optical axis is +Z in body frame; MuJoCo cameras look along
            # -Z, so 180° around X flips the view to look along +Z (workspace).
            spec.body("d435i_d435i").add_camera(
                name="gripper_cam", pos=[0.0, 0.0, 0.0],
                quat=[0, 1, 0, 0],
                fovy=87, mode=mujoco.mjtCamLight.mjCAMLIGHT_FIXED)
        else:
            spec.body("adapter_adapter").add_camera(
                name="gripper_cam",
                pos=[0.0, -0.067, 0.0171],
                quat=[0.5255, 0.4732, -0.4732, 0.5255],
                fovy=87, mode=mujoco.mjtCamLight.mjCAMLIGHT_FIXED)
    else:
        # no bracket — attach gripper directly to flange
        spec.site(ee_site_name).attach_body(grip.body("base_mount"), "2f85_", "")
        if d435i_xml and os.path.exists(d435i_xml):
            d435i_spec = mujoco.MjSpec.from_file(d435i_xml)
            mount = spec.body("2f85_base").add_site(
                name="d435i_mount", pos=[0.0, 0.0, 0.05],
                quat=[0.7071068, 0.0, 0.7071068, 0.0])
            mount.attach_body(d435i_spec.body("d435i"), "d435i_", "")
            spec.body("d435i_d435i").add_camera(
                name="gripper_cam", pos=[0.0, 0.0, 0.0], quat=[0, 1, 0, 0],
                fovy=87, mode=mujoco.mjtCamLight.mjCAMLIGHT_FIXED)
        else:
            spec.body("2f85_base").add_camera(
                name="gripper_cam", pos=[0.05, 0.0, 0.02], quat=[0, 1, 0, 0],
                fovy=58, mode=mujoco.mjtCamLight.mjCAMLIGHT_FIXED)

    # overview camera that always points at the object (auto-frames the action)
    spec.worldbody.add_camera(name="overview", pos=[0.95, -0.85, 0.75],
                              mode=mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY,
                              targetbody="object", fovy=50)

    # --- grasp weld (object <-> gripper), disabled by default ----------------
    eq = spec.add_equality()
    eq.type = mujoco.mjtEq.mjEQ_WELD
    eq.name1 = "2f85_base"
    eq.name2 = "object"
    eq.objtype = mujoco.mjtObj.mjOBJ_BODY
    eq.active = False
    eq.name = "grasp_weld"
    # weld data: [anchor(3), relpose pos(3), relpose quat(4), torquescale(1)]
    eq.data = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1], dtype=float)

    return spec


def build_model(cfg: Config) -> Tuple["mujoco.MjModel", "mujoco.MjSpec"]:
    spec = build_scene(cfg)
    return spec.compile(), spec


if __name__ == "__main__":
    import sys
    from .config_loader import load_config
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "config.yaml")
    m, _ = build_model(cfg)
    print(f"Scene OK: nq={m.nq} nu={m.nu} ngeom={m.ngeom} nlight={m.nlight} "
          f"ntex={m.ntex} nmat={m.nmat} ncam={m.ncam}")
