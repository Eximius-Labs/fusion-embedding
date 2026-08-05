"""Before/after validation of the PDEEPoseController rotation-scaling bug on the Unitree G1.

Context: ManiSkill issue #1469 (rotation actions scaled by rot_lower, which is negative, so the
commanded rotation direction is inverted), fix PR #1472 (scale by rot_upper), and issue #1435
(Lrogic's G1 right-arm EE-pose control "moves very slightly then freezes" whenever a waypoint
has a rotation tolerance; position-only waypoints work).

This script reproduces Lrogic's setup as faithfully as their posted snippets allow:
  * their custom_g1.py agent (right arm = torso_joint + 5 right-arm joints, right-hand fingers
    on PD joint position, everything else passive, balance_passive_force=True,
    ee_link right_tcp_link),
  * their control loop: each step the action is the clipped EE pose error in the base frame,
    rotation error = euler_XYZ(q_target * q_current^-1), norm-clipped,
  * CPU sim (physx_cpu), num_envs=1, as in their report.

Three controller variants of pd_ee_delta_pose are tested (their attachment fully specifies the
delta-pose config as normalize_action=False, which bypasses _clip_and_scale_action, so the two
normalized variants document the bug class for the default/recommended setting):
  * lrogic_exact             pos +-0.1, rot +-0.1, normalize_action=False (their posted config)
  * normalized_stock_bounds  pos +-0.1, rot +-0.1, normalize_action=True (ManiSkill default,
                             same bounds as the stock Panda/XArm6 pd_ee_delta_pose configs)
  * normalized_default_bounds pos +-0.1, rot bounds left at the PDEEPoseControllerConfig
                             defaults (-2*pi, 2*pi), normalize_action=True

Conditions: "before" = stock ManiSkill main at the pinned commit; "after" = the same tree with
the one-line fix applied programmatically (rot_lower -> rot_upper in
mani_skill/agents/controllers/pd_ee_pose.py, exactly PR #1472's change). Each condition runs in
its own subprocess so the patched module is imported fresh.

Per variant x condition:
  * closed-loop tracking of a +30 deg orientation offset about each of x, y, z (200 steps,
    per-step orientation error logged, IK failure count from the compute_ik wrapper),
  * a position-only target as control,
  * a one-step open-loop check: +1 normalized rotation action about z -> signed realized
    target delta (expected +rot_upper after the fix, rot_lower before).

Run (from repo root):
    PYTHONUTF8=1 uv run --env-file .env modal run scripts/maniskill_g1_rot_validation.py --action smoke
    PYTHONUTF8=1 uv run --env-file .env modal run scripts/maniskill_g1_rot_validation.py --action run

Results (JSON + PNG error curves + printed table) are written locally under
scratchpad/maniskill_g1_validation/ (override with MSK_VAL_OUT).
"""

import base64
import json
import os
import pathlib

import modal

# Upstream main at validation time (the commit our local fix branch is based on).
MANISKILL_COMMIT = "62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3"
# unitree_ros master at follow-up time; provides g1_29dof_rev_1_0.urdf (7-DOF arms
# with wrist roll/pitch/yaw), BSD-3-Clause. ManiSkill itself ships no wrist variant.
UNITREE_ROS_COMMIT = "f3772ce54c56ef2d34c6aee8100bc768896c7d19"

BUG_LINE = "rot_action = rot_action * self.config.rot_lower"
FIX_LINE = "rot_action = rot_action * self.config.rot_upper"

DEFAULT_OUT_DIR = (
    r"C:\Users\abdul\AppData\Local\Temp\claude\D--eximius-labs-fusion-embeddings"
    r"\82087ead-dcc4-4f92-8683-f1ad15f2e644\scratchpad\maniskill_g1_validation"
)

app = modal.App("maniskill-g1-rot-validation")

img = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git", "libgl1", "libglib2.0-0", "libvulkan1", "libx11-6",
        "libxext6", "libegl1",
        # software Vulkan device (lavapipe): sapien's URDF loader instantiates
        # RenderMaterial even with render_backend="none" and needs a device
        "mesa-vulkan-drivers",
    )
    .pip_install("numpy==1.26.4", "matplotlib")
    .pip_install("torch==2.5.1", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install(f"git+https://github.com/haosulab/ManiSkill.git@{MANISKILL_COMMIT}")
    # Expose the NVIDIA driver's Vulkan ICD so SAPIEN finds a rendering device
    # in the headless container (standard Modal + Vulkan recipe).
    .env({"NVIDIA_DRIVER_CAPABILITIES": "all"})
    .run_commands(
        "mkdir -p /usr/share/vulkan/icd.d && printf '%s' "
        '\'{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.3.242"}}\' '
        "> /usr/share/vulkan/icd.d/nvidia_icd.json"
    )
    # Wrist-DOF G1 (29-DOF, 7-DOF arms) from Unitree's official description repo,
    # pinned; sparse checkout of just robots/g1_description.
    .run_commands(
        "git clone --depth 1 --filter=blob:none --sparse "
        "https://github.com/unitreerobotics/unitree_ros /tmp/unitree_ros "
        "&& cd /tmp/unitree_ros && git sparse-checkout set robots/g1_description "
        f"&& test \"$(git rev-parse HEAD)\" = {UNITREE_ROS_COMMIT} "
        "&& mv robots/g1_description /root/g1_description && rm -rf /tmp/unitree_ros"
    )
)

# ---------------------------------------------------------------------------
# Worker: runs in a fresh subprocess per condition so the (possibly patched)
# mani_skill module is imported cleanly.
# ---------------------------------------------------------------------------
WORKER_SRC = r'''
"""Worker: G1 EE-pose rotation-tracking experiment for one before/after condition."""
import argparse
import json
import sys

import numpy as np
import torch
import sapien
import gymnasium as gym

import mani_skill.envs  # noqa: F401  (registers Empty-v1)
from mani_skill.agents.controllers import (
    PDEEPoseControllerConfig,
    PDJointPosControllerConfig,
    PassiveControllerConfig,
    deepcopy_dict,
)
from mani_skill.agents.registration import register_agent
from mani_skill.agents.robots.unitree_g1.g1_upper_body import UnitreeG1UpperBody
from mani_skill.utils.geometry.rotation_conversions import (
    matrix_to_euler_angles,
    quaternion_invert,
    quaternion_multiply,
    quaternion_to_matrix,
)

POS_BOUND = 0.1
ROT_BOUND_SMALL = 0.1
ROT_BOUND_DEFAULT = float(2 * np.pi)
# Per-step command caps, the analog of Lrogic's delta_steps/_ee_delta_step_limits:
# their loop limits the per-step error step below the controller bound. Without
# these the exact-pose CPU IK is asked for single-step jumps (e.g. 8.7 cm, or 30
# deg with the +-2*pi bounds) and fails every step in BOTH conditions.
POS_STEP = 0.02   # m per control step
ROT_STEP = 0.1    # rad per control step
TARGET_DEG = 30.0
POS_TOL = 0.03      # Lrogic's pos_tol
ROT_TOL = 0.5       # Lrogic's rot_tol (radians)
STABLE_STEPS = 10   # Lrogic's stable_steps


@register_agent()
class ValidationG1(UnitreeG1UpperBody):
    """Mirror of Lrogic's custom_g1.py (issue #1435 attachment) with three
    pd_ee_delta_pose variants registered as separate control modes."""

    uid = "g1_rot_validation"
    ee_link_name = "right_tcp_link"

    right_arm_joint_names = [
        "torso_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_pitch_joint",
        "right_elbow_roll_joint",
    ]
    right_finger_joint_names = [
        "right_zero_joint",
        "right_three_joint",
        "right_five_joint",
        "right_one_joint",
        "right_four_joint",
        "right_six_joint",
        "right_two_joint",
    ]

    @property
    def _passive_joint_names(self):
        controlled = set(self.right_arm_joint_names + self.right_finger_joint_names)
        return [j for j in self.body_joints if j not in controlled]

    def _arm_cfg(self, rot_lower, rot_upper, normalize_action):
        return PDEEPoseControllerConfig(
            joint_names=self.right_arm_joint_names,
            pos_lower=-POS_BOUND,
            pos_upper=POS_BOUND,
            rot_lower=rot_lower,
            rot_upper=rot_upper,
            stiffness=self.body_stiffness,
            damping=self.body_damping,
            force_limit=self.body_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
            normalize_action=normalize_action,
        )

    @property
    def _controller_configs(self):
        right_hand = PDJointPosControllerConfig(
            self.right_finger_joint_names,
            lower=-1.0,
            upper=1.0,
            stiffness=self.body_stiffness,
            damping=self.body_damping,
            force_limit=self.body_force_limit,
            normalize_action=False,
        )
        passive = PassiveControllerConfig(
            self._passive_joint_names, damping=self.body_damping
        )
        common = dict(right_hand=right_hand, passive=passive, balance_passive_force=True)

        # Lrogic's exact posted pd_ee_delta_pose config (normalize_action=False).
        lrogic = self._arm_cfg(-ROT_BOUND_SMALL, ROT_BOUND_SMALL, False)
        # Same bounds, normalize_action left at the ManiSkill default (True) as in
        # the stock Panda/XArm6 configs.
        norm = self._arm_cfg(-ROT_BOUND_SMALL, ROT_BOUND_SMALL, True)
        # normalize_action=True with rot bounds left at the PDEEPoseControllerConfig
        # defaults (-2*pi, 2*pi), i.e. what a config that does not set them inherits.
        default_bounds = PDEEPoseControllerConfig(
            joint_names=self.right_arm_joint_names,
            pos_lower=-POS_BOUND,
            pos_upper=POS_BOUND,
            stiffness=self.body_stiffness,
            damping=self.body_damping,
            force_limit=self.body_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )
        return deepcopy_dict(
            {
                "pd_ee_delta_pose_lrogic": dict(right_arm=lrogic, **common),
                "pd_ee_delta_pose_norm": dict(right_arm=norm, **common),
                "pd_ee_delta_pose_defaultb": dict(right_arm=default_bounds, **common),
            }
        )


VARIANTS = {
    "lrogic_exact": dict(
        mode="pd_ee_delta_pose_lrogic", rot_bound=ROT_BOUND_SMALL, normalized=False
    ),
    "normalized_stock_bounds": dict(
        mode="pd_ee_delta_pose_norm", rot_bound=ROT_BOUND_SMALL, normalized=True
    ),
    "normalized_default_bounds": dict(
        mode="pd_ee_delta_pose_defaultb", rot_bound=ROT_BOUND_DEFAULT, normalized=True
    ),
}

AXES = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}


def quat_angle_deg(qa, qb):
    """Angle in degrees between two wxyz quaternions."""
    d = float(torch.abs(torch.dot(qa, qb)).clamp(max=1.0))
    return float(np.degrees(2.0 * np.arccos(d)))


def rot_err_euler(q_tgt, q_cur):
    """Lrogic's rotation error: euler XYZ of q_tgt * q_cur^-1 (root-aligned delta)."""
    dq = quaternion_multiply(q_tgt, quaternion_invert(q_cur))
    return matrix_to_euler_angles(quaternion_to_matrix(dq), "XYZ")


def make_env(mode):
    env = gym.make(
        "Empty-v1",
        num_envs=1,
        obs_mode="state",
        reward_mode="none",
        control_mode=mode,
        robot_uids="g1_rot_validation",
        sim_backend="physx_cpu",
        render_backend="none",
    )
    return env


# Non-singular start: the all-zeros "standing" keyframe leaves the arm fully
# extended (straight elbow), a singular configuration where the exact-pose CPU IK
# fails on every step and nothing moves in either condition. Bend the right arm
# and put the left arm in Lrogic's "down at side" preset.
START_QPOS_OVERRIDES = {
    "left_shoulder_roll_joint": 0.21,
    "left_elbow_pitch_joint": 1.56,
    "right_shoulder_pitch_joint": 0.4,
    "right_shoulder_roll_joint": -0.3,
    "right_elbow_pitch_joint": 1.2,
}


def reset_and_settle(env, settle_steps):
    env.reset(seed=0)
    agent = env.unwrapped.agent
    kf = agent.keyframes["standing"]
    agent.robot.set_pose(kf.pose)
    qpos = np.asarray(kf.qpos, dtype=np.float32).copy()
    names = [j.name for j in agent.robot.get_active_joints()]
    for jname, val in START_QPOS_OVERRIDES.items():
        qpos[names.index(jname)] = val
    agent.robot.set_qpos(qpos)
    agent.robot.set_qvel(np.zeros(len(qpos), dtype=np.float32))
    zero = np.zeros(env.action_space.shape, dtype=np.float32)
    for _ in range(settle_steps):
        env.step(zero)
    return agent


def wrap_ik_counter(arm_ctrl, counter):
    orig = arm_ctrl.kinematics.compute_ik

    def wrapped(*a, **k):
        r = orig(*a, **k)
        counter["calls"] += 1
        if r is None:
            counter["fails"] += 1
        return r

    arm_ctrl.kinematics.compute_ik = wrapped


def arm_slice(env):
    agent = env.unwrapped.agent
    start, end = agent.controller.action_mapping["right_arm"]
    return agent, start, end


def build_action(env, start, end, a_arm):
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    action[..., start:end] = a_arm
    return action


def run_tracking(env, variant, target_kind, steps, settle_steps):
    """target_kind: 'x'|'y'|'z' rotation offsets, or 'pos_only'."""
    agent = reset_and_settle(env, settle_steps)
    _, start, end = arm_slice(env)
    arm = agent.controller.controllers["right_arm"]
    counter = {"calls": 0, "fails": 0}
    wrap_ik_counter(arm, counter)
    fails_per_step = []

    cur = arm.ee_pose_at_base
    p0 = cur.p[0].detach().clone()
    q0 = cur.q[0].detach().clone()

    if target_kind == "pos_only":
        # direction chosen inside the feasible cone found by run_ik_probe
        # (forward, slightly left, slightly up; -y/-z beyond ~2 cm are at the
        # workspace boundary from this start configuration)
        p_t = p0 + torch.tensor([0.06, 0.03, 0.03])
        q_t = q0.clone()
    else:
        ax = torch.tensor(AXES[target_kind], dtype=torch.float64)
        half = np.deg2rad(TARGET_DEG) / 2.0
        dq = torch.zeros(4, dtype=torch.float64)
        dq[0] = np.cos(half)
        dq[1:] = ax * np.sin(half)
        q_t = quaternion_multiply(dq, q0.to(torch.float64)).to(q0.dtype)
        q_t = q_t / torch.linalg.norm(q_t)
        p_t = p0.clone()

    pos_bound, rot_bound = POS_BOUND, variant["rot_bound"]
    normalized = variant["normalized"]

    ang_hist, pos_hist = [], []
    achieved_at = None
    stable = 0
    for i in range(steps):
        cur = arm.ee_pose_at_base
        p_c = cur.p[0].detach()
        q_c = cur.q[0].detach()
        ang = quat_angle_deg(q_t, q_c)
        pos_err = float(torch.linalg.norm(p_t - p_c))
        ang_hist.append(ang)
        pos_hist.append(pos_err)

        ok = pos_err < POS_TOL and (
            target_kind == "pos_only" or np.deg2rad(ang) < ROT_TOL
        )
        if ok:
            stable += 1
            if stable >= STABLE_STEPS and achieved_at is None:
                achieved_at = i
        else:
            stable = 0

        pos_step = min(POS_STEP, pos_bound)
        rot_step = min(ROT_STEP, rot_bound)
        dpos = (p_t - p_c).cpu().numpy()
        dpos = np.clip(dpos, -pos_step, pos_step)
        # NOTE: rotation error is fed for pos_only too (toward the unchanged start
        # orientation), matching Lrogic's loop where position-only refers to the
        # waypoint ACCEPTANCE, not the action; a hard zero-rotation action would
        # demand an exact orientation hold the 6-joint chain's exact-pose IK
        # cannot satisfy while translating.
        drot = rot_err_euler(q_t, q_c).cpu().numpy()
        n = float(np.linalg.norm(drot))
        if n > rot_step:
            drot = drot / n * rot_step
        if normalized:
            a_arm = np.concatenate([dpos / pos_bound, drot / rot_bound])
            a_arm = np.clip(a_arm, -1.0, 1.0)
        else:
            a_arm = np.concatenate([dpos, drot])
        fails_before = counter["fails"]
        env.step(build_action(env, start, end, a_arm.astype(np.float32)))
        fails_per_step.append(counter["fails"] - fails_before)

    # final sample
    cur = arm.ee_pose_at_base
    ang_hist.append(quat_angle_deg(q_t, cur.q[0].detach()))
    pos_hist.append(float(torch.linalg.norm(p_t - cur.p[0].detach())))

    end_streak = 0
    for f in reversed(fails_per_step):
        if f > 0:
            end_streak += 1
        else:
            break

    return {
        "target": target_kind,
        "initial_ang_deg": ang_hist[0],
        "final_ang_deg": ang_hist[-1],
        "min_ang_deg": min(ang_hist),
        "final_pos_err_m": pos_hist[-1],
        "ik_calls": counter["calls"],
        "ik_failures": counter["fails"],
        "ik_fail_streak_end": end_streak,
        "waypoint_achieved_step": achieved_at,
        "ang_hist_deg": [round(a, 3) for a in ang_hist],
        "pos_hist_m": [round(p, 5) for p in pos_hist],
    }


def run_ik_probe(env, settle_steps):
    """Direct compute_ik probe from the settled start pose: can the exact-pose CPU
    IK solve the current pose itself, and small translations with the orientation
    held? Diagnoses solver feasibility independent of the control loop."""
    from mani_skill.utils.structs import Pose as MsPose

    agent = reset_and_settle(env, settle_steps)
    arm = agent.controller.controllers["right_arm"]
    cur = arm.ee_pose_at_base
    p0 = cur.p[0].detach().clone()
    q0 = cur.q[0].detach().clone()
    probes = {"identity": torch.zeros(3)}
    for axis, name in [(0, "x"), (1, "y"), (2, "z")]:
        for sign, sname in [(1.0, "+"), (-1.0, "-")]:
            for mag in [0.01, 0.02, 0.04]:
                d = torch.zeros(3)
                d[axis] = sign * mag
                probes[f"{sname}{name}{int(mag * 100)}cm"] = d
    out = {}
    for name, d in probes.items():
        pose = MsPose.create_from_pq((p0 + d).unsqueeze(0), q0.unsqueeze(0))
        r = arm.kinematics.compute_ik(
            pose=pose,
            q0=agent.robot.get_qpos(),
            solver_config=arm.config.delta_solver_config,
        )
        out[name] = r is not None
    return out


def run_open_loop(env, variant, settle_steps):
    """One step: +1 normalized (or +rot_bound raw) rotation action about z.
    Returns the signed realized target-orientation delta about z (radians)."""
    agent = reset_and_settle(env, settle_steps)
    _, start, end = arm_slice(env)
    arm = agent.controller.controllers["right_arm"]
    q_before = arm.ee_pose_at_base.q[0].detach().clone()
    a_arm = np.zeros(6, dtype=np.float32)
    if variant["normalized"]:
        # a full-scale action with a +-2*pi bound realizes a 2*pi rotation, which
        # aliases to zero in the euler extraction; use 0.05 there instead
        a_arm[5] = 1.0 if variant["rot_bound"] <= np.pi else 0.05
    else:
        a_arm[5] = min(variant["rot_bound"], 0.1)
    env.step(build_action(env, start, end, a_arm))
    tq = arm._target_pose.q[0].detach()
    d_euler = rot_err_euler(tq, q_before).cpu().numpy()
    return {
        "commanded_action_z": float(a_arm[5]),
        "realized_target_delta_euler_xyz_rad": [round(float(v), 5) for v in d_euler],
        "realized_target_delta_z_rad": round(float(d_euler[2]), 5),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=["before", "after"])
    ap.add_argument("--expect-scale", required=True, choices=["rot_lower", "rot_upper"])
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--settle", type=int, default=30)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Verify the loaded module really has the scaling we think it has.
    import inspect
    from mani_skill.agents.controllers.pd_ee_pose import PDEEPoseController

    src = inspect.getsource(PDEEPoseController._clip_and_scale_action)
    has_bug = "rot_action * self.config.rot_lower" in src
    has_fix = "rot_action * self.config.rot_upper" in src
    assert has_bug != has_fix, "ambiguous scaling source"
    actual = "rot_lower" if has_bug else "rot_upper"
    assert actual == args.expect_scale, (
        f"loaded module scales by {actual}, expected {args.expect_scale}"
    )

    import mani_skill

    results = {
        "condition": args.condition,
        "scaling": actual,
        "versions": {
            "mani_skill": getattr(mani_skill, "__version__", "unknown"),
            "sapien": sapien.__version__,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "python": sys.version.split()[0],
        },
        "variants": {},
    }

    for name, variant in VARIANTS.items():
        env = make_env(variant["mode"])
        runs = {}
        for tk in ["z", "y", "x", "pos_only"]:
            print(f"[{args.condition}] {name} target={tk}", flush=True)
            runs[tk] = run_tracking(env, variant, tk, args.steps, args.settle)
        ol = run_open_loop(env, variant, args.settle)
        probe = run_ik_probe(env, args.settle)
        env.close()
        results["variants"][name] = {"runs": runs, "open_loop": ol, "ik_probe": probe}

    with open(args.out, "w") as f:
        json.dump(results, f)
    print(f"[{args.condition}] done -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
'''

# ---------------------------------------------------------------------------
# Remedies worker: tests fixes for Lrogic's EXACT config (normalize_action=False,
# so the rot_lower/rot_upper bug path is never entered; runs on the stock tree).
#   1. wrist-DOF G1 (29-DOF from unitree_ros, 7-DOF arm + waist yaw in the chain)
#   2. best-effort IK fallback (use the unconverged pinocchio iterate, clamped,
#      instead of silently holding qpos)
#   3. smaller per-step rotation caps (0.05 / 0.02 rad)
# ---------------------------------------------------------------------------
REMEDIES_WORKER_SRC = r'''
"""Remedies worker: Lrogic-exact config on wristless vs wrist-DOF G1, with and
without a best-effort IK fallback and smaller rotation caps."""
import argparse
import json
import sys

import numpy as np
import torch
import sapien
import gymnasium as gym

import mani_skill.envs  # noqa: F401
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import (
    PDEEPoseControllerConfig,
    PDJointPosControllerConfig,
    PassiveControllerConfig,
    deepcopy_dict,
)
from mani_skill.agents.registration import register_agent
from mani_skill.agents.robots.unitree_g1.g1_upper_body import UnitreeG1UpperBody
from mani_skill.utils import common as ms_common
from mani_skill.utils.geometry.rotation_conversions import (
    matrix_to_euler_angles,
    quaternion_invert,
    quaternion_multiply,
    quaternion_to_matrix,
)
from mani_skill.utils.structs import Pose as MsPose

POS_BOUND = 0.1
ROT_BOUND = 0.1
POS_STEP = 0.02
TARGET_DEG = 30.0
POS_TOL = 0.03
ROT_TOL = 0.5
STABLE_STEPS = 10
FALLBACK_CLAMP = 0.2  # rad, per-joint clamp on the unconverged IK iterate

AXES = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}

WRIST_URDF = "/root/g1_description/g1_29dof_rev_1_0.urdf"


@register_agent()
class WristlessG1(UnitreeG1UpperBody):
    """Lrogic's custom_g1.py, exact posted pd_ee_delta_pose config only."""

    uid = "g1_wristless_lrogic"
    ee_link_name = "right_tcp_link"

    right_arm_joint_names = [
        "torso_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_pitch_joint",
        "right_elbow_roll_joint",
    ]
    right_finger_joint_names = [
        "right_zero_joint",
        "right_three_joint",
        "right_five_joint",
        "right_one_joint",
        "right_four_joint",
        "right_six_joint",
        "right_two_joint",
    ]

    @property
    def _passive_joint_names(self):
        controlled = set(self.right_arm_joint_names + self.right_finger_joint_names)
        return [j for j in self.body_joints if j not in controlled]

    @property
    def _controller_configs(self):
        arm = PDEEPoseControllerConfig(
            joint_names=self.right_arm_joint_names,
            pos_lower=-POS_BOUND,
            pos_upper=POS_BOUND,
            rot_lower=-ROT_BOUND,
            rot_upper=ROT_BOUND,
            stiffness=self.body_stiffness,
            damping=self.body_damping,
            force_limit=self.body_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
            normalize_action=False,
        )
        right_hand = PDJointPosControllerConfig(
            self.right_finger_joint_names,
            lower=-1.0,
            upper=1.0,
            stiffness=self.body_stiffness,
            damping=self.body_damping,
            force_limit=self.body_force_limit,
            normalize_action=False,
        )
        passive = PassiveControllerConfig(
            self._passive_joint_names, damping=self.body_damping
        )
        return deepcopy_dict(
            {
                "pd_ee_delta_pose_lrogic": dict(
                    right_arm=arm,
                    right_hand=right_hand,
                    passive=passive,
                    balance_passive_force=True,
                )
            }
        )


_W29_LEGS = [
    f"{side}_{j}_joint"
    for side in ["left", "right"]
    for j in ["hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"]
]
_W29_WAIST = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
_W29_ARM = {
    side: [
        f"{side}_{j}_joint"
        for j in [
            "shoulder_pitch",
            "shoulder_roll",
            "shoulder_yaw",
            "elbow",
            "wrist_roll",
            "wrist_pitch",
            "wrist_yaw",
        ]
    ]
    for side in ["left", "right"]
}
_W29_ALL = _W29_LEGS + _W29_WAIST + _W29_ARM["left"] + _W29_ARM["right"]


@register_agent()
class G1Wrist29(BaseAgent):
    """29-DOF G1 (unitree_ros g1_29dof_rev_1_0): 7-DOF right arm with wrist
    roll/pitch/yaw. Same posture as Lrogic's setup: waist yaw in the chain
    (their torso_joint analog), everything else passive, root fixed."""

    uid = "g1_wrist29_lrogic"
    urdf_path = WRIST_URDF
    urdf_config = dict()
    fix_root_link = True
    load_multiple_collisions = False

    keyframes = dict(
        standing=Keyframe(
            pose=sapien.Pose(p=[0, 0, 0.80]),
            qpos=np.zeros(29, dtype=np.float32),
        )
    )

    body_stiffness = 1e3
    body_damping = 1e2
    body_force_limit = 100

    right_arm_joint_names = ["waist_yaw_joint"] + _W29_ARM["right"]

    @property
    def _passive_joint_names(self):
        controlled = set(self.right_arm_joint_names)
        return [j for j in _W29_ALL if j not in controlled]

    @property
    def _controller_configs(self):
        arm = PDEEPoseControllerConfig(
            joint_names=self.right_arm_joint_names,
            pos_lower=-POS_BOUND,
            pos_upper=POS_BOUND,
            rot_lower=-ROT_BOUND,
            rot_upper=ROT_BOUND,
            stiffness=self.body_stiffness,
            damping=self.body_damping,
            force_limit=self.body_force_limit,
            ee_link="right_rubber_hand",
            urdf_path=self.urdf_path,
            normalize_action=False,
        )
        passive = PassiveControllerConfig(
            self._passive_joint_names, damping=self.body_damping
        )
        return deepcopy_dict(
            {
                "pd_ee_delta_pose_lrogic": dict(
                    right_arm=arm, passive=passive, balance_passive_force=True
                )
            }
        )


MODELS = {
    "wristless": dict(
        uid="g1_wristless_lrogic",
        overrides={
            "left_shoulder_roll_joint": 0.21,
            "left_elbow_pitch_joint": 1.56,
            "right_shoulder_pitch_joint": 0.4,
            "right_shoulder_roll_joint": -0.3,
            "right_elbow_pitch_joint": 1.2,
        },
    ),
    "wrist29": dict(
        uid="g1_wrist29_lrogic",
        overrides={
            "right_shoulder_pitch_joint": 0.4,
            "right_shoulder_roll_joint": -0.3,
            "right_elbow_joint": 1.0,
        },
    ),
}

RUNS = [
    ("wristless_base", "wristless", 0.1, False),
    ("wristless_cap05", "wristless", 0.05, False),
    ("wristless_cap02", "wristless", 0.02, False),
    ("wristless_fallback", "wristless", 0.1, True),
    ("wristless_fallback_cap05", "wristless", 0.05, True),
    ("wrist29_base", "wrist29", 0.1, False),
    ("wrist29_fallback", "wrist29", 0.1, True),
]


def quat_angle_deg(qa, qb):
    d = float(torch.abs(torch.dot(qa, qb)).clamp(max=1.0))
    return float(np.degrees(2.0 * np.arccos(d)))


def rot_err_euler(q_tgt, q_cur):
    dq = quaternion_multiply(q_tgt, quaternion_invert(q_cur))
    return matrix_to_euler_angles(quaternion_to_matrix(dq), "XYZ")


def make_env(uid):
    return gym.make(
        "Empty-v1",
        num_envs=1,
        obs_mode="state",
        reward_mode="none",
        control_mode="pd_ee_delta_pose_lrogic",
        robot_uids=uid,
        sim_backend="physx_cpu",
        render_backend="none",
    )


def reset_and_settle(env, model, settle_steps):
    env.reset(seed=0)
    agent = env.unwrapped.agent
    kf = agent.keyframes["standing"]
    agent.robot.set_pose(kf.pose)
    qpos = np.asarray(kf.qpos, dtype=np.float32).copy()
    names = [j.name for j in agent.robot.get_active_joints()]
    for jname, val in model["overrides"].items():
        qpos[names.index(jname)] = val
    agent.robot.set_qpos(qpos)
    agent.robot.set_qvel(np.zeros(len(qpos), dtype=np.float32))
    zero = np.zeros(env.action_space.shape, dtype=np.float32)
    for _ in range(settle_steps):
        env.step(zero)
    return agent


def attach_counting_ik(arm, counters, best_effort):
    """Replace arm.kinematics.compute_ik on this instance. Counts failures; if
    best_effort, return the unconverged pinocchio iterate with the per-joint step
    clamped to FALLBACK_CLAMP instead of None (prototype of a config-gated
    controller option; stock behavior silently holds _start_qpos)."""
    kin = arm.kinematics

    def compute_ik(pose, q0, is_delta_pose=False, current_pose=None,
                   solver_config=None):
        q0m = q0[:, kin.pmodel_active_joint_indices]
        init = q0m.cpu().numpy()[0]
        result, success, _err = kin.pmodel.compute_inverse_kinematics(
            kin.end_link_idx,
            pose.sp,
            initial_qpos=init,
            active_qmask=kin.qmask,
            max_iterations=100,
        )
        counters["calls"] += 1
        if success:
            return ms_common.to_tensor(
                [result[kin.pmodel_controlled_joint_indices]], device=kin.device
            )
        counters["fails"] += 1
        if not best_effort:
            return None
        counters["fallback_used"] += 1
        res = np.asarray(result, dtype=np.float64)
        cur = np.asarray(init, dtype=np.float64)
        new_ctrl = res[kin.pmodel_controlled_joint_indices]
        cur_ctrl = cur[kin.pmodel_controlled_joint_indices]
        step = np.clip(new_ctrl - cur_ctrl, -FALLBACK_CLAMP, FALLBACK_CLAMP)
        return ms_common.to_tensor([cur_ctrl + step], device=kin.device)

    arm.kinematics.compute_ik = compute_ik


def run_tracking(env, model, target_kind, rot_step, best_effort, steps, settle):
    agent = reset_and_settle(env, model, settle)
    start, end = agent.controller.action_mapping["right_arm"]
    arm = agent.controller.controllers["right_arm"]
    counters = {"calls": 0, "fails": 0, "fallback_used": 0}
    attach_counting_ik(arm, counters, best_effort)

    cur = arm.ee_pose_at_base
    p0 = cur.p[0].detach().clone()
    q0 = cur.q[0].detach().clone()

    if target_kind == "pos_only":
        p_t = p0 + torch.tensor([0.06, 0.03, 0.03])
        q_t = q0.clone()
    else:
        ax = torch.tensor(AXES[target_kind], dtype=torch.float64)
        half = np.deg2rad(TARGET_DEG) / 2.0
        dq = torch.zeros(4, dtype=torch.float64)
        dq[0] = np.cos(half)
        dq[1:] = ax * np.sin(half)
        q_t = quaternion_multiply(dq, q0.to(torch.float64)).to(q0.dtype)
        q_t = q_t / torch.linalg.norm(q_t)
        p_t = p0.clone()

    ang_hist, pos_hist = [], []
    achieved_at = None
    stable = 0
    for i in range(steps):
        cur = arm.ee_pose_at_base
        p_c = cur.p[0].detach()
        q_c = cur.q[0].detach()
        ang = quat_angle_deg(q_t, q_c)
        pos_err = float(torch.linalg.norm(p_t - p_c))
        ang_hist.append(ang)
        pos_hist.append(pos_err)

        ok = pos_err < POS_TOL and (
            target_kind == "pos_only" or np.deg2rad(ang) < ROT_TOL
        )
        if ok:
            stable += 1
            if stable >= STABLE_STEPS and achieved_at is None:
                achieved_at = i
        else:
            stable = 0

        dpos = (p_t - p_c).cpu().numpy()
        dpos = np.clip(dpos, -POS_STEP, POS_STEP)
        drot = rot_err_euler(q_t, q_c).cpu().numpy()
        n = float(np.linalg.norm(drot))
        if n > rot_step:
            drot = drot / n * rot_step
        a_arm = np.concatenate([dpos, drot]).astype(np.float32)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        action[..., start:end] = a_arm
        env.step(action)

    cur = arm.ee_pose_at_base
    ang_hist.append(quat_angle_deg(q_t, cur.q[0].detach()))
    pos_hist.append(float(torch.linalg.norm(p_t - cur.p[0].detach())))

    return {
        "target": target_kind,
        "final_ang_deg": ang_hist[-1],
        "min_ang_deg": min(ang_hist),
        "final_pos_err_m": pos_hist[-1],
        "ik_calls": counters["calls"],
        "ik_failures": counters["fails"],
        "fallback_used": counters["fallback_used"],
        "waypoint_achieved_step": achieved_at,
        "ang_hist_deg": [round(a, 3) for a in ang_hist],
        "pos_hist_m": [round(p, 5) for p in pos_hist],
    }


def run_ik_probe(env, model, settle):
    agent = reset_and_settle(env, model, settle)
    arm = agent.controller.controllers["right_arm"]
    cur = arm.ee_pose_at_base
    p0 = cur.p[0].detach().clone()
    q0 = cur.q[0].detach().clone()
    out = {}
    for axis, name in [(0, "x"), (1, "y"), (2, "z")]:
        for sign, sname in [(1.0, "+"), (-1.0, "-")]:
            d = torch.zeros(3)
            d[axis] = sign * 0.02
            pose = MsPose.create_from_pq((p0 + d).unsqueeze(0), q0.unsqueeze(0))
            r = arm.kinematics.compute_ik(
                pose=pose,
                q0=agent.robot.get_qpos(),
                solver_config=arm.config.delta_solver_config,
            )
            out[f"{sname}{name}2cm"] = r is not None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--settle", type=int, default=30)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Record which scaling the loaded tree uses (stock expected; the exact config
    # under test bypasses it either way since normalize_action=False).
    import inspect
    from mani_skill.agents.controllers.pd_ee_pose import PDEEPoseController

    src = inspect.getsource(PDEEPoseController._clip_and_scale_action)
    scaling = "rot_lower" if "rot_action * self.config.rot_lower" in src else "rot_upper"

    import mani_skill

    results = {
        "scaling_in_tree": scaling,
        "versions": {
            "mani_skill": getattr(mani_skill, "__version__", "unknown"),
            "sapien": sapien.__version__,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "python": sys.version.split()[0],
        },
        "runs": {},
        "ik_probe": {},
    }

    envs = {}
    for name, model_key, rot_step, best_effort in RUNS:
        model = MODELS[model_key]
        if model_key not in envs:
            envs[model_key] = make_env(model["uid"])
        env = envs[model_key]
        runs = {}
        for tk in ["z", "y", "x", "pos_only"]:
            print(f"[remedies] {name} target={tk}", flush=True)
            runs[tk] = run_tracking(
                env, model, tk, rot_step, best_effort, args.steps, args.settle
            )
        results["runs"][name] = {
            "model": model_key,
            "rot_step": rot_step,
            "ik_fallback": best_effort,
            "targets": runs,
        }
    for model_key, env in envs.items():
        results["ik_probe"][model_key] = run_ik_probe(
            env, MODELS[model_key], args.settle
        )
        env.close()

    with open(args.out, "w") as f:
        json.dump(results, f)
    print(f"[remedies] done -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
'''


# ---------------------------------------------------------------------------
# Plotting (runs remotely; matplotlib is in the image, not the local env).
# Eximius series colors (dataviz-validated pair): stock main = steel blue,
# after fix = terracotta; ink #1b2f4b, muted #7d8a9c, grid #c6cdd8.
# ---------------------------------------------------------------------------
COL_BEFORE = "#3A6EA5"
COL_AFTER = "#C2622E"
COL_INK = "#1b2f4b"
COL_MUTED = "#7d8a9c"
COL_GRID = "#c6cdd8"


def _make_plots(results):
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = {}
    panels = [("x", "rotation +30° about x"), ("y", "rotation +30° about y"),
              ("z", "rotation +30° about z")]
    variant_titles = {
        "normalized_stock_bounds":
            "G1 right arm, pd_ee_delta_pose, normalize_action=True, rot bounds ±0.1 rad",
        "normalized_default_bounds":
            "G1 right arm, pd_ee_delta_pose, normalize_action=True, default rot bounds ±2π",
    }
    for variant, subtitle in variant_titles.items():
        fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), sharey=True)
        for ax, (axis_key, title) in zip(axes, panels):
            for cond, col, label in [
                ("before", COL_BEFORE, "stock main (scale by rot_lower)"),
                ("after", COL_AFTER, "one-line fix (scale by rot_upper)"),
            ]:
                run = results[cond]["variants"][variant]["runs"][axis_key]
                y = run["ang_hist_deg"]
                ax.plot(range(len(y)), y, color=col, linewidth=2.0, label=label)
                ax.annotate(
                    f'{run["final_ang_deg"]:.1f}°',
                    xy=(len(y) - 1, y[-1]),
                    xytext=(4, 0),
                    textcoords="offset points",
                    color=col,
                    fontsize=9,
                    va="center",
                )
            ax.set_title(title, color=COL_INK, fontsize=11)
            ax.set_xlabel("control step", color=COL_INK, fontsize=10)
            ax.grid(True, color=COL_GRID, linewidth=0.5, alpha=0.7)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            for spine in ["left", "bottom"]:
                ax.spines[spine].set_color(COL_MUTED)
            ax.tick_params(colors=COL_MUTED, labelsize=9)
        axes[0].set_ylabel("orientation error (deg)", color=COL_INK, fontsize=10)
        axes[0].legend(loc="upper left", fontsize=9, frameon=False,
                       labelcolor=COL_INK)
        fig.suptitle(
            "EE-pose rotation tracking, Unitree G1 (ManiSkill #1435 control loop)\n"
            + subtitle,
            color=COL_INK,
            fontsize=12,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, facecolor="white")
        plt.close(fig)
        figs[f"g1_rot_tracking_{variant}.png"] = buf.getvalue()
    return figs


# Remedies series colors (dataviz-validated triple): baseline gold, IK-fallback
# steel blue, wrist-DOF model terracotta (the headline series).
COL_BASE = "#9a7522"
COL_FALLBACK = "#3A6EA5"
COL_WRIST = "#C2622E"


def _make_remedies_plot(results):
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = [
        ("wristless_base", COL_BASE, "wristless G1 (Lrogic exact config)"),
        ("wristless_fallback", COL_FALLBACK, "wristless + best-effort IK fallback"),
        ("wrist29_base", COL_WRIST, "wrist-DOF G1 (29-DOF, unitree_ros)"),
    ]
    panels = [("x", "rotation +30° about x"), ("y", "rotation +30° about y"),
              ("z", "rotation +30° about z")]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), sharey=True)
    for ax, (axis_key, title) in zip(axes, panels):
        for run_name, col, label in series:
            run = results["runs"][run_name]["targets"][axis_key]
            y = run["ang_hist_deg"]
            ax.plot(range(len(y)), y, color=col, linewidth=2.0, label=label)
            ax.annotate(
                f'{run["final_ang_deg"]:.1f}°',
                xy=(len(y) - 1, y[-1]),
                xytext=(4, 0),
                textcoords="offset points",
                color=col,
                fontsize=9,
                va="center",
            )
        ax.set_title(title, color=COL_INK, fontsize=11)
        ax.set_xlabel("control step", color=COL_INK, fontsize=10)
        ax.grid(True, color=COL_GRID, linewidth=0.5, alpha=0.7)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color(COL_MUTED)
        ax.tick_params(colors=COL_MUTED, labelsize=9)
    axes[0].set_ylabel("orientation error (deg)", color=COL_INK, fontsize=10)
    axes[0].legend(loc="upper right", fontsize=9, frameon=False, labelcolor=COL_INK)
    fig.suptitle(
        "Remedies for the exact #1435 config (normalize_action=False, stock ManiSkill)",
        color=COL_INK,
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, facecolor="white")
    plt.close(fig)
    return {"g1_remedies_exact_config.png": buf.getvalue()}


def _remedies_impl(steps: int, settle: int):
    import subprocess
    import sys

    worker = pathlib.Path("/root/remedies_worker.py")
    worker.write_text(REMEDIES_WORKER_SRC)
    out = "/root/remedies.json"
    subprocess.run(
        [sys.executable, str(worker), "--steps", str(steps),
         "--settle", str(settle), "--out", out],
        check=True,
    )
    with open(out) as f:
        results = json.load(f)
    plots = _make_remedies_plot(results)
    return {
        "results": results,
        "meta": {
            "maniskill_commit": MANISKILL_COMMIT,
            "unitree_ros_commit": UNITREE_ROS_COMMIT,
            "steps": steps,
            "settle": settle,
            "sim_backend": "physx_cpu",
            "num_envs": 1,
        },
        "plots": {k: base64.b64encode(v).decode() for k, v in plots.items()},
    }


def _validate_impl(steps: int, settle: int):
    import importlib.util
    import py_compile
    import subprocess
    import sys

    spec = importlib.util.find_spec("mani_skill")
    pd_file = pathlib.Path(spec.origin).parent / "agents" / "controllers" / "pd_ee_pose.py"
    src = pd_file.read_text()
    assert src.count(BUG_LINE) == 1, "expected exactly one rot_lower scaling line in stock main"
    pristine = src

    worker = pathlib.Path("/root/rot_worker.py")
    worker.write_text(WORKER_SRC)

    results = {}
    for cond, text, expect in [
        ("before", pristine, "rot_lower"),
        ("after", pristine.replace(BUG_LINE, FIX_LINE), "rot_upper"),
    ]:
        pd_file.write_text(text)
        py_compile.compile(str(pd_file), doraise=True)
        out = f"/root/{cond}.json"
        subprocess.run(
            [
                sys.executable,
                str(worker),
                "--condition", cond,
                "--expect-scale", expect,
                "--steps", str(steps),
                "--settle", str(settle),
                "--out", out,
            ],
            check=True,
        )
        with open(out) as f:
            results[cond] = json.load(f)

    plots = _make_plots(results)
    return {
        "results": results,
        "meta": {
            "maniskill_commit": MANISKILL_COMMIT,
            "steps": steps,
            "settle": settle,
            "sim_backend": "physx_cpu",
            "num_envs": 1,
        },
        "plots": {k: base64.b64encode(v).decode() for k, v in plots.items()},
    }


# NOTE: A10G was tried first but its driver rejects Vulkan device creation here
# (vk ErrorInitializationFailed at RenderMaterial init); T4 works and the
# simulation itself is physx_cpu, so the GPU only serves SAPIEN's Vulkan init.
@app.function(image=img, gpu="T4", timeout=5400)
def validate(steps: int = 200, settle: int = 30):
    return _validate_impl(steps, settle)


@app.function(image=img, gpu="T4", timeout=3600)
def validate_smoke(steps: int = 12, settle: int = 8):
    return _validate_impl(steps, settle)


@app.function(image=img, gpu="T4", timeout=5400)
def remedies(steps: int = 200, settle: int = 30):
    return _remedies_impl(steps, settle)


@app.function(image=img, gpu="T4", timeout=3600)
def remedies_smoke(steps: int = 12, settle: int = 8):
    return _remedies_impl(steps, settle)


def _print_remedies_table(payload):
    results = payload["results"]
    meta = payload["meta"]
    print(f"\nRemedies on Lrogic-exact config (stock tree, scaling={results['scaling_in_tree']}), "
          f"ManiSkill {meta['maniskill_commit'][:9]}, unitree_ros {meta['unitree_ros_commit'][:9]}, "
          f"steps={meta['steps']}")
    header = (f"{'run':<26} {'rot_step':>8} {'fb':>3} "
              f"{'z final°':>9} {'y final°':>9} {'x final°':>9} {'pos(cm)':>8} "
              f"{'IKfail z/y/x':>14} {'wp z/y/x/pos':>13}")
    print(header)
    print("-" * len(header))
    for name, run in results["runs"].items():
        t = run["targets"]
        wp = "/".join(
            "y" if t[k]["waypoint_achieved_step"] is not None else "n"
            for k in ["z", "y", "x", "pos_only"]
        )
        ikf = "/".join(str(t[k]["ik_failures"]) for k in ["z", "y", "x"])
        print(f"{name:<26} {run['rot_step']:>8} {('+' if run['ik_fallback'] else '-'):>3} "
              f"{t['z']['final_ang_deg']:>9.2f} {t['y']['final_ang_deg']:>9.2f} "
              f"{t['x']['final_ang_deg']:>9.2f} {t['pos_only']['final_pos_err_m'] * 100:>8.2f} "
              f"{ikf:>14} {wp:>13}")
    print("\nIK probe (2 cm translations, orientation held):")
    for model, probe in results["ik_probe"].items():
        print(f"  {model:<10} " + "  ".join(f"{k}:{'ok' if s else 'FAIL'}" for k, s in probe.items()))


def _print_table(payload):
    results = payload["results"]
    meta = payload["meta"]
    print(f"\nManiSkill commit {meta['maniskill_commit']}  "
          f"({meta['sim_backend']}, num_envs={meta['num_envs']}, steps={meta['steps']})")
    v = results["before"]["variants"]
    header = (f"{'variant':<28} {'target':<9} "
              f"{'before final°':>13} {'after final°':>12} "
              f"{'before IKfail':>13} {'after IKfail':>12} "
              f"{'before wp':>9} {'after wp':>8}")
    print(header)
    print("-" * len(header))
    for variant in v:
        for tk in ["z", "y", "x", "pos_only"]:
            b = results["before"]["variants"][variant]["runs"][tk]
            a = results["after"]["variants"][variant]["runs"][tk]
            wp_b = "yes" if b["waypoint_achieved_step"] is not None else "no"
            wp_a = "yes" if a["waypoint_achieved_step"] is not None else "no"
            if tk == "pos_only":
                print(f"{variant:<28} {'pos(cm)':<9} "
                      f"{b['final_pos_err_m'] * 100:>13.2f} {a['final_pos_err_m'] * 100:>12.2f} "
                      f"{b['ik_failures']:>13d} {a['ik_failures']:>12d} "
                      f"{wp_b:>9} {wp_a:>8}")
            else:
                print(f"{variant:<28} {tk:<9} "
                      f"{b['final_ang_deg']:>13.2f} {a['final_ang_deg']:>12.2f} "
                      f"{b['ik_failures']:>13d} {a['ik_failures']:>12d} "
                      f"{wp_b:>9} {wp_a:>8}")
    probe = results["before"]["variants"][next(iter(v))].get("ik_probe")
    if probe:
        ok = [k for k, s in probe.items() if s]
        bad = [k for k, s in probe.items() if not s]
        print(f"\nIK probe from start pose (orientation held): OK={ok}")
        print(f"                                            FAIL={bad}")
    print("\nOpen-loop (+1 normalized rot-z action, signed realized target delta about z, rad):")
    for variant in v:
        b = results["before"]["variants"][variant]["open_loop"]
        a = results["after"]["variants"][variant]["open_loop"]
        print(f"  {variant:<28} before {b['realized_target_delta_z_rad']:+.4f}   "
              f"after {a['realized_target_delta_z_rad']:+.4f}")


@app.local_entrypoint()
def main(action: str = "run", steps: int = 200):
    out_dir = pathlib.Path(os.environ.get("MSK_VAL_OUT", DEFAULT_OUT_DIR))
    out_dir.mkdir(parents=True, exist_ok=True)

    if action == "smoke":
        payload = validate_smoke.remote()
        tag = "smoke_"
    elif action == "run":
        payload = validate.remote(steps=steps)
        tag = ""
    elif action == "remedies-smoke":
        payload = remedies_smoke.remote()
        tag = "smoke_remedies_"
    elif action == "remedies":
        payload = remedies.remote(steps=steps)
        tag = "remedies_"
    else:
        raise SystemExit(f"unknown action {action}")

    with open(out_dir / f"{tag}results.json", "w") as f:
        json.dump({"results": payload["results"], "meta": payload["meta"]}, f, indent=2)
    for name, b64 in payload["plots"].items():
        (out_dir / f"{tag}{name}").write_bytes(base64.b64decode(b64))
    if action.startswith("remedies"):
        _print_remedies_table(payload)
    else:
        _print_table(payload)
    print(f"\nSaved to {out_dir}")
