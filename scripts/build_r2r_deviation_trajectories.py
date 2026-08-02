#!/usr/bin/env python3
"""Build deterministic, full-trajectory R2R deviation variants.

Each output episode follows the original route until the middle of every
eligible fine-grained segment, inserts one rule-based perturbation, and then
uses Habitat's shortest-path follower to reconnect to that segment's reference
path endpoint.  The next segment starts only after that endpoint is reached.
No language model is queried.

The first version intentionally inserts at most one perturbation per segment:
this makes the semantic boundary and the replay trace straightforward to audit.
Segments shorter than ``--min-segment-frames`` are replayed unchanged.

Examples:
  # Small rendered smoke test (writes isolated *_test outputs).
  PYTHONPATH=. python scripts/build_r2r_deviation_trajectories.py \\
      --episode-id 1 --gpu-id 0 \\
      --images-root data/StreamVLN-Trajectory-Data/R2R/gt_val_unseen_deviation_test_images \\
      --output data/datasets/R2R_VLNCE_v1-3/val_unseen/fgr2r_val_unseen_deviation_test.json

  # Build all valid R2R val-unseen variants.
  PYTHONPATH=. python scripts/build_r2r_deviation_trajectories.py --gpu-id 0
"""

import argparse
import copy
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import quaternion
from PIL import Image

import habitat
from habitat.config.default import get_config
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEGMENTATION = ROOT / "data/datasets/R2R_VLNCE_v1-3/val_unseen/fgr2r_val_unseen_continuous.json"
DEFAULT_EPISODES = ROOT / "data/datasets/R2R_VLNCE_v1-3/val_unseen/val_unseen.json"
DEFAULT_OUTPUT = ROOT / "data/datasets/R2R_VLNCE_v1-3/val_unseen/fgr2r_val_unseen_deviation.json"
DEFAULT_IMAGES_ROOT = ROOT / "data/StreamVLN-Trajectory-Data/R2R/gt_val_unseen_deviation_images"
DEFAULT_REPORT = ROOT / "data/datasets/R2R_VLNCE_v1-3/val_unseen/fgr2r_val_unseen_deviation_report.json"

ACTION_STOP = 0
ACTION_FORWARD = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
HABITAT_TO_ACTION = {
    HabitatSimActions.stop: ACTION_STOP,
    HabitatSimActions.move_forward: ACTION_FORWARD,
    HabitatSimActions.turn_left: ACTION_LEFT,
    HabitatSimActions.turn_right: ACTION_RIGHT,
}
PERTURBATION_TYPES = ("heading", "lateral_detour", "short_backtrack", "local_loop")


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def atomic_dump(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segmentation", type=Path, default=DEFAULT_SEGMENTATION)
    parser.add_argument("--episodes", type=Path, default=DEFAULT_EPISODES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--images-root", type=Path, default=DEFAULT_IMAGES_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--habitat-config", type=Path, default=ROOT / "configs/vln_r2r_dual.yaml")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--episode-id", action="append", default=[])
    parser.add_argument("--max-episodes", type=int, default=0, help="0 means all selected valid episodes.")
    parser.add_argument("--min-segment-frames", type=int, default=10)
    parser.add_argument(
        "--anchor-boundary-margin-frames",
        type=int,
        default=4,
        help="Minimum source frames from both segment boundaries for a perturbation anchor.",
    )
    parser.add_argument("--goal-radius", type=float, default=0.25)
    parser.add_argument("--max-reconnect-actions", type=int, default=160)
    parser.add_argument("--heading-tolerance-deg", type=float, default=10.0)
    parser.add_argument("--max-heading-actions", type=int, default=24)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def valid_segments(entry: Dict[str, Any]) -> bool:
    segments = entry.get("instruction_segments", [])
    if not segments or int(segments[0]["start_frame"]) != 1:
        return False
    if int(segments[-1]["end_frame"]) != int(entry.get("num_frames", -1)):
        return False
    return all(
        int(segment["start_frame"]) <= int(segment["end_frame"])
        for segment in segments
    ) and all(
        int(left["end_frame"]) + 1 == int(right["start_frame"])
        for left, right in zip(segments, segments[1:])
    )


def xyzw_to_quaternion(values: Sequence[float]) -> np.quaternion:
    x, y, z, w = [float(value) for value in values]
    return np.quaternion(w, x, y, z)


def rotation_dict(rotation: np.quaternion) -> Dict[str, float]:
    values = quaternion.as_float_array(rotation)
    return {"w": float(values[0]), "x": float(values[1]), "y": float(values[2]), "z": float(values[3])}


def yaw_degrees(rotation: np.quaternion) -> float:
    """Return Habitat's yaw in [-180, 180] degrees."""
    yaw = math.degrees(math.atan2(
        2 * (rotation.w * rotation.y + rotation.x * rotation.z),
        1 - 2 * (rotation.y**2 + rotation.z**2),
    ))
    return (yaw + 180.0) % 360.0 - 180.0


def wrapped_angle_degrees(target: float, current: float) -> float:
    return (target - current + 180.0) % 360.0 - 180.0


def make_env(config_path: Path, gpu_id: int) -> Tuple[habitat.Env, Any]:
    config = get_config(str(config_path))
    config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    habitat_config = config.habitat
    habitat_config.simulator.habitat_sim_v0.gpu_device_id = gpu_id
    habitat_config.simulator.forward_step_size = 0.25
    habitat_config.simulator.turn_angle = 15
    habitat_config.environment.max_episode_steps = 100000
    habitat_config.simulator.scene = ""
    return habitat.Env(config=config), config


def switch_scene(env: habitat.Env, config: Any, scene_id: str) -> None:
    scene_path = str(ROOT / "data/scene_datasets" / scene_id)
    if env.sim.config.sim_cfg.scene_id == scene_path:
        return
    config.habitat.simulator.scene = scene_path
    env.sim.reconfigure(config.habitat.simulator)


def state_distance(position: Sequence[float], target: Sequence[float]) -> float:
    return math.dist([float(value) for value in position], [float(value) for value in target])


def target_for_segment(segment_entry: Dict[str, Any], episode: Dict[str, Any], segment_index: int) -> List[float]:
    """Recover this fine-grained segment's original reference-path endpoint."""
    reference_path = episode["reference_path"]
    if segment_index == len(segment_entry["instruction_segments"]) - 1:
        return [float(value) for value in reference_path[-1]]
    ranges = segment_entry.get("source_annotation", {}).get("chunk_view", [])
    if segment_index >= len(ranges) or len(ranges[segment_index]) != 2:
        raise ValueError(f"episode {segment_entry['episode_id']}: missing FGR chunk_view for segment {segment_index}")
    endpoint = int(ranges[segment_index][1]) - 1
    if not 0 <= endpoint < len(reference_path):
        raise ValueError(f"episode {segment_entry['episode_id']}: invalid endpoint {endpoint}")
    return [float(value) for value in reference_path[endpoint]]


def target_yaw_for_segment(segment_entry: Dict[str, Any], episode: Dict[str, Any], segment_index: int) -> float:
    """Recover the original 15-degree-grid heading at this segment endpoint."""
    end_frame = int(segment_entry["instruction_segments"][segment_index]["end_frame"])
    yaw = yaw_degrees(xyzw_to_quaternion(episode["start_rotation"]))
    for action in segment_entry["continuous_gt"]["actions"][:end_frame]:
        if int(action) == ACTION_LEFT:
            yaw += 15.0
        elif int(action) == ACTION_RIGHT:
            yaw -= 15.0
    return (yaw + 180.0) % 360.0 - 180.0


def perturbation_candidates(episode_id: int, segment_index: int) -> Iterable[Tuple[str, List[int]]]:
    """Yield deterministic variants, ordered by the desired perturbation type."""
    perturbation = PERTURBATION_TYPES[(episode_id + segment_index) % len(PERTURBATION_TYPES)]
    sign = 1 if ((episode_id * 31 + segment_index * 17) % 2 == 0) else -1
    turn = ACTION_LEFT if sign > 0 else ACTION_RIGHT
    opposite = ACTION_RIGHT if sign > 0 else ACTION_LEFT
    if perturbation == "heading":
        # 30/45/60/90 degrees, represented as 15-degree Habitat primitives.
        for steps in (2, 3, 4, 6):
            yield perturbation, [turn] * steps
    elif perturbation == "lateral_detour":
        for forwards in (2, 1, 3):
            yield perturbation, [turn] * 3 + [ACTION_FORWARD] * forwards
    elif perturbation == "short_backtrack":
        for forwards in (2, 1, 3):
            yield perturbation, [turn] * 12 + [ACTION_FORWARD] * forwards
    else:  # local_loop
        for forwards in (1, 2):
            yield perturbation, [turn] * 3 + [ACTION_FORWARD] * forwards + [opposite] * 6 + [ACTION_FORWARD] * forwards


def can_execute_perturbation(
    sim: Any,
    actions: Sequence[int],
    target: Sequence[float],
    forbidden_waypoints: Sequence[Sequence[float]],
    goal_radius: float,
) -> bool:
    """Reject collisions and any perturbation that enters another segment's waypoint.

    The current segment's target is also forbidden during the perturbation, so
    a rule-based detour cannot complete the current segment early.  It may only
    be reached by the explicit shortest-path reconnect phase afterwards.
    """
    saved = sim.get_agent_state()
    saved_position = np.array(saved.position, dtype=np.float32).copy()
    saved_rotation = saved.rotation
    try:
        for action in actions:
            before = sim.get_agent_state().position
            sim.step(int(action))
            after = sim.get_agent_state().position
            if action == ACTION_FORWARD and state_distance(before, after) < 0.05:
                return False
            if state_distance(after, target) <= goal_radius:
                return False
            if any(state_distance(after, waypoint) <= goal_radius for waypoint in forbidden_waypoints):
                return False
        return True
    finally:
        sim.set_agent_state(saved_position, saved_rotation)


class TrajectoryWriter:
    """Capture initial/post-action frames using the project's next-action convention."""

    def __init__(self, sim: Any, rgb_dir: Path, jpeg_quality: int, overwrite: bool):
        self.sim = sim
        self.rgb_dir = rgb_dir
        self.jpeg_quality = jpeg_quality
        self.overwrite = overwrite
        self.records: List[Dict[str, Any]] = []

    def capture(self, segment_index: int, event: str, executed_action: int) -> None:
        frame = len(self.records) + 1
        image_path = self.rgb_dir / f"{frame:03d}.jpg"
        if self.overwrite or not image_path.is_file():
            image_path.parent.mkdir(parents=True, exist_ok=True)
            rgb = self.sim.get_sensor_observations()["rgb"][:, :, :3]
            Image.fromarray(rgb).convert("RGB").save(image_path, quality=self.jpeg_quality)
        state = self.sim.get_agent_state()
        self.records.append({
            "frame": frame,
            "step": frame,
            "position": [float(value) for value in state.position],
            "rotation": rotation_dict(state.rotation),
            "action": ACTION_STOP,  # Set immediately before the next transition.
            "executed_action": int(executed_action),
            "segment_index": int(segment_index),
            "event": event,
        })

    def execute(self, action: int, next_segment_index: int, event: str) -> None:
        if not self.records:
            raise RuntimeError("initial state must be captured before executing an action")
        self.records[-1]["action"] = int(action)
        self.sim.step(int(action))
        self.capture(next_segment_index, event, action)


def reconnect(
    writer: TrajectoryWriter,
    segment_index: int,
    target: Sequence[float],
    target_yaw: float,
    goal_radius: float,
    maximum_actions: int,
    heading_tolerance_degrees: float,
    maximum_heading_actions: int,
) -> Tuple[bool, int, float, float]:
    follower = ShortestPathFollower(writer.sim, goal_radius=goal_radius, return_one_hot=False, stop_on_error=True)
    for count in range(maximum_actions):
        if state_distance(writer.sim.get_agent_state().position, target) <= goal_radius:
            break
        next_action = follower.get_next_action(np.array(target, dtype=np.float32))
        action = HABITAT_TO_ACTION.get(next_action, ACTION_STOP)
        if action == ACTION_STOP:
            break
        writer.execute(action, segment_index, "reconnect")
    position_error = state_distance(writer.sim.get_agent_state().position, target)
    if position_error > goal_radius:
        return False, maximum_actions, position_error, float("inf")

    heading_actions = 0
    for _ in range(maximum_heading_actions):
        current_yaw = yaw_degrees(writer.sim.get_agent_state().rotation)
        heading_error = abs(wrapped_angle_degrees(target_yaw, current_yaw))
        if heading_error <= heading_tolerance_degrees:
            return True, count + heading_actions, position_error, heading_error
        action = ACTION_LEFT if wrapped_angle_degrees(target_yaw, current_yaw) > 0 else ACTION_RIGHT
        writer.execute(action, segment_index, "reconnect_heading")
        heading_actions += 1
    current_yaw = yaw_degrees(writer.sim.get_agent_state().rotation)
    heading_error = abs(wrapped_angle_degrees(target_yaw, current_yaw))
    return heading_error <= heading_tolerance_degrees, count + heading_actions, position_error, heading_error


def build_variant(
    env: habitat.Env,
    config: Any,
    entry: Dict[str, Any],
    source_episode: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    switch_scene(env, config, source_episode["scene_id"])
    env.sim.set_agent_state(
        np.array(source_episode["start_position"], dtype=np.float32),
        xyzw_to_quaternion(source_episode["start_rotation"]),
    )
    episode_id = int(entry["episode_id"])
    scan_id = str(entry["scan_id"])
    variant_id = f"{episode_id}_deviation"
    data_path = f"{scan_id}_r2r_{episode_id:06d}_deviation"
    writer = TrajectoryWriter(env.sim, args.images_root / data_path / "rgb", args.jpeg_quality, args.overwrite)
    writer.capture(0, "initial", -1)

    perturbation_plan = []
    segments = entry["instruction_segments"]
    base_actions = [int(action) for action in entry["continuous_gt"]["actions"]]
    segment_targets = [target_for_segment(entry, source_episode, index) for index in range(len(segments))]
    anchor_margin = max(0, int(args.anchor_boundary_margin_frames))
    for segment_index, segment in enumerate(segments):
        start = int(segment["start_frame"]) - 1
        end = int(segment["end_frame"])
        original_actions = [action for action in base_actions[start:end] if action != ACTION_STOP]
        segment_length = int(segment["end_frame"]) - int(segment["start_frame"]) + 1
        target = segment_targets[segment_index]
        target_yaw = target_yaw_for_segment(entry, source_episode, segment_index)
        foreign_waypoints = [waypoint for index, waypoint in enumerate(segment_targets) if index != segment_index]
        # A few human fine-grained segments consist only of an observation/
        # STOP boundary.  They still need one labeled output frame; otherwise
        # the derived segmentation would contain an empty interval.
        if not original_actions:
            writer.capture(segment_index, "static_zero_action_segment", ACTION_STOP)
            reconnected, reconnect_actions, arrival_distance, heading_error = reconnect(
                writer, segment_index, target, target_yaw, args.goal_radius, args.max_reconnect_actions,
                args.heading_tolerance_deg, args.max_heading_actions)
            if not reconnected:
                raise RuntimeError(
                    f"episode {episode_id}, segment {segment_index}: reconnect failed after {reconnect_actions} actions")
            perturbation_plan.append({
                "segment_index": segment_index,
                "applied": False,
                "reason": "zero_action_segment",
                "target_position": target,
                "target_yaw_degrees": target_yaw,
                "reconnect_actions": reconnect_actions,
                "arrival_distance_m": arrival_distance,
                "arrival_heading_error_degrees": heading_error,
            })
            continue
        # A perturbation anchor must sit in the semantic middle of its source
        # segment, rather than near a hand-annotated waypoint boundary.
        minimum_middle_length = max(args.min_segment_frames, 2 * anchor_margin + 2)
        if segment_length < minimum_middle_length or len(original_actions) < 2 * anchor_margin:
            for action in original_actions:
                writer.execute(action, segment_index, "original_short_segment")
            # Never let a short segment inherit position drift from the prior
            # synthetic segment.  As with perturbed segments, its endpoint is
            # explicitly reconnected with ShortestPathFollower.
            reconnected, reconnect_actions, arrival_distance, heading_error = reconnect(
                writer, segment_index, target, target_yaw, args.goal_radius, args.max_reconnect_actions,
                args.heading_tolerance_deg, args.max_heading_actions)
            if not reconnected:
                raise RuntimeError(
                    f"episode {episode_id}, segment {segment_index}: reconnect failed after {reconnect_actions} actions")
            perturbation_plan.append({
                "segment_index": segment_index,
                "applied": False,
                "reason": "segment_too_short_for_middle_anchor",
                "target_position": target,
                "target_yaw_degrees": target_yaw,
                "reconnect_actions": reconnect_actions,
                "arrival_distance_m": arrival_distance,
                "arrival_heading_error_degrees": heading_error,
            })
            continue

        prefix_count = len(original_actions) // 2
        prefix_count = max(anchor_margin, prefix_count)
        prefix_count = min(len(original_actions) - anchor_margin, prefix_count)
        source_anchor_frame = int(segment["start_frame"]) + prefix_count
        source_frames_before_anchor = source_anchor_frame - int(segment["start_frame"])
        source_frames_after_anchor = int(segment["end_frame"]) - source_anchor_frame
        for action in original_actions[:prefix_count]:
            writer.execute(action, segment_index, "original_prefix")

        selected_type: Optional[str] = None
        selected_actions: List[int] = []
        for perturbation_type, candidate in perturbation_candidates(episode_id, segment_index):
            if can_execute_perturbation(env.sim, candidate, target, foreign_waypoints, args.goal_radius):
                selected_type, selected_actions = perturbation_type, candidate
                break
        if selected_type is None:
            # A blocked perturbation must not leave a partial trajectory behind.
            for action in original_actions[prefix_count:]:
                writer.execute(action, segment_index, "original_perturbation_unavailable")
            reconnected, reconnect_actions, arrival_distance, heading_error = reconnect(
                writer, segment_index, target, target_yaw, args.goal_radius, args.max_reconnect_actions,
                args.heading_tolerance_deg, args.max_heading_actions)
            if not reconnected:
                raise RuntimeError(
                    f"episode {episode_id}, segment {segment_index}: reconnect failed after {reconnect_actions} actions")
            perturbation_plan.append({
                "segment_index": segment_index,
                "applied": False,
                "reason": "no_collision_free_rule_plan",
                "target_position": target,
                "target_yaw_degrees": target_yaw,
                "reconnect_actions": reconnect_actions,
                "arrival_distance_m": arrival_distance,
                "arrival_heading_error_degrees": heading_error,
            })
            continue

        anchor_frame = len(writer.records)
        for action in selected_actions:
            writer.execute(action, segment_index, f"perturb_{selected_type}")
        reconnected, reconnect_actions, arrival_distance, heading_error = reconnect(
            writer, segment_index, target, target_yaw, args.goal_radius, args.max_reconnect_actions,
            args.heading_tolerance_deg, args.max_heading_actions)
        if not reconnected:
            raise RuntimeError(
                f"episode {episode_id}, segment {segment_index}: reconnect failed after {reconnect_actions} actions")
        perturbation_plan.append({
            "segment_index": segment_index,
            "applied": True,
            "type": selected_type,
            "anchor_frame": anchor_frame,
            "source_anchor_frame": source_anchor_frame,
            "source_frames_before_anchor": source_frames_before_anchor,
            "source_frames_after_anchor": source_frames_after_anchor,
            "anchor_boundary_margin_frames": anchor_margin,
            "actions": selected_actions,
            "reconnect_actions": reconnect_actions,
            "target_position": target,
            "target_yaw_degrees": target_yaw,
            "foreign_waypoint_clearance_m": args.goal_radius,
            "arrival_distance_m": arrival_distance,
            "arrival_heading_error_degrees": heading_error,
        })

    # The final captured state is the terminal observation and therefore keeps STOP.
    writer.records[-1]["action"] = ACTION_STOP
    dynamic_segments = []
    for segment_index, segment in enumerate(segments):
        frame_ids = [record["frame"] for record in writer.records if record["segment_index"] == segment_index]
        if not frame_ids:
            raise RuntimeError(f"episode {episode_id}: output segment {segment_index} has no frames")
        start_frame, end_frame = min(frame_ids), max(frame_ids)
        dynamic_segments.append({
            "sub_instruction_index": segment_index,
            "sub_instruction": segment["sub_instruction"],
            "start_frame": start_frame,
            "end_frame": end_frame,
            "frame_range": f"{start_frame}-{end_frame}",
        })
    if any(left["end_frame"] + 1 != right["start_frame"] for left, right in zip(dynamic_segments, dynamic_segments[1:])):
        raise RuntimeError(f"episode {episode_id}: output segments are not contiguous")

    output = copy.deepcopy(entry)
    output.update({
        "episode_id": episode_id,
        "source_episode_id": episode_id,
        "variant_id": variant_id,
        "trajectory_uid": variant_id,
        "data_path": data_path,
        "video": None,
        "num_frames": len(writer.records),
        "instruction_segments": dynamic_segments,
        "sub_instructions": {str(item["sub_instruction_index"]): item["sub_instruction"] for item in dynamic_segments},
        "split_instructions": [item["sub_instruction"] for item in dynamic_segments],
        "segment_ranges": [
            {"segment_index": item["sub_instruction_index"], "start_frame": item["start_frame"], "end_frame": item["end_frame"]}
            for item in dynamic_segments
        ],
        "cut_points": {str(index): item["start_frame"] for index, item in enumerate(dynamic_segments)}
        | {str(len(dynamic_segments)): dynamic_segments[-1]["end_frame"]},
        "cut_point_ranges": {str(index): item["frame_range"] for index, item in enumerate(dynamic_segments)}
        | {str(len(dynamic_segments)): f"{dynamic_segments[-1]['end_frame']}-{dynamic_segments[-1]['end_frame']}"},
        "continuous_gt": {
            "actions": [record["action"] for record in writer.records],
            "coordinates": writer.records,
            "reference_path_endpoint_projection": copy.deepcopy(entry["continuous_gt"].get("reference_path_endpoint_projection", [])),
        },
        "perturbation_plan": perturbation_plan,
        "variant_spec": {
            "name": "r2r_deviation_rule_based",
            "min_segment_frames": args.min_segment_frames,
            "goal_radius_m": args.goal_radius,
            "description": "one deterministic rule-based perturbation per eligible segment, then shortest-path reconnect",
        },
    })
    return output


def main() -> None:
    args = parse_args()
    raw_segments = load_json(args.segmentation)
    entries = list(raw_segments.values()) if isinstance(raw_segments, dict) else list(raw_segments)
    selected_ids = {str(int(value)) for value in args.episode_id}
    episodes = {str(item["episode_id"]): item for item in load_json(args.episodes)["episodes"]}
    selected = []
    skipped_invalid = []
    for entry in entries:
        episode_id = str(int(entry["episode_id"]))
        if selected_ids and episode_id not in selected_ids:
            continue
        if not valid_segments(entry):
            skipped_invalid.append(int(episode_id))
            continue
        if episode_id not in episodes:
            skipped_invalid.append(int(episode_id))
            continue
        selected.append(entry)
    if args.max_episodes:
        selected = selected[:args.max_episodes]
    if not selected:
        raise SystemExit("No valid episodes selected.")
    # Reuse one Habitat scene instance for all of its episodes.  The released
    # split interleaves scans, whereas scene-grouped rendering avoids hundreds
    # of expensive simulator reloads without changing a variant's deterministic
    # trajectory or metadata.
    selected.sort(key=lambda item: (str(item["scene_id"]), int(item["episode_id"])))

    env, config = make_env(args.habitat_config, args.gpu_id)
    outputs: Dict[str, Dict[str, Any]] = {}
    failures = []
    try:
        for index, entry in enumerate(selected, start=1):
            episode_id = int(entry["episode_id"])
            try:
                output = build_variant(env, config, entry, episodes[str(episode_id)], args)
                outputs[output["variant_id"]] = output
                applied = sum(1 for item in output["perturbation_plan"] if item["applied"])
                print(f"[{index}/{len(selected)}] {output['variant_id']}: {output['num_frames']} frames, {applied} perturbations")
            except Exception as exc:
                failures.append({"episode_id": episode_id, "error": str(exc)})
                print(f"[{index}/{len(selected)}] {episode_id}: FAILED: {exc}", file=sys.stderr)
    finally:
        env.close()

    atomic_dump(outputs, args.output)
    report = {
        "selected_valid_episodes": len(selected),
        "written_variants": len(outputs),
        "skipped_invalid_segment_episodes": sorted(set(skipped_invalid)),
        "failures": failures,
        "output": str(args.output),
        "images_root": str(args.images_root),
    }
    atomic_dump(report, args.report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
