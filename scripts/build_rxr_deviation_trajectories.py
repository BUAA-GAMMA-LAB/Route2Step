#!/usr/bin/env python3
"""Build English RxR val-unseen deviation trajectories.

This is the RxR counterpart of ``build_r2r_deviation_trajectories.py``.
For every eligible landmark sub-instruction it replays the original GT prefix
to a middle anchor, inserts one deterministic heading/detour/backtrack/loop
perturbation, then uses Habitat's ``ShortestPathFollower`` to return to the
original segment endpoint and heading.  It never uses a language model.

RxR's released guide GT is represented in 15-degree primitive frames by
``collect_rxr_val_unseen_gt.py``.  During original-prefix replay this builder
snaps each post-action state back to that recorded GT state, matching the
existing unseen images (especially on stairs where navmesh replay can have a
small vertical drift).  Synthetic perturbation and reconnect states remain
actual Habitat states.
"""

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import quaternion

from build_r2r_deviation_trajectories import (
    ACTION_STOP,
    TrajectoryWriter,
    can_execute_perturbation,
    make_env,
    perturbation_candidates,
    reconnect,
    rotation_dict,
    switch_scene,
    valid_segments,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEGMENTATION = ROOT / "data/datasets/rxr/val_unseen/landmark_rxr_val_unseen_en_continuous.json"
DEFAULT_SOURCE = ROOT / "data/StreamVLN-Trajectory-Data/RxR/val_unseen_guide_en_annotations_with_coordinates.json"
DEFAULT_OUTPUT = ROOT / "data/datasets/rxr/val_unseen/landmark_rxr_val_unseen_en_deviation.json"
DEFAULT_IMAGES_ROOT = ROOT / "data/StreamVLN-Trajectory-Data/RxR/gt_val_unseen_deviation_images"
DEFAULT_REPORT = ROOT / "data/datasets/rxr/val_unseen/landmark_rxr_val_unseen_en_deviation_report.json"


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
    parser.add_argument("--source-annotations", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--images-root", type=Path, default=DEFAULT_IMAGES_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--habitat-config", type=Path, default=ROOT / "configs/vln_rxr_dual.yaml")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--episode-id", action="append", default=[])
    parser.add_argument("--max-episodes", type=int, default=0, help="0 means every valid English episode.")
    parser.add_argument("--min-segment-frames", type=int, default=10)
    parser.add_argument("--anchor-boundary-margin-frames", type=int, default=4)
    parser.add_argument("--goal-radius", type=float, default=0.25)
    parser.add_argument("--max-reconnect-actions", type=int, default=160)
    parser.add_argument("--heading-tolerance-deg", type=float, default=10.0)
    parser.add_argument("--max-heading-actions", type=int, default=24)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate and report selection without opening Habitat.")
    return parser.parse_args()


def wxyz_to_quaternion(values: Dict[str, Any]) -> np.quaternion:
    return np.quaternion(float(values["w"]), float(values["x"]), float(values["y"]), float(values["z"]))


def source_state(annotation: Dict[str, Any], frame_index: int) -> Tuple[List[float], np.quaternion]:
    coordinates = annotation["coordinates"]
    if not 0 <= frame_index < len(coordinates):
        raise IndexError(f"source frame index {frame_index} outside [0, {len(coordinates)})")
    coordinate = coordinates[frame_index]
    return [float(value) for value in coordinate["position"]], wxyz_to_quaternion(coordinate["rotation"])


def segment_target(annotation: Dict[str, Any], segment: Dict[str, Any], is_final: bool) -> Tuple[List[float], np.quaternion]:
    """Return the state after the segment's final next-action transition.

    Input frame ``f`` labels the action into frame ``f + 1``.  Non-final
    segment targets therefore use the following GT observation.  The final
    frame is STOP and stays at the last recorded state.
    """
    end_frame = int(segment["end_frame"])
    target_index = len(annotation["coordinates"]) - 1 if is_final else end_frame
    return source_state(annotation, target_index)


class SourceAlignedWriter(TrajectoryWriter):
    """Writer that snaps only original GT replay transitions to recorded RxR states."""

    def execute_source(
        self,
        action: int,
        segment_index: int,
        event: str,
        post_position: Sequence[float],
        post_rotation: np.quaternion,
    ) -> None:
        if not self.records:
            raise RuntimeError("initial state must be captured before executing an action")
        self.records[-1]["action"] = int(action)
        self.sim.step(int(action))
        self.sim.set_agent_state(np.array(post_position, dtype=np.float32), post_rotation)
        self.capture(segment_index, event, action)


def action_slice(entry: Dict[str, Any], segment: Dict[str, Any]) -> List[int]:
    start = int(segment["start_frame"]) - 1
    end = int(segment["end_frame"])
    return [int(action) for action in entry["continuous_gt"]["actions"][start:end] if int(action) != ACTION_STOP]


def validate_entry(entry: Dict[str, Any], annotation: Dict[str, Any]) -> Optional[str]:
    if not valid_segments(entry):
        return "invalid_or_noncontiguous_segments"
    source_coordinates = annotation.get("coordinates", [])
    actions = entry.get("continuous_gt", {}).get("actions", [])
    if len(source_coordinates) != int(entry.get("num_frames", -1)) or len(actions) != len(source_coordinates):
        return "segmentation_source_frame_count_mismatch"
    source_actions = [int(item.get("action", ACTION_STOP)) for item in source_coordinates]
    if [int(action) for action in actions] != source_actions:
        return "segmentation_source_action_mismatch"
    if not all("rotation" in item and "position" in item for item in source_coordinates):
        return "source_coordinates_missing_pose"
    return None


def build_variant(env: Any, config: Any, entry: Dict[str, Any], annotation: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    switch_scene(env, config, str(entry["scene_id"]))
    initial_position, initial_rotation = source_state(annotation, 0)
    env.sim.set_agent_state(np.array(initial_position, dtype=np.float32), initial_rotation)

    episode_id = int(entry["episode_id"])
    scan_id = str(entry["scan_id"])
    variant_id = f"{episode_id}_deviation"
    data_path = f"{scan_id}_rxr_{episode_id:06d}_deviation"
    writer = SourceAlignedWriter(env.sim, args.images_root / data_path / "rgb", args.jpeg_quality, args.overwrite)
    writer.capture(0, "initial", -1)

    segments = entry["instruction_segments"]
    segment_targets = [segment_target(annotation, segment, index == len(segments) - 1) for index, segment in enumerate(segments)]
    anchor_margin = max(0, int(args.anchor_boundary_margin_frames))
    perturbation_plan = []

    def replay_original(actions: Sequence[int], start_frame_index: int, segment_index: int, event: str) -> None:
        for offset, action in enumerate(actions):
            position, rotation = source_state(annotation, start_frame_index + offset + 1)
            writer.execute_source(action, segment_index, event, position, rotation)

    for segment_index, segment in enumerate(segments):
        start_frame_index = int(segment["start_frame"]) - 1
        segment_length = int(segment["end_frame"]) - int(segment["start_frame"]) + 1
        original_actions = action_slice(entry, segment)
        target_position, target_rotation = segment_targets[segment_index]
        target_yaw_rotation = rotation_dict(target_rotation)
        # ``reconnect`` accepts degrees, but the shared helper only needs the
        # numeric yaw.  Import lazily to keep the imported R2R utility surface
        # small and ensure both builders use identical yaw semantics.
        from build_r2r_deviation_trajectories import yaw_degrees
        target_yaw = yaw_degrees(target_rotation)
        foreign_waypoints = [position for index, (position, _) in enumerate(segment_targets) if index != segment_index]

        if not original_actions:
            writer.capture(segment_index, "static_zero_action_segment", ACTION_STOP)
            reconnected, reconnect_actions, arrival_distance, heading_error = reconnect(
                writer, segment_index, target_position, target_yaw, args.goal_radius,
                args.max_reconnect_actions, args.heading_tolerance_deg, args.max_heading_actions)
            if not reconnected:
                raise RuntimeError(f"episode {episode_id}, segment {segment_index}: reconnect failed")
            perturbation_plan.append({
                "segment_index": segment_index, "applied": False, "reason": "zero_action_segment",
                "target_position": target_position, "target_rotation": target_yaw_rotation,
                "target_yaw_degrees": target_yaw, "reconnect_actions": reconnect_actions,
                "arrival_distance_m": arrival_distance, "arrival_heading_error_degrees": heading_error,
            })
            continue

        minimum_middle_length = max(args.min_segment_frames, 2 * anchor_margin + 2)
        if segment_length < minimum_middle_length or len(original_actions) < 2 * anchor_margin:
            replay_original(original_actions, start_frame_index, segment_index, "original_short_segment")
            reconnected, reconnect_actions, arrival_distance, heading_error = reconnect(
                writer, segment_index, target_position, target_yaw, args.goal_radius,
                args.max_reconnect_actions, args.heading_tolerance_deg, args.max_heading_actions)
            if not reconnected:
                raise RuntimeError(f"episode {episode_id}, segment {segment_index}: reconnect failed")
            perturbation_plan.append({
                "segment_index": segment_index, "applied": False,
                "reason": "segment_too_short_for_middle_anchor", "target_position": target_position,
                "target_rotation": target_yaw_rotation, "target_yaw_degrees": target_yaw,
                "reconnect_actions": reconnect_actions, "arrival_distance_m": arrival_distance,
                "arrival_heading_error_degrees": heading_error,
            })
            continue

        prefix_count = min(len(original_actions) - anchor_margin, max(anchor_margin, len(original_actions) // 2))
        source_anchor_frame = int(segment["start_frame"]) + prefix_count
        replay_original(original_actions[:prefix_count], start_frame_index, segment_index, "original_prefix")

        selected_type: Optional[str] = None
        selected_actions: List[int] = []
        for perturbation_type, candidate in perturbation_candidates(episode_id, segment_index):
            if can_execute_perturbation(env.sim, candidate, target_position, foreign_waypoints, args.goal_radius):
                selected_type, selected_actions = perturbation_type, candidate
                break
        if selected_type is None:
            replay_original(original_actions[prefix_count:], start_frame_index + prefix_count, segment_index,
                            "original_perturbation_unavailable")
            reconnected, reconnect_actions, arrival_distance, heading_error = reconnect(
                writer, segment_index, target_position, target_yaw, args.goal_radius,
                args.max_reconnect_actions, args.heading_tolerance_deg, args.max_heading_actions)
            if not reconnected:
                raise RuntimeError(f"episode {episode_id}, segment {segment_index}: reconnect failed")
            perturbation_plan.append({
                "segment_index": segment_index, "applied": False, "reason": "no_collision_free_rule_plan",
                "target_position": target_position, "target_rotation": target_yaw_rotation,
                "target_yaw_degrees": target_yaw, "reconnect_actions": reconnect_actions,
                "arrival_distance_m": arrival_distance, "arrival_heading_error_degrees": heading_error,
            })
            continue

        anchor_frame = len(writer.records)
        for action in selected_actions:
            writer.execute(action, segment_index, f"perturb_{selected_type}")
        reconnected, reconnect_actions, arrival_distance, heading_error = reconnect(
            writer, segment_index, target_position, target_yaw, args.goal_radius,
            args.max_reconnect_actions, args.heading_tolerance_deg, args.max_heading_actions)
        if not reconnected:
            raise RuntimeError(f"episode {episode_id}, segment {segment_index}: reconnect failed")
        perturbation_plan.append({
            "segment_index": segment_index, "applied": True, "type": selected_type,
            "anchor_frame": anchor_frame, "source_anchor_frame": source_anchor_frame,
            "source_frames_before_anchor": source_anchor_frame - int(segment["start_frame"]),
            "source_frames_after_anchor": int(segment["end_frame"]) - source_anchor_frame,
            "anchor_boundary_margin_frames": anchor_margin, "actions": selected_actions,
            "reconnect_actions": reconnect_actions, "target_position": target_position,
            "target_rotation": target_yaw_rotation, "target_yaw_degrees": target_yaw,
            "foreign_waypoint_clearance_m": args.goal_radius, "arrival_distance_m": arrival_distance,
            "arrival_heading_error_degrees": heading_error,
        })

    writer.records[-1]["action"] = ACTION_STOP
    dynamic_segments = []
    for segment_index, segment in enumerate(segments):
        frame_ids = [record["frame"] for record in writer.records if record["segment_index"] == segment_index]
        if not frame_ids:
            raise RuntimeError(f"episode {episode_id}: output segment {segment_index} has no frames")
        dynamic_segments.append({
            "sub_instruction_index": segment_index, "sub_instruction": segment["sub_instruction"],
            "start_frame": min(frame_ids), "end_frame": max(frame_ids),
            "frame_range": f"{min(frame_ids)}-{max(frame_ids)}",
        })
    if any(left["end_frame"] + 1 != right["start_frame"] for left, right in zip(dynamic_segments, dynamic_segments[1:])):
        raise RuntimeError(f"episode {episode_id}: output segments are not contiguous")

    output = copy.deepcopy(entry)
    output.update({
        "episode_id": episode_id, "source_episode_id": episode_id, "variant_id": variant_id,
        "trajectory_uid": variant_id, "data_path": data_path,
        "video": f"gt_val_unseen_deviation_images/{data_path}", "num_frames": len(writer.records),
        "instruction_segments": dynamic_segments,
        "sub_instructions": {str(item["sub_instruction_index"]): item["sub_instruction"] for item in dynamic_segments},
        "split_instructions": [item["sub_instruction"] for item in dynamic_segments],
        "segment_ranges": [{"segment_index": item["sub_instruction_index"], "start_frame": item["start_frame"], "end_frame": item["end_frame"]} for item in dynamic_segments],
        "cut_points": {str(index): item["start_frame"] for index, item in enumerate(dynamic_segments)} | {str(len(dynamic_segments)): dynamic_segments[-1]["end_frame"]},
        "cut_point_ranges": {str(index): item["frame_range"] for index, item in enumerate(dynamic_segments)} | {str(len(dynamic_segments)): f"{dynamic_segments[-1]['end_frame']}-{dynamic_segments[-1]['end_frame']}"},
        "continuous_gt": {"actions": [record["action"] for record in writer.records], "coordinates": writer.records},
        "perturbation_plan": perturbation_plan,
        "variant_spec": {"name": "rxr_deviation_rule_based", "min_segment_frames": args.min_segment_frames,
                         "goal_radius_m": args.goal_radius,
                         "source_state_alignment": "published_rxr_guide_gt",
                         "description": "one deterministic rule-based perturbation per eligible segment, then shortest-path reconnect"},
    })
    return output


def main() -> None:
    args = parse_args()
    raw_entries = load_json(args.segmentation)
    entries = list(raw_entries.values()) if isinstance(raw_entries, dict) else list(raw_entries)
    annotations = {str(int(item["episode_id"])): item for item in load_json(args.source_annotations)}
    wanted = {str(int(value)) for value in args.episode_id}
    selected: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    skipped: List[Dict[str, Any]] = []
    for entry in entries:
        episode_id = str(int(entry["episode_id"]))
        if wanted and episode_id not in wanted:
            continue
        annotation = annotations.get(episode_id)
        reason = "missing_source_annotation" if annotation is None else validate_entry(entry, annotation)
        if reason:
            skipped.append({"episode_id": int(episode_id), "reason": reason})
        else:
            selected.append((entry, annotation))
    selected.sort(key=lambda pair: (str(pair[0]["scene_id"]), int(pair[0]["episode_id"])))
    if args.max_episodes:
        selected = selected[:args.max_episodes]
    if not selected:
        raise SystemExit("No valid English RxR unseen episodes selected.")
    if args.dry_run:
        print(json.dumps({"valid_episodes": len(selected), "skipped": skipped, "scenes": len({item[0]['scene_id'] for item in selected})}, ensure_ascii=False, indent=2))
        return

    env, config = make_env(args.habitat_config, args.gpu_id)
    outputs: Dict[str, Dict[str, Any]] = {}
    failures = []
    try:
        for index, (entry, annotation) in enumerate(selected, start=1):
            episode_id = int(entry["episode_id"])
            try:
                output = build_variant(env, config, entry, annotation, args)
                outputs[output["variant_id"]] = output
                applied = sum(item["applied"] for item in output["perturbation_plan"])
                print(f"[{index}/{len(selected)}] {output['variant_id']}: {output['num_frames']} frames, {applied} perturbations", flush=True)
            except Exception as exc:
                failures.append({"episode_id": episode_id, "error": str(exc)})
                print(f"[{index}/{len(selected)}] {episode_id}: FAILED: {exc}", file=sys.stderr, flush=True)
    finally:
        env.close()

    atomic_dump(outputs, args.output)
    report = {"selected_valid_episodes": len(selected), "written_variants": len(outputs),
              "skipped": skipped, "failures": failures, "output": str(args.output), "images_root": str(args.images_root)}
    atomic_dump(report, args.report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
