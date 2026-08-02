#!/usr/bin/env python3
"""Project human fine-grained R2R/RxR annotations to VLN-CE frames.

The source annotations are collected in Matterport's discrete viewpoint graph:

* FGR2R stores tokenized sub-instructions and inclusive 1-based ``chunk_view``
  intervals for an R2R path.
* Landmark-RxR stores human sub-instructions and their discrete ``sub_paths``.

This script maps each discrete sub-path endpoint to the corresponding waypoint
in the VLN-CE ``reference_path``, then to the closest frame of the continuous
oracle trajectory supplied by the split's ``*_gt`` file.  In particular, it
does *not* treat a viewpoint index as a continuous-frame index.

The output follows the repository's common segmentation schema.  By default it
also runs the existing text-aware turn-boundary API repair from
``fix_turn_boundary`` after the projection.  The API result is cached and can
be disabled explicitly when a deterministic projection-only output is needed.

Examples:
  # Projection with turn-boundary refinement (default).
  python seg/discrete_to_continuous/convert_human_finegrained_unseen_to_continuous.py --dataset all

  # Deterministic projection only, without the external API.
  python seg/discrete_to_continuous/convert_human_finegrained_unseen_to_continuous.py \
      --dataset all --no-turn-boundary-api

  # Project FGR2R train using the collected continuous train trajectories.
  python seg/discrete_to_continuous/convert_human_finegrained_unseen_to_continuous.py \\
      --dataset r2r --r2r-split train

  # Reuse / extend a cached API analysis for turn spans.
  python seg/discrete_to_continuous/convert_human_finegrained_unseen_to_continuous.py \\
      --dataset rxr --turn-boundary-api --api-keys-file path/to/api_keys.txt \\
      --api-base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \\
      --api-model qwen3-vl-plus
"""

import argparse
import ast
import gzip
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]

# val_unseen has the official VLN-CE oracle actions/locations.  Train uses the
# project's collected continuous oracle trajectories, whose episode IDs cover
# precisely R2R-VLNCE train (see all_coordinates.json).
R2R_CONFIGS = {
    "val_unseen": {
        "fgr": ROOT / "data/datasets/FGR2R/FGR2R_val_unseen.json",
        "split": ROOT / "data/datasets/R2R_VLNCE_v1-3/val_unseen/val_unseen.json",
        "gt": ROOT / "data/datasets/R2R_VLNCE_v1-3/val_unseen/val_unseen_gt.json",
        "output": ROOT / "data/datasets/R2R_VLNCE_v1-3/val_unseen/fgr2r_val_unseen_continuous.json",
        "report": ROOT / "seg/outputs/human_finegrained_unseen/fgr2r_val_unseen_projection_report.json",
        "cache": ROOT / "seg/outputs/human_finegrained_unseen/fgr2r_val_unseen_turn_boundary_cache.json",
        "state_source": "vlnce_gt",
    },
    "train": {
        "fgr": ROOT / "data/datasets/FGR2R/FGR2R_train.json",
        "split": ROOT / "data/datasets/R2R_VLNCE_v1-3/train/train.json.gz",
        "coordinates": ROOT / "data/StreamVLN-Trajectory-Data/R2R/all_coordinates.json",
        "output": ROOT / "data/datasets/R2R_VLNCE_v1-3/train/fgr2r_train_continuous.json",
        "report": ROOT / "seg/outputs/human_finegrained_train/fgr2r_train_projection_report.json",
        "cache": ROOT / "seg/outputs/human_finegrained_train/fgr2r_train_turn_boundary_cache.json",
        "state_source": "collected_coordinates",
    },
}

RXR_LANDMARK = ROOT / "data/datasets/Landmark-RxR/LandmarkRxR_val_unseen.json"
RXR_SPLIT = ROOT / "data/datasets/rxr/val_unseen/val_unseen_guide.json"
RXR_GT = ROOT / "data/datasets/rxr/val_unseen/val_unseen_guide_gt.json.gz"
RXR_IMAGES_ROOT = ROOT / "data/StreamVLN-Trajectory-Data/RxR"
RXR_COLLECTED_MANIFEST = RXR_IMAGES_ROOT / "val_unseen_guide_en_annotations_with_coordinates.json"
RXR_TRAJECTORY_FILENAME = "gt_rxr_val_unseen_trajectory.json"
RXR_OUTPUT = ROOT / "data/datasets/rxr/val_unseen/landmark_rxr_val_unseen_en_continuous.json"
RXR_REPORT = ROOT / "seg/outputs/human_finegrained_unseen/landmark_rxr_val_unseen_en_projection_report.json"
RXR_CACHE = ROOT / "seg/outputs/human_finegrained_unseen/landmark_rxr_val_unseen_en_turn_boundary_cache.json"

FORWARD_ACTION = 1


def load_json(path: Path) -> Any:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def atomic_dump(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(temporary, path)


def episode_key(value: Any) -> str:
    text = str(value).strip()
    if not text:
        return ""
    try:
        return str(int(text))
    except ValueError:
        return text


def normalize_text(text: Any) -> str:
    return " ".join(str(text).split()).lower()


def scan_from_scene(scene_id: str) -> str:
    parts = str(scene_id).split("/")
    return parts[1] if len(parts) >= 2 else ""


def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((float(x) - float(y))**2 for x, y in zip(a, b)))


def materialize_positions(episode: Dict[str, Any], gt: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expand forward-only GT locations to one state per oracle action.

    This deliberately mirrors ``materialize_r2r_vlnce_val_unseen_frame_states``:
    the first GT location is the initial state, and every later forward action
    consumes the next location.  Turn/stop actions retain the current position.
    """
    actions = [int(action) for action in gt.get("actions", [])]
    locations = gt.get("locations", [])
    if not actions or not locations:
        raise ValueError(f"episode {episode_key(episode.get('episode_id'))}: missing actions or locations")

    expected_forward = int(gt.get("forward_steps", sum(action == FORWARD_ACTION for action in actions)))
    actual_forward = sum(action == FORWARD_ACTION for action in actions)
    if actual_forward != expected_forward:
        raise ValueError(
            f"episode {episode_key(episode.get('episode_id'))}: forward_steps "
            f"gt={expected_forward}, actions={actual_forward}")

    position = [float(value) for value in locations[0]]
    loc_index = 0
    states = []
    for index, action in enumerate(actions, start=1):
        if action == FORWARD_ACTION and loc_index + 1 < len(locations):
            loc_index += 1
            position = [float(value) for value in locations[loc_index]]
        # ``frame`` is used by the projection helper; ``step`` keeps the
        # records directly compatible with seg.fix_turn_boundary's API pass.
        states.append({"frame": index, "step": index, "action": action, "position": position.copy()})

    if loc_index != len(locations) - 1:
        raise ValueError(
            f"episode {episode_key(episode.get('episode_id'))}: consumed {loc_index + 1}/"
            f"{len(locations)} GT locations")
    return states


def normalize_collected_steps(raw_steps: Sequence[Any], episode_id: str) -> List[int]:
    """Normalize a collected trajectory's step labels to contiguous 1-based frames.

    The RxR train coordinate release contains two internally consistent
    conventions: most records use ``1..N`` and a subset uses ``0..N-1``.
    Keep the source file unchanged and normalize only at this reproducibility
    entry point, so every projected output uses the common 1-based frame
    convention.
    """
    steps = [int(step) for step in raw_steps]
    if steps == list(range(1, len(steps) + 1)):
        return steps
    if steps == list(range(len(steps))):
        return [step + 1 for step in steps]
    raise ValueError(f"episode {episode_id}: collected coordinate steps are neither contiguous 0- nor 1-based frames")


def states_from_collected_coordinates(record: Dict[str, Any], episode_id: str) -> List[Dict[str, Any]]:
    """Normalize the project's collected train oracle trajectory to frame states."""
    coordinates = record.get("coordinates", [])
    if not isinstance(coordinates, list) or not coordinates:
        raise ValueError(f"episode {episode_id}: missing collected coordinates")
    states = []
    normalized_steps = normalize_collected_steps(
        [coordinate.get("step") for coordinate in coordinates], episode_id)
    for coordinate, frame in zip(coordinates, normalized_steps):
        try:
            position = [float(value) for value in coordinate["position"]]
            action = int(coordinate["action"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"episode {episode_id}: invalid collected coordinate") from exc
        states.append({"frame": frame, "step": frame, "action": action, "position": position})
    return states


def closest_frame(
    states: Sequence[Dict[str, Any]],
    target_position: Sequence[float],
    minimum_frame: int,
) -> Tuple[int, float]:
    candidates = [state for state in states if int(state["frame"]) >= minimum_frame]
    if not candidates:
        raise ValueError("no continuous frames remain for a sub-path endpoint")
    state = min(candidates, key=lambda item: euclidean(item["position"], target_position))
    return int(state["frame"]), euclidean(state["position"], target_position)


def endpoint_indices_from_ranges(ranges: Iterable[Sequence[Any]], path_length: int) -> List[int]:
    endpoints = []
    for interval in ranges:
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            raise ValueError(f"invalid inclusive sub-path interval: {interval!r}")
        end_index = int(interval[1]) - 1  # FGR2R's chunk_view is 1-based.
        if not 0 <= end_index < path_length:
            raise ValueError(f"endpoint {end_index + 1} outside a path of length {path_length}")
        endpoints.append(end_index)
    return endpoints


def endpoint_indices_from_subpaths(path: Sequence[str], sub_paths: Iterable[Sequence[str]]) -> List[int]:
    """Map Landmark-RxR sub-path ends to monotonic indices in its full path."""
    endpoints = []
    previous_end = 0
    for sub_path in sub_paths:
        if not isinstance(sub_path, (list, tuple)) or not sub_path:
            raise ValueError(f"invalid Landmark-RxR sub_path: {sub_path!r}")
        endpoint = str(sub_path[-1])
        matches = [index for index, viewpoint in enumerate(path) if viewpoint == endpoint and index >= previous_end]
        if not matches:
            raise ValueError(f"sub_path endpoint {endpoint!r} cannot be aligned monotonically")
        endpoint_index = matches[0]
        endpoints.append(endpoint_index)
        previous_end = endpoint_index
    return endpoints


def build_segments(
    sub_instructions: Sequence[str],
    endpoint_indices: Sequence[int],
    reference_path: Sequence[Sequence[float]],
    states: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not sub_instructions or len(sub_instructions) != len(endpoint_indices):
        raise ValueError("sub-instruction and discrete-subpath counts differ")
    if len(reference_path) == 0:
        raise ValueError("empty continuous reference_path")

    # Each annotation sub-path ends at a discrete viewpoint.  Only internal
    # endpoints determine a cut: the final segment always includes the stop.
    boundary_ends = []
    endpoint_records = []
    min_frame = 1
    for segment_index, endpoint_index in enumerate(endpoint_indices[:-1]):
        if not 0 <= endpoint_index < len(reference_path):
            raise ValueError(f"reference-path endpoint {endpoint_index} is out of range")
        frame, distance = closest_frame(states, reference_path[endpoint_index], min_frame)
        # A non-final segment must leave one frame for every following segment.
        maximum = len(states) - (len(sub_instructions) - segment_index - 1)
        frame = min(frame, maximum)
        frame = max(frame, min_frame)
        boundary_ends.append(frame)
        endpoint_records.append({
            "segment_index": segment_index,
            "discrete_endpoint_index": endpoint_index + 1,
            "continuous_frame": frame,
            "distance_to_reference_path_m": distance,
        })
        min_frame = frame + 1

    starts = [1] + [frame + 1 for frame in boundary_ends]
    ends = boundary_ends + [len(states)]
    segments = []
    for index, (text, start, end) in enumerate(zip(sub_instructions, starts, ends)):
        if start > end:
            raise ValueError(f"empty segment {index}: {start}-{end}")
        segments.append({
            "sub_instruction_index": index,
            "sub_instruction": str(text).strip(),
            "start_frame": start,
            "end_frame": end,
            "frame_range": f"{start}-{end}",
        })
    return segments, endpoint_records


def segmentation_entry(
    episode: Dict[str, Any],
    source: Dict[str, Any],
    source_name: str,
    sub_instructions: Sequence[str],
    endpoint_indices: Sequence[int],
    states: Sequence[Dict[str, Any]],
    video: Optional[str] = None,
) -> Dict[str, Any]:
    segments, endpoint_records = build_segments(
        sub_instructions, endpoint_indices, episode["reference_path"], states)
    final_frame = len(states)
    cut_points = {str(index): segment["start_frame"] for index, segment in enumerate(segments)}
    # This N+1 item is retained for compatibility with the existing Landmark-RxR
    # mapping, where it records the final frame rather than a new segment start.
    cut_points[str(len(segments))] = final_frame
    scan_id = scan_from_scene(episode.get("scene_id", ""))
    episode_id = episode_key(episode.get("episode_id"))
    dataset_token = "r2r" if source_name == "fgr2r" else "rxr"
    data_path = f"{scan_id}_{dataset_token}_{int(episode_id):06d}" if episode_id.isdigit() else episode_id
    return {
        "episode_id": int(episode_id) if episode_id.isdigit() else episode_id,
        "trajectory_id": episode.get("trajectory_id"),
        "scan_id": scan_id,
        "scene_id": episode.get("scene_id", ""),
        "language": episode.get("instruction", {}).get("language", "en"),
        "original_instruction": episode.get("instruction", {}).get("instruction_text", ""),
        "instruction": episode.get("instruction", {}).get("instruction_text", ""),
        "sub_instructions": {str(i): segment["sub_instruction"] for i, segment in enumerate(segments)},
        "split_instructions": [segment["sub_instruction"] for segment in segments],
        "cut_points": cut_points,
        "cut_point_ranges": {
            str(i): segment["frame_range"] for i, segment in enumerate(segments)
        } | {str(len(segments)): f"{final_frame}-{final_frame}"},
        "segment_ranges": [
            {"segment_index": i, "start_frame": segment["start_frame"], "end_frame": segment["end_frame"]}
            for i, segment in enumerate(segments)
        ],
        "instruction_segments": segments,
        "num_frames": final_frame,
        "data_path": data_path,
        "video": video,
        "continuous_gt": {
            "actions": [state["action"] for state in states],
            "coordinates": states,
            "reference_path_endpoint_projection": endpoint_records,
        },
        "source_annotation": source,
        "source_dataset": source_name,
    }


def load_rxr_collected_states(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Read the extracted 15° RxR unseen frames used for segmentation.

    The official guide GT serializes 30° turns, while the collected unseen
    trajectories match the project's train convention: initial observation,
    then 15° primitive frames, with each frame labelling its next action.
    """
    video = str(record.get("video", ""))
    metadata_path = RXR_IMAGES_ROOT / video / "rgb" / RXR_TRAJECTORY_FILENAME
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing extracted RxR trajectory metadata: {metadata_path}")
    metadata = load_json(metadata_path)
    coordinates = metadata.get("coordinates", [])
    if not coordinates:
        raise ValueError(f"no coordinates in {metadata_path}")
    states = []
    normalized_steps = normalize_collected_steps(
        [coordinate.get("step") for coordinate in coordinates], str(record.get("video", "")))
    for coordinate, step in zip(coordinates, normalized_steps):
        states.append({
            "frame": step,
            "step": step,
            "action": int(coordinate["action"]),
            "position": [float(value) for value in coordinate["position"]],
        })
    expected = list(range(1, len(states) + 1))
    if [state["frame"] for state in states] != expected:
        raise ValueError(f"non-contiguous collected frames in {metadata_path}")
    return states


def parse_fgr_subinstructions(value: Any, instruction_index: int) -> List[str]:
    parsed = ast.literal_eval(value) if isinstance(value, str) else value
    if not isinstance(parsed, list) or instruction_index >= len(parsed):
        raise ValueError(f"invalid FGR2R new_instructions: {value!r}")
    chunks = parsed[instruction_index]
    if not isinstance(chunks, list):
        raise ValueError("FGR2R instruction chunks are not a list")
    texts = [" ".join(map(str, chunk)).strip() for chunk in chunks]
    if not all(texts):
        raise ValueError("FGR2R contains an empty sub-instruction")
    return texts


def build_r2r_entries(split_name: str) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    config = R2R_CONFIGS[split_name]
    fgr_items = load_json(config["fgr"])
    episodes = load_json(config["split"]).get("episodes", [])
    state_source = str(config["state_source"])
    if state_source == "vlnce_gt":
        state_by_episode = load_json(config["gt"])
    elif state_source == "collected_coordinates":
        state_by_episode = {
            episode_key(item.get("r2r_id")): item for item in load_json(config["coordinates"])
        }
    else:
        raise ValueError(f"unsupported R2R state source: {state_source}")

    by_path = {str(item["path_id"]): item for item in fgr_items}
    result, errors, skipped_missing_fgr_segmentation = {}, [], []
    for episode in episodes:
        ep_id = episode_key(episode.get("episode_id"))
        fgr = by_path.get(str(episode.get("trajectory_id")))
        state_record = state_by_episode.get(ep_id)
        if not fgr or not isinstance(state_record, dict):
            errors.append({"episode_id": ep_id, "error": "missing_fgr_or_continuous_trajectory"})
            continue
        instruction_text = normalize_text(episode["instruction"]["instruction_text"])
        matched_indices = [
            index for index, text in enumerate(fgr.get("instructions", [])) if normalize_text(text) == instruction_text
        ]
        if len(matched_indices) != 1:
            errors.append({"episode_id": ep_id, "error": "instruction_not_unique_in_fgr", "matches": matched_indices})
            continue
        instruction_index = matched_indices[0]
        try:
            parsed_subinstructions = ast.literal_eval(fgr["new_instructions"])
            # The official train release has ten R2R paths with a fourth raw
            # instruction but annotations for only the first three.  They have
            # no FGR2R supervision, rather than a bad continuous projection.
            if not isinstance(parsed_subinstructions, list) or instruction_index >= len(parsed_subinstructions):
                skipped_missing_fgr_segmentation.append({
                    "episode_id": ep_id,
                    "trajectory_id": episode.get("trajectory_id"),
                    "instruction_index": instruction_index,
                    "fgr_instruction_count": len(fgr.get("instructions", [])),
                    "fgr_segmented_instruction_count": (
                        len(parsed_subinstructions) if isinstance(parsed_subinstructions, list) else None
                    ),
                })
                continue
            sub_instructions = parse_fgr_subinstructions(fgr["new_instructions"], instruction_index)
            endpoint_indices = endpoint_indices_from_ranges(fgr["chunk_view"][instruction_index], len(fgr["path"]))
            states = (
                materialize_positions(episode, state_record)
                if state_source == "vlnce_gt"
                else states_from_collected_coordinates(state_record, ep_id)
            )
            result[ep_id] = segmentation_entry(
                episode,
                {"path_id": fgr["path_id"], "instruction_index": instruction_index, "chunk_view": fgr["chunk_view"][instruction_index]},
                "fgr2r",
                sub_instructions,
                endpoint_indices,
                states,
            )
        except (ValueError, IndexError, SyntaxError) as exc:
            errors.append({"episode_id": ep_id, "error": str(exc)})
    report = {
        "dataset": f"FGR2R {split_name} -> R2R-VLNCE v1-3 {split_name}",
        "continuous_state_source": state_source,
        "source_entries": len(fgr_items),
        "target_episodes": len(episodes),
        "converted_episodes": len(result),
        "skipped_missing_fgr_segmentation": skipped_missing_fgr_segmentation,
        "errors": errors,
    }
    return result, errors, report


def build_rxr_entries() -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    landmark_items = load_json(RXR_LANDMARK)
    episodes = load_json(RXR_SPLIT).get("episodes", [])
    collected_manifest = load_json(RXR_COLLECTED_MANIFEST)
    english_episodes = [
        item for item in episodes if str(item.get("instruction", {}).get("language", "")).startswith("en-")
    ]
    by_instruction_id = {
        str(item["instruction"]["instruction_id"]): item for item in english_episodes
    }
    collected_by_episode = {
        episode_key(item.get("episode_id")): item for item in collected_manifest
    }

    result, errors = {}, []
    for landmark in landmark_items:
        instruction_id = str(landmark.get("instruction_id"))
        episode = by_instruction_id.get(instruction_id)
        if episode is None:
            continue  # Landmark-RxR has 882 non-VLNCE-portable English samples.
        ep_id = episode_key(episode.get("episode_id"))
        collected = collected_by_episode.get(ep_id)
        if not isinstance(collected, dict):
            errors.append({"episode_id": ep_id, "instruction_id": instruction_id, "error": "missing_collected_gt"})
            continue
        try:
            sub_instructions = [str(value).strip() for value in landmark["sub_instructions"]]
            endpoint_indices = endpoint_indices_from_subpaths(landmark["path"], landmark["sub_paths"])
            states = load_rxr_collected_states(collected)
            result[ep_id] = segmentation_entry(
                episode,
                {"instruction_id": landmark["instruction_id"], "path_id": landmark["path_id"], "sub_paths": landmark["sub_paths"]},
                "landmark_rxr",
                sub_instructions,
                endpoint_indices,
                states,
                video=str(collected["video"]),
            )
        except (ValueError, IndexError, OSError) as exc:
            errors.append({"episode_id": ep_id, "instruction_id": instruction_id, "error": str(exc)})
    report = {
        "dataset": "Landmark-RxR val_unseen English -> RxR-VLNCE val_unseen guide",
        "source_english_entries": len(landmark_items),
        "target_english_episodes": len(english_episodes),
        "collected_english_episodes": len(collected_by_episode),
        "source_entries_matching_vlnce": len(result) + len(errors),
        "converted_episodes": len(result),
        "errors": errors,
    }
    return result, errors, report


def add_distance_summary(report: Dict[str, Any], entries: Dict[str, Dict[str, Any]]) -> None:
    distances = [
        item["distance_to_reference_path_m"]
        for entry in entries.values()
        for item in entry["continuous_gt"]["reference_path_endpoint_projection"]
    ]
    if distances:
        values = sorted(distances)
        report["internal_endpoint_projection_distance_m"] = {
            "count": len(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "p95": values[min(len(values) - 1, math.ceil(0.95 * len(values)) - 1)],
            "over_0_5m": sum(value > 0.5 for value in values),
        }


def run_turn_boundary_api(
    entries: Dict[str, Dict[str, Any]],
    output_path: Path,
    cache_path: Path,
    args: argparse.Namespace,
) -> None:
    """Apply the repository's established API decision rules to projected frames."""
    try:
        from seg.discrete_to_continuous import fix_turn_boundary as turn_fix
    except ModuleNotFoundError:
        # Support both ``python -m seg...`` and direct script execution.
        import fix_turn_boundary as turn_fix

    coordinates = {
        entry["data_path"]: entry["continuous_gt"]["coordinates"] for entry in entries.values()
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    turn_fix.OUTPUT_FILE = str(output_path)
    turn_fix.fix_boundaries_with_api(
        entries,
        coordinates,
        args.api_base_url,
        args.api_model,
        args.api_keys_file,
        str(cache_path),
        args.api_max_tokens,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("r2r", "rxr", "all"), default="all")
    parser.add_argument(
        "--r2r-split",
        choices=tuple(R2R_CONFIGS),
        default="val_unseen",
        help="R2R split to project; train uses collected continuous coordinates.",
    )
    parser.add_argument(
        "--turn-boundary-api",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run text-aware API repair for same-position turn spans (default: enabled).",
    )
    parser.add_argument("--api-base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--api-model", default="qwen3-vl-plus")
    parser.add_argument("--api-keys-file", default="path/to/api_keys.txt")
    parser.add_argument("--api-max-tokens", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jobs = []
    if args.dataset in {"r2r", "all"}:
        r2r_config = R2R_CONFIGS[args.r2r_split]
        jobs.append((
            f"r2r_{args.r2r_split}",
            lambda: build_r2r_entries(args.r2r_split),
            r2r_config["output"],
            r2r_config["report"],
            r2r_config["cache"],
        ))
    if args.dataset in {"rxr", "all"}:
        jobs.append(("rxr", build_rxr_entries, RXR_OUTPUT, RXR_REPORT, RXR_CACHE))

    for name, builder, output_path, report_path, cache_path in jobs:
        entries, _, report = builder()
        add_distance_summary(report, entries)
        if report["errors"]:
            atomic_dump(report, report_path)
            raise RuntimeError(f"{name}: {len(report['errors'])} failed projections; see {report_path}")
        atomic_dump(entries, output_path)
        if args.turn_boundary_api:
            run_turn_boundary_api(entries, output_path, cache_path, args)
        atomic_dump(report, report_path)
        print(
            f"{name}: wrote {output_path} | episodes={len(entries)} | "
            f"turn_api={args.turn_boundary_api}")
        print(json.dumps(report["internal_endpoint_projection_distance_m"], ensure_ascii=False))


if __name__ == "__main__":
    main()
