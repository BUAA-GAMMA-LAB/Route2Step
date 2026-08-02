#!/usr/bin/env python3
"""
Shared dataset helpers for stair-specific segmentation workflows.
"""
import json
import math
import os
from typing import Any, Dict, List, Optional


STAIR_DATASET_CONFIGS: Dict[str, Dict[str, str]] = {
    "r2r": {
        "dataset_token": "r2r",
        "input_file": "data/StreamVLN-Trajectory-Data/R2R/r2r_train_seg.json",
        "audit_output_file": "seg/outputs/stairs/r2r_train_stair_subinstruction_audit.json",
        "rewrite_file": "seg/outputs/stairs/r2r_train_stair_subinstruction_episode_rewrites.json",
        "rewritten_output_file": "data/StreamVLN-Trajectory-Data/R2R/r2r_train_seg_stair_rewritten.json",
        "coord_file": "data/StreamVLN-Trajectory-Data/R2R/all_coordinates.json",
        "annotation_file": "data/StreamVLN-Trajectory-Data/R2R/annotations.json",
        "images_base_dir": "data/StreamVLN-Trajectory-Data/R2R",
        "action_map_file": "seg/outputs/actions/sub_instruction_actions_r2r_train.json",
        "segmentation_output_file": "seg/outputs/segmentation/seg_r2r_train.json",
        "stair_only_output_file": "seg/outputs/stairs/seg_r2r_train_stairs.json",
        "stair_filter_cache": "seg/outputs/stairs/r2r_train_stair_episode_filter.json",
        "stair_alignment_file": "seg/outputs/stairs/r2r_train_stair_alignment_api.json",
        "stair_compare_markdown_file": "seg/outputs/stairs/seg_r2r_train_stairs_links.md",
        "per_episode_output_filename": "",
        "coordinate_lookup_field": "coordinate_key",
    },
}


def get_stair_dataset_config(dataset: str) -> Dict[str, str]:
    if dataset not in STAIR_DATASET_CONFIGS:
        raise ValueError(f"Unsupported stair dataset: {dataset}")
    return dict(STAIR_DATASET_CONFIGS[dataset])


def normalize_episode_key(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        return str(int(text))
    except Exception:
        return text


def episode_sort_key(value: Any):
    key = normalize_episode_key(value)
    if key.isdigit():
        return (0, int(key))
    return (1, key)


def _coerce_episode_id(value: Any) -> Any:
    key = normalize_episode_key(value)
    if not key:
        return value
    try:
        return int(key)
    except Exception:
        return key


def _ordered_instruction_values(values: Any) -> List[str]:
    if isinstance(values, dict):
        ordered_keys = sorted(
            values.keys(),
            key=lambda x: int(x) if str(x).isdigit() else str(x),
        )
        return [str(values[k]) for k in ordered_keys if str(values[k]).strip()]
    if isinstance(values, list):
        return [str(item) for item in values if str(item).strip()]
    return []


def infer_scan_id(item: Dict[str, Any]) -> str:
    scan_id = str(item.get("scan_id", "")).strip()
    if scan_id:
        return scan_id

    scene_id = str(item.get("scene_id", "")).strip()
    if scene_id:
        base = os.path.basename(scene_id)
        if base.endswith(".glb"):
            base = base[:-4]
        if base:
            return base

    for field in ("data_path", "video"):
        raw = str(item.get(field, "")).strip()
        if not raw:
            continue
        base = os.path.basename(raw)
        for token in ("_rxr_", "_r2r_", "_scalevln_"):
            if token in base:
                return base.split(token, 1)[0]
        stem, _ = os.path.splitext(base)
        if stem:
            return stem
    return ""


def infer_data_path(item: Dict[str, Any], dataset: str) -> str:
    existing = str(item.get("data_path", "")).strip()
    if existing:
        return existing

    video = str(item.get("video", "")).strip()
    if video:
        return os.path.basename(video)

    episode_id = normalize_episode_key(item.get("episode_id", item.get("id")))
    scan_id = infer_scan_id(item)
    if not episode_id or not scan_id:
        return ""

    try:
        suffix = f"{int(episode_id):06d}"
    except Exception:
        suffix = episode_id
    return f"{scan_id}_{get_stair_dataset_config(dataset)['dataset_token']}_{suffix}"


def infer_scene_id(item: Dict[str, Any]) -> str:
    scene_id = str(item.get("scene_id", "")).strip()
    if scene_id:
        return scene_id

    scan_id = infer_scan_id(item)
    if not scan_id:
        return ""
    return f"mp3d/{scan_id}/{scan_id}.glb"


def normalize_episode_item(item: Dict[str, Any], dataset: str) -> Dict[str, Any]:
    dataset_cfg = get_stair_dataset_config(dataset)
    normalized = dict(item)
    episode_id = _coerce_episode_id(item.get("episode_id", item.get("id")))
    normalized["episode_id"] = episode_id
    normalized["scan_id"] = infer_scan_id(item)
    normalized["scene_id"] = infer_scene_id(item)
    normalized["data_path"] = infer_data_path(item, dataset)
    normalized["annotation_key"] = normalize_episode_key(
        item.get(dataset_cfg.get("annotation_lookup_field", "episode_id"), episode_id)
    )
    normalized["coordinate_key"] = normalize_episode_key(
        item.get(dataset_cfg.get("coordinate_lookup_field", "episode_id"), episode_id)
    )

    original_instruction = item.get("original_instruction", item.get("instruction", ""))
    if isinstance(original_instruction, dict):
        original_instruction = original_instruction.get("instruction_text", "")
    normalized["original_instruction"] = str(original_instruction)

    sub_instructions = _ordered_instruction_values(
        item.get("sub_instructions", item.get("split_instructions", []))
    )
    normalized["sub_instructions"] = sub_instructions
    return normalized


def load_episode_dataset(path: str, dataset: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    items: List[Dict[str, Any]]
    if isinstance(raw, list):
        items = [item for item in raw if isinstance(item, dict)]
    elif isinstance(raw, dict) and isinstance(raw.get("episodes"), list):
        items = [item for item in raw["episodes"] if isinstance(item, dict)]
    elif isinstance(raw, dict):
        items = []
        for episode_id, item in raw.items():
            if not isinstance(item, dict):
                continue
            merged = dict(item)
            merged.setdefault("episode_id", episode_id)
            items.append(merged)
    else:
        raise ValueError(f"Unsupported episode payload in {path}")

    normalized = [normalize_episode_item(item, dataset) for item in items]
    normalized.sort(key=lambda item: episode_sort_key(item.get("episode_id")))
    return normalized


def load_annotation_index(path: str) -> Dict[str, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    result: Dict[str, Dict[str, Any]] = {}
    if isinstance(raw, dict):
        iterator = raw.items()
    else:
        iterator = enumerate(raw if isinstance(raw, list) else [])
    for raw_key, item in iterator:
        if not isinstance(item, dict):
            continue
        key = normalize_episode_key(
            item.get("episode_id", item.get("id", item.get("r2r_id")))
        )
        if not key and not isinstance(raw_key, int):
            key = normalize_episode_key(raw_key)
        if not key:
            continue
        normalized_item = dict(item)
        normalized_item.setdefault("episode_id", key)
        result[key] = normalized_item
    return result


def expand_locations_by_actions(actions: List[Any], locations: List[Any]) -> List[Any]:
    if not isinstance(actions, list) or not isinstance(locations, list) or not locations:
        return []
    expanded = []
    loc_idx = 0
    current = locations[loc_idx]
    for action in actions:
        try:
            action_id = int(action)
        except Exception:
            action_id = action
        if action_id == 1 and loc_idx + 1 < len(locations):
            loc_idx += 1
            current = locations[loc_idx]
        expanded.append(current)
    return expanded


def load_coordinate_map(path: str) -> Dict[str, List[dict]]:
    result: Dict[str, List[dict]] = {}
    for key, item in load_annotation_index(path).items():
        coordinates = item.get("coordinates")
        if coordinates is None:
            coordinates = item.get("trajectory")
        if coordinates is None and isinstance(item.get("locations"), list):
            actions = item.get("actions", [])
            locations = item.get("locations", [])
            expanded_locations = expand_locations_by_actions(actions, locations)
            coordinates = [
                {
                    "step": idx + 1,
                    "position": position,
                    "action": actions[idx] if idx < len(actions) else None,
                }
                for idx, position in enumerate(expanded_locations)
            ]
        if isinstance(coordinates, list):
            result[key] = coordinates
    return result


def build_rgb_dir(item: Dict[str, Any], images_base_dir: str, dataset: str) -> str:
    data_path = infer_data_path(item, dataset)
    if not data_path:
        return ""
    dataset_cfg = get_stair_dataset_config(dataset)
    if dataset_cfg.get("images_layout") == "direct":
        return os.path.join(images_base_dir, data_path, "rgb")
    return os.path.join(images_base_dir, "images", data_path, "rgb")


def _is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _normalize_position(position: Any) -> Optional[List[float]]:
    if not isinstance(position, (list, tuple)) or len(position) != 3:
        return None
    if not all(_is_finite_number(v) for v in position):
        return None
    return [float(position[0]), float(position[1]), float(position[2])]


def _normalize_rotation(rotation: Any) -> Optional[Dict[str, float]]:
    if isinstance(rotation, dict):
        w = rotation.get("w", rotation.get("real"))
        x = rotation.get("x", rotation.get("i"))
        y = rotation.get("y", rotation.get("j"))
        z = rotation.get("z", rotation.get("k"))
        values = [w, x, y, z]
        if all(_is_finite_number(v) for v in values):
            return {
                "w": float(w),
                "x": float(x),
                "y": float(y),
                "z": float(z),
            }
    if isinstance(rotation, (list, tuple)) and len(rotation) == 4 and all(_is_finite_number(v) for v in rotation):
        return {
            "w": float(rotation[0]),
            "x": float(rotation[1]),
            "y": float(rotation[2]),
            "z": float(rotation[3]),
        }
    return None


def build_cut_points_details(cut_points: List[int], positions: Optional[List[dict]]) -> Dict[int, Dict[str, Any]]:
    if not isinstance(positions, list) or not positions:
        return {}

    step_index = {}
    for point in positions:
        if not isinstance(point, dict):
            continue
        step = point.get("step")
        if _is_finite_number(step):
            step_index[int(step)] = point

    details: Dict[int, Dict[str, Any]] = {}
    for idx, frame in enumerate(cut_points):
        frame_num = int(frame)
        candidates = []
        if frame_num in step_index:
            candidates.append(step_index[frame_num])
        if (frame_num - 1) in step_index:
            candidates.append(step_index[frame_num - 1])
        if 0 <= frame_num - 1 < len(positions):
            candidates.append(positions[frame_num - 1])
        if 0 <= frame_num < len(positions):
            candidates.append(positions[frame_num])

        chosen = None
        for point in candidates:
            pos = _normalize_position(point.get("position"))
            rot = _normalize_rotation(point.get("rotation"))
            if pos is None or rot is None:
                continue
            chosen = {
                "frame": frame_num,
                "position": pos,
                "rotation": rot,
            }
            break
        if chosen is not None:
            details[idx] = chosen
    return details


def atomic_json_dump(data: Any, path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)
