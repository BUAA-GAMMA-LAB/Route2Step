#!/usr/bin/env python3
"""
Standalone API-based stair block aligner.

The script does not run image segmentation. It only asks an LLM to decide:
- which sub-instructions belong to the stair block
- which code-detected stair blocks correspond to that stair traversal
- whether separated stair blocks should be merged across a landing/platform
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from json import JSONDecoder
import math
import os
import re
import tempfile
import time
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it

try:
    from seg.r2r_train.stair_alignment_utils import (
        build_candidate_stair_blocks,
        choose_primary_block_indices,
        has_strong_opposite_span_between_blocks,
        infer_stair_directions,
        is_down_stair_motion_text,
        is_up_stair_motion_text,
        select_primary_stair_span_indices,
        trim_noncore_stair_instruction_indices,
    )
except ImportError:
    from stair_alignment_utils import (
        build_candidate_stair_blocks,
        choose_primary_block_indices,
        has_strong_opposite_span_between_blocks,
        infer_stair_directions,
        is_down_stair_motion_text,
        is_up_stair_motion_text,
        select_primary_stair_span_indices,
        trim_noncore_stair_instruction_indices,
    )

try:
    from seg.r2r_train.stair_dataset_utils import (
        STAIR_DATASET_CONFIGS,
        get_stair_dataset_config,
        load_coordinate_map as load_coordinate_map_generic,
        load_episode_dataset,
    )
except ImportError:
    from stair_dataset_utils import (
        STAIR_DATASET_CONFIGS,
        get_stair_dataset_config,
        load_coordinate_map as load_coordinate_map_generic,
        load_episode_dataset,
    )


DEFAULT_QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_QWEN_MODEL = "qwen3-vl-plus"
DEFAULT_QWEN_API_KEYS = "path/to/api_keys.txt"
DEFAULT_BATCH_SIZE = 1
DEFAULT_MAX_WORKERS = 8
DEFAULT_DATASET = "r2r"

STAIR_CONTEXT_WORDS = ["stair", "stairs", "staircase", "stairway", "steps", "flight", "landing"]
PROGRESS_WORDS = [
    "halfway", "half way", "middle", "second stair", "third stair", "fourth stair",
    "second step", "third step", "fourth step", "on the stairs", "on the staircase",
    "landing", "between the staircases", "rest of the stairs",
]
ENDPOINT_WORDS = [
    "top of the stairs", "top of stairs", "top of the staircase", "top of staircase",
    "bottom of the stairs", "bottom of stairs", "bottom of the staircase", "bottom of staircase",
    "base of the stairs", "base of stairs", "foot of the stairs", "foot of stairs",
    "top stair", "bottom stair",
]

ALIGNMENT_RESULT_FIELDS = {
    "stair_instruction_indices",
    "stair_core_instruction_indices",
    "selected_block_indices",
    "stair_height_span_indices",
    "merge_selected_blocks",
    "merge_to_single_block",
    "reason",
}

ALIGN_SYSTEM_PROMPT = """You are an indoor-navigation stair alignment analyst.

For each episode, you are given:
- ordered sub-instructions
- heuristic labels for each sub-instruction
- compressed height-change spans from the trajectory
- code-detected candidate stair blocks built from the height signal

Your task is to identify the stair block boundary before segmentation.

Important rules:
- Only include instructions that are actually inside the stair traversal process.
- "Go up/down the stairs" is inside the stair block.
- Return only the instruction indices that must be inside the stair traversal itself.
- Do not include approach-only instructions such as "walk towards the stairs".
- Do not include exit-only instructions such as "turn right at the bottom of the stairs".
- Do not include endpoint-only or turn/navigation actions that happen after finishing the staircase, even if they mention the top or bottom of the stairs.
- `stair_instruction_indices` must be non-empty and contiguous.
- `selected_block_indices` must be non-empty and must point to the provided candidate stair blocks.
- Intermediate landing/platform instructions can still be inside the stair block if the stair traversal clearly continues after them.
- "Turn right at the bottom of the stairs" is usually not inside the stair block; it is post-stair unless the wording explicitly means stopping on-stair.
- "Turn left/right at the very bottom/top" should still be treated as post-stair in normal cases.
- "Stop/wait/stand at the top/bottom of the stairs" should also be treated as post-stair for this task unless the same sub-instruction explicitly contains the up/down stair motion itself.
- If there is exactly one explicit stair-motion instruction, do not require a second stair-motion sentence for later same-direction candidate stair blocks.
- In that single-instruction case, a later or larger same-direction block may still belong to the same "go up/down the stairs" instruction.
- Do not call a long, continuous same-direction block "noise" just because no later sub-instruction explicitly repeats "stairs".
- If one candidate block is short/weak and another same-direction block is longer or has larger total vertical change, prefer the stronger block unless the text clearly says the stair traversal already ended before it.
- If the candidate stair blocks are separated by a short flat landing and the text describes one continuous stair process, select both and set `merge_selected_blocks=true`.
- A single stair-motion instruction may align to multiple same-direction candidate stair blocks if one staircase is broken by a landing, switchback, or brief turn before reaching the final top/bottom.
- If there are two distinct stair traversals with a real non-stair action between them, keep them separate.
- If you cannot satisfy all rules, return an empty object for that episode.

Return JSON only, keyed by episode id, with this schema:
{
  "123": {
    "stair_instruction_indices": [1, 2],
    "selected_block_indices": [0, 1],
    "merge_selected_blocks": true,
    "reason": "..."
  }
}
"""


def clean_instruction(text: str) -> str:
    return re.sub(r"^\s*\d+[\d\.\)\s:-]*", "", str(text)).strip()


def load_api_keys(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"API keys file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        keys = [line.strip() for line in f if line.strip()]
    if not keys:
        raise ValueError(f"No API keys found in {path}")
    return keys


def load_train_seg(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        return raw

    if isinstance(raw, dict):
        normalized = []
        for episode_id, item in raw.items():
            if not isinstance(item, dict):
                continue
            normalized.append({
                "episode_id": int(episode_id),
                "original_instruction": item.get("instruction", ""),
                "sub_instructions": item.get("split_instructions", []),
                "scene_id": item.get("scene_id", ""),
                "rgb_dir": item.get("rgb_dir", ""),
                "cut_points": item.get("cut_points", {}),
            })
        normalized.sort(key=lambda x: int(x["episode_id"]))
        return normalized

    raise ValueError(f"Unsupported input format in {path}")


def load_coordinate_map(path: str) -> Dict[str, list]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    coord_by_ep_id = {}
    for item in raw:
        ep_id = str(item.get("r2r_id", "")).lstrip("0") or "0"
        coord_by_ep_id[ep_id] = item.get("coordinates", [])
    return coord_by_ep_id


def has_stair_context(text: str) -> bool:
    t = text.lower()
    return any(word in t for word in STAIR_CONTEXT_WORDS)


def classify_instruction(text: str) -> str:
    t = text.lower()
    if not has_stair_context(t):
        return "non_stair"
    if any(word in t for word in PROGRESS_WORDS):
        return "stair_progress"
    if any(word in t for word in ENDPOINT_WORDS):
        return "stair_endpoint"
    if is_up_stair_motion_text(t):
        return "stair_motion_up"
    if is_down_stair_motion_text(t):
        return "stair_motion_down"
    return "stair_context"


def has_explicit_stair_motion(text: str) -> bool:
    t = str(text).lower()
    if is_up_stair_motion_text(t) or is_down_stair_motion_text(t):
        return True
    if not has_stair_context(t):
        return False
    return bool(re.search(r"\b(up|down|upstairs|downstairs|ascend|ascending|descend|descending|climb|climbing)\b", t))


def extract_height_value(point) -> Optional[float]:
    if not isinstance(point, dict):
        return None
    position = point.get("position")
    if isinstance(position, list) and len(position) >= 2:
        return float(position[1])
    return None


def summarize_heights(positions: List[dict]) -> dict:
    heights = [extract_height_value(point) for point in positions]
    heights = [h for h in heights if h is not None]
    if not heights:
        return {
            "num_frames": len(positions),
            "overall_delta": 0.0,
            "height_range": 0.0,
            "spans": [],
        }

    diffs = [0.0]
    diffs.extend(heights[idx] - heights[idx - 1] for idx in range(1, len(heights)))

    spans = []
    current_label = None
    start = 0
    accum = 0.0
    peak_abs = 0.0
    for idx, diff in enumerate(diffs[1:], start=1):
        if diff > 0.01:
            label = "up"
        elif diff < -0.01:
            label = "down"
        else:
            label = "flat"

        if current_label is None:
            current_label = label
            start = idx
            accum = diff
            peak_abs = abs(diff)
            continue

        if label == current_label:
            accum += diff
            peak_abs = max(peak_abs, abs(diff))
            continue

        spans.append({
            "span_index": len(spans),
            "start_frame": start,
            "end_frame": idx - 1,
            "label": current_label,
            "delta": round(accum, 4),
            "peak_abs_diff": round(peak_abs, 4),
            "length": idx - start,
        })
        current_label = label
        start = idx
        accum = diff
        peak_abs = abs(diff)

    spans.append({
        "span_index": len(spans),
        "start_frame": start,
        "end_frame": len(diffs) - 1,
        "label": current_label or "flat",
        "delta": round(accum, 4),
        "peak_abs_diff": round(peak_abs, 4),
        "length": max(1, len(diffs) - start),
    })

    significant_spans = []
    for span in spans:
        if span["label"] == "flat":
            if abs(span["delta"]) >= 0.03 and span["length"] <= 6:
                significant_spans.append(span)
            continue
        if abs(span["delta"]) >= 0.08 or span["peak_abs_diff"] >= 0.08:
            significant_spans.append(span)

    return {
        "num_frames": len(positions),
        "overall_delta": round(heights[-1] - heights[0], 4),
        "height_range": round(max(heights) - min(heights), 4),
        "spans": significant_spans,
    }


def heuristic_fallback(item: dict, height_summary: dict) -> dict:
    sub_instructions = item.get("sub_instructions", [])
    labels = [classify_instruction(text) for text in sub_instructions]
    stair_indices = [idx for idx, label in enumerate(labels) if label.startswith("stair_motion")]

    if stair_indices:
        stair_start = stair_indices[0]
        stair_end = stair_indices[-1]
        while stair_start > 0 and labels[stair_start - 1] == "stair_progress":
            stair_start -= 1
        while stair_end + 1 < len(labels) and labels[stair_end + 1] == "stair_progress":
            stair_end += 1
        stair_indices = list(range(stair_start, stair_end + 1))

    preferred_labels = infer_stair_directions(labels, sub_instructions, focus_indices=stair_indices)
    stair_span_indices = select_primary_stair_span_indices(height_summary, preferred_labels)
    candidate_blocks = build_candidate_stair_blocks(height_summary, preferred_labels)
    if stair_indices:
        pre_instruction_indices = list(range(stair_indices[0]))
        post_instruction_indices = list(range(stair_indices[-1] + 1, len(labels)))
    else:
        pre_instruction_indices = []
        post_instruction_indices = list(range(len(labels)))
    selected_block_indices = [int(block.get("block_index", idx)) for idx, block in enumerate(candidate_blocks)]
    selected_span_indices = []
    for block in candidate_blocks:
        selected_span_indices.extend(int(span_idx) for span_idx in block.get("span_indices", []))
    return {
        "stair_instruction_indices": stair_indices,
        "pre_instruction_indices": pre_instruction_indices,
        "post_instruction_indices": post_instruction_indices,
        "candidate_stair_blocks": candidate_blocks,
        "selected_block_indices": selected_block_indices,
        "merge_selected_blocks": len(selected_block_indices) > 1,
        "stair_height_span_indices": sorted(set(selected_span_indices)) or stair_span_indices,
        "merge_to_single_block": len(selected_block_indices) > 1,
        "reason": "heuristic_fallback_directional_cluster",
        "source": "heuristic_fallback",
    }


def build_prompt(batch: List[dict], coord_by_ep_id: Dict[str, list]) -> str:
    sections = []
    for item in batch:
        ep_id = str(item["episode_id"])
        coord_key = str(item.get("coordinate_key") or ep_id)
        sub_instructions = [clean_instruction(x) for x in item.get("sub_instructions", []) if clean_instruction(x)]
        labels = [classify_instruction(text) for text in sub_instructions]
        coords = coord_by_ep_id.get(coord_key, [])
        height_summary = summarize_heights(coords)

        sections.append(f"Episode {ep_id}:")
        sections.append("Sub-instructions:")
        for idx, (text, label) in enumerate(zip(sub_instructions, labels)):
            sections.append(f"- [{idx}] ({label}) {text}")
        sections.append("Height summary:")
        sections.append(
            f"- frames={height_summary['num_frames']}, overall_delta={height_summary['overall_delta']}, "
            f"height_range={height_summary['height_range']}"
        )
        if height_summary["spans"]:
            sections.append("- significant_spans:")
            for span in height_summary["spans"]:
                sections.append(
                    f"  - [{span['span_index']}] frames {span['start_frame']}-{span['end_frame']}, "
                    f"{span['label']}, delta={span['delta']}, peak_abs_diff={span['peak_abs_diff']}, len={span['length']}"
                )
        else:
            sections.append("- significant_spans: []")
        candidate_blocks = build_candidate_stair_blocks(height_summary, infer_stair_directions(labels, sub_instructions))
        if candidate_blocks:
            sections.append("- candidate_stair_blocks:")
            for block in candidate_blocks:
                sections.append(
                    f"  - [{block['block_index']}] frames {block['start_frame']}-{block['end_frame']}, "
                    f"span_indices={block['span_indices']}, labels={block['labels']}, "
                    f"total_abs_delta={block['total_abs_delta']}, max_peak_abs_diff={block['max_peak_abs_diff']}"
                )
        else:
            sections.append("- candidate_stair_blocks: []")
        sections.append("")
    return "\n".join(sections)


def normalize_indices(values, max_len: int) -> List[int]:
    if not isinstance(values, list):
        return []
    result = []
    for value in values:
        try:
            idx = int(value)
        except Exception:
            continue
        if 0 <= idx < max_len:
            result.append(idx)
    return sorted(set(result))


def normalize_span_indices(values, spans: List[dict]) -> List[int]:
    if not isinstance(values, list):
        return []

    span_id_to_pos = {}
    for pos, span in enumerate(spans):
        try:
            span_id = int(span.get("span_index", pos))
        except Exception:
            span_id = pos
        span_id_to_pos.setdefault(span_id, pos)

    result = []
    for value in values:
        try:
            idx = int(value)
        except Exception:
            continue
        if idx in span_id_to_pos:
            result.append(idx)
        elif 0 <= idx < len(spans):
            result.append(int(spans[idx].get("span_index", idx)))
    return sorted(set(result))


def normalize_block_indices(values, candidate_blocks: List[dict]) -> List[int]:
    if not isinstance(values, list):
        return []
    valid_indices = {int(block.get("block_index", idx)) for idx, block in enumerate(candidate_blocks)}
    result = []
    for value in values:
        try:
            idx = int(value)
        except Exception:
            continue
        if idx in valid_indices:
            result.append(idx)
    return sorted(set(result))


def build_disjoint_stair_alignment(raw: dict, sub_instructions: List[str], height_summary: dict) -> Optional[dict]:
    if not isinstance(raw, dict):
        return None

    num_subs = len(sub_instructions)
    if num_subs <= 1:
        return None

    stair_indices = set(normalize_indices(raw.get("stair_instruction_indices", []), num_subs))
    legacy_core = set(normalize_indices(raw.get("stair_core_instruction_indices", []), num_subs))
    if not stair_indices and legacy_core:
        stair_indices = legacy_core
    if len(stair_indices) < 2:
        return None

    stair_sorted = sorted(stair_indices)
    if stair_sorted == list(range(stair_sorted[0], stair_sorted[-1] + 1)):
        return None

    instruction_labels = [classify_instruction(text) for text in sub_instructions]
    trimmed = list(stair_sorted)
    while trimmed and not has_explicit_stair_motion(sub_instructions[trimmed[0]]):
        trimmed.pop(0)
    while trimmed and not has_explicit_stair_motion(sub_instructions[trimmed[-1]]):
        trimmed.pop()
    if len(trimmed) < 2:
        return None
    if trimmed == list(range(trimmed[0], trimmed[-1] + 1)):
        return None

    instruction_spans = []
    span_start = trimmed[0]
    span_end = trimmed[0]
    for idx in trimmed[1:]:
        if idx == span_end + 1:
            span_end = idx
            continue
        instruction_spans.append([span_start, span_end])
        span_start = idx
        span_end = idx
    instruction_spans.append([span_start, span_end])
    if len(instruction_spans) < 2:
        return None

    if not all(
        any(
            has_explicit_stair_motion(sub_instructions[idx])
            for idx in range(start, end + 1)
        )
        for start, end in instruction_spans
    ):
        return None

    preferred_labels = infer_stair_directions(instruction_labels, sub_instructions, focus_indices=trimmed)
    candidate_blocks = build_candidate_stair_blocks(height_summary, preferred_labels)
    selected_block_indices = normalize_block_indices(raw.get("selected_block_indices", []), candidate_blocks)
    spans = height_summary.get("spans", []) if isinstance(height_summary, dict) else []
    if not selected_block_indices:
        legacy_span_indices = set(normalize_span_indices(raw.get("stair_height_span_indices", []), spans))
        if legacy_span_indices:
            selected_block_indices = [
                int(block.get("block_index", idx))
                for idx, block in enumerate(candidate_blocks)
                if legacy_span_indices & set(int(span_idx) for span_idx in block.get("span_indices", []))
            ]

    return {
        "alignment_type": "multi_stair_disjoint",
        "stair_instruction_indices": trimmed,
        "disjoint_stair_instruction_spans": instruction_spans,
        "candidate_stair_blocks": candidate_blocks,
        "selected_block_indices": selected_block_indices,
        "merge_selected_blocks": bool(raw.get("merge_selected_blocks", raw.get("merge_to_single_block", False))),
        "reason": str(raw.get("reason", "")).strip(),
    }


def validate_and_canonicalize_alignment(raw: dict, sub_instructions: List[str], height_summary: dict) -> Optional[dict]:
    if not isinstance(raw, dict):
        return None

    num_subs = len(sub_instructions)
    spans = height_summary.get("spans", []) if isinstance(height_summary, dict) else []

    stair_indices = set(normalize_indices(raw.get("stair_instruction_indices", []), num_subs))
    legacy_core = set(normalize_indices(raw.get("stair_core_instruction_indices", []), num_subs))
    if not stair_indices and legacy_core:
        stair_indices = legacy_core
    if not stair_indices:
        return None

    stair_sorted = sorted(stair_indices)
    if stair_sorted != list(range(stair_sorted[0], stair_sorted[-1] + 1)):
        return None

    instruction_labels = [classify_instruction(text) for text in sub_instructions]
    stair_sorted = trim_noncore_stair_instruction_indices(instruction_labels, stair_sorted)
    if not stair_sorted:
        return None
    preferred_labels = infer_stair_directions(instruction_labels, sub_instructions, focus_indices=stair_sorted)
    candidate_blocks = build_candidate_stair_blocks(height_summary, preferred_labels)
    if not candidate_blocks:
        primary_span_indices = select_primary_stair_span_indices(height_summary, preferred_labels)
        if primary_span_indices:
            primary_span_set = set(primary_span_indices)
            selected_spans = [
                span for span in spans
                if int(span.get("span_index", -1)) in primary_span_set
            ]
            if selected_spans:
                candidate_blocks = [{
                    "block_index": 0,
                    "span_indices": sorted(primary_span_set),
                    "start_frame": min(int(span.get("start_frame", 0)) for span in selected_spans),
                    "end_frame": max(int(span.get("end_frame", 0)) for span in selected_spans),
                    "labels": sorted({str(span.get("label", "")) for span in selected_spans if span.get("label")}),
                    "total_abs_delta": round(sum(abs(float(span.get("delta", 0.0))) for span in selected_spans), 4),
                    "max_peak_abs_diff": round(max(abs(float(span.get("peak_abs_diff", 0.0))) for span in selected_spans), 4),
                }]
    if not candidate_blocks:
        return None

    selected_block_indices = normalize_block_indices(raw.get("selected_block_indices", []), candidate_blocks)
    if not selected_block_indices:
        legacy_span_indices = set(normalize_span_indices(raw.get("stair_height_span_indices", []), spans))
        if legacy_span_indices:
            selected_block_indices = [
                int(block.get("block_index", idx))
                for idx, block in enumerate(candidate_blocks)
                if legacy_span_indices & set(int(span_idx) for span_idx in block.get("span_indices", []))
            ]
    if not selected_block_indices:
        selected_block_indices = [int(candidate_blocks[0].get("block_index", 0))]

    merge_selected_blocks = bool(raw.get("merge_selected_blocks", raw.get("merge_to_single_block", False)))
    if len(selected_block_indices) > 1 and not merge_selected_blocks:
        return None
    if merge_selected_blocks and has_strong_opposite_span_between_blocks(spans, candidate_blocks, selected_block_indices):
        selected_block_indices = choose_primary_block_indices(candidate_blocks, selected_block_indices)
        merge_selected_blocks = False

    selected_block_set = set(selected_block_indices)
    height_span_indices = sorted({
        int(span_idx)
        for block in candidate_blocks
        if int(block.get("block_index", -1)) in selected_block_set
        for span_idx in block.get("span_indices", [])
    })
    if not height_span_indices:
        return None

    pre_indices = list(range(stair_sorted[0]))
    post_indices = list(range(stair_sorted[-1] + 1, num_subs))

    return {
        "stair_instruction_indices": stair_sorted,
        "pre_instruction_indices": pre_indices,
        "post_instruction_indices": post_indices,
        "candidate_stair_blocks": candidate_blocks,
        "selected_block_indices": selected_block_indices,
        "merge_selected_blocks": merge_selected_blocks,
        "stair_height_span_indices": height_span_indices,
        "merge_to_single_block": merge_selected_blocks,
        "reason": str(raw.get("reason", "")).strip(),
    }


def sanitize_alignment_entry(entry: dict) -> dict:
    if not isinstance(entry, dict):
        return entry

    sub_instructions = [clean_instruction(x) for x in entry.get("sub_instructions", []) if clean_instruction(x)]
    height_summary = entry.get("height_summary", {})
    canonical = validate_and_canonicalize_alignment(entry, sub_instructions, height_summary)

    sanitized = dict(entry)
    if canonical is not None:
        sanitized.update(canonical)
    else:
        disjoint = build_disjoint_stair_alignment(entry, sub_instructions, height_summary)
        if disjoint is not None:
            sanitized.update(disjoint)

    sanitized["sub_instructions"] = sub_instructions
    sanitized["instruction_labels"] = [classify_instruction(text) for text in sub_instructions]
    sanitized["height_summary"] = height_summary if isinstance(height_summary, dict) else {}
    return sanitized


def _decode_first_json_value(response_text: str) -> tuple[Optional[Any], Optional[str], Optional[str]]:
    text = (response_text or "").strip()
    if not text:
        return None, None, "empty_response"

    decoder = json.JSONDecoder()
    last_error = None
    for match in re.finditer(r"[\{\[]", text):
        start = match.start()
        try:
            obj, end = decoder.raw_decode(text[start:])
            return obj, text[start:start + end], None
        except json.JSONDecodeError as exc:
            last_error = str(exc)
            continue

    return None, None, last_error or "no_json_value_found"


def _looks_like_alignment_payload(obj: Any) -> bool:
    return isinstance(obj, dict) and any(key in obj for key in ALIGNMENT_RESULT_FIELDS)


def _normalize_response_payload(obj: Any, batch: List[dict]) -> tuple[Optional[Dict[str, dict]], Optional[str]]:
    ep_ids = [str(item["episode_id"]) for item in batch]

    if isinstance(obj, dict):
        result = {}
        for ep_id in ep_ids:
            raw = obj.get(ep_id)
            if isinstance(raw, dict):
                result[ep_id] = raw
        if result:
            return result, None

        for container_key in ("results", "episodes", "data", "output", "alignments"):
            nested = obj.get(container_key)
            if isinstance(nested, (dict, list)):
                normalized, error = _normalize_response_payload(nested, batch)
                if normalized is not None:
                    return normalized, None
                if error:
                    return None, error

        if len(batch) == 1:
            ep_id = ep_ids[0]
            if _looks_like_alignment_payload(obj):
                return {ep_id: obj}, None

            raw_episode_id = obj.get("episode_id")
            if raw_episode_id is not None and str(raw_episode_id) == ep_id:
                payload = {k: v for k, v in obj.items() if k != "episode_id"}
                if _looks_like_alignment_payload(payload):
                    return {ep_id: payload}, None

        return None, "missing_episode_keyed_payload"

    if isinstance(obj, list):
        if len(batch) == 1 and len(obj) == 1 and _looks_like_alignment_payload(obj[0]):
            return {ep_ids[0]: obj[0]}, None

        result = {}
        for entry in obj:
            if not isinstance(entry, dict):
                continue
            raw_episode_id = entry.get("episode_id")
            if raw_episode_id is None:
                continue
            ep_id = str(raw_episode_id)
            if ep_id in ep_ids:
                payload = {k: v for k, v in entry.items() if k != "episode_id"}
                result[ep_id] = payload
        if result:
            return result, None
        return None, "missing_episode_id_in_payload_list"

    return None, f"unsupported_json_type:{type(obj).__name__}"


def parse_response(response_text: str, batch: List[dict], coord_by_ep_id: Dict[str, list]) -> tuple[Dict[str, dict], bool]:
    obj, json_snippet, extract_error = _decode_first_json_value(response_text)
    normalized_payload = None
    normalize_error = None
    if obj is not None:
        normalized_payload, normalize_error = _normalize_response_payload(obj, batch)

    parsed = {}
    all_resolved = True
    for item in batch:
        ep_id = str(item["episode_id"])
        coord_key = str(item.get("coordinate_key") or ep_id)
        sub_instructions = [clean_instruction(x) for x in item.get("sub_instructions", []) if clean_instruction(x)]
        height_summary = summarize_heights(coord_by_ep_id.get(coord_key, []))
        fallback = heuristic_fallback(item, height_summary)
        raw = normalized_payload.get(ep_id) if normalized_payload else None
        canonical = validate_and_canonicalize_alignment(raw, sub_instructions, height_summary)
        if canonical is not None:
            parsed[ep_id] = {
                **canonical,
                "source": "api",
                "sub_instructions": sub_instructions,
                "instruction_labels": [classify_instruction(text) for text in sub_instructions],
                "height_summary": height_summary,
            }
            continue

        disjoint = build_disjoint_stair_alignment(raw, sub_instructions, height_summary)
        if disjoint is not None:
            parsed[ep_id] = {
                **disjoint,
                "source": "api_multi_stair_disjoint",
                "sub_instructions": sub_instructions,
                "instruction_labels": [classify_instruction(text) for text in sub_instructions],
                "height_summary": height_summary,
            }
            continue

        all_resolved = False
        if obj is None:
            error = extract_error or "failed_to_extract_json"
            failure_stage = "json_extract_failed"
            payload_for_debug = None
        elif normalized_payload is None:
            error = normalize_error or "failed_to_normalize_payload"
            failure_stage = "payload_normalization_failed"
            payload_for_debug = obj
        elif raw is None:
            error = "episode_payload_missing"
            failure_stage = "episode_payload_missing"
            payload_for_debug = obj
        else:
            error = "api_alignment_invalid_or_empty"
            failure_stage = "canonical_validation_failed"
            payload_for_debug = raw

        parsed[ep_id] = {
            **fallback,
            "error": error,
            "api_failure_stage": failure_stage,
            "api_raw_response_text": response_text,
            "api_json_snippet": json_snippet,
            "api_response_json": payload_for_debug,
            "sub_instructions": sub_instructions,
            "instruction_labels": [classify_instruction(text) for text in sub_instructions],
            "height_summary": height_summary,
        }
    return parsed, all_resolved


def _recover_partial_result_dict(path: str) -> Dict[str, dict]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    decoder = JSONDecoder()
    pos = 0
    total = len(text)
    while pos < total and text[pos].isspace():
        pos += 1
    if pos >= total or text[pos] != "{":
        raise ValueError(f"{path} is not a JSON object")
    pos += 1

    recovered: Dict[str, dict] = {}
    while True:
        while pos < total and text[pos].isspace():
            pos += 1
        if pos >= total or text[pos] == "}":
            break

        try:
            key, pos = decoder.raw_decode(text, pos)
        except Exception:
            break

        while pos < total and text[pos].isspace():
            pos += 1
        if pos >= total or text[pos] != ":":
            break
        pos += 1

        while pos < total and text[pos].isspace():
            pos += 1
        try:
            value, pos = decoder.raw_decode(text, pos)
        except Exception:
            break

        recovered[str(key)] = value

        while pos < total and text[pos].isspace():
            pos += 1
        if pos < total and text[pos] == ",":
            pos += 1
            continue
        break

    if not recovered:
        raise ValueError(f"Failed to recover any complete entries from {path}")
    return recovered


def _load_existing_result_dict(path: str) -> Dict[str, dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_result = json.load(f)
    except json.JSONDecodeError:
        raw_result = _recover_partial_result_dict(path)
        backup_path = f"{path}.truncated_backup_{time.strftime('%Y%m%d_%H%M%S')}"
        with open(path, "r", encoding="utf-8") as src, open(backup_path, "w", encoding="utf-8") as dst:
            dst.write(src.read())
        write_json_atomic(raw_result, path)
        print(f"Recovered {len(raw_result)} entries from truncated output -> {path}")
        print(f"Backup written to {backup_path}")

    if not isinstance(raw_result, dict):
        raise ValueError(f"{path} does not contain a JSON object at top level")
    return raw_result


def write_json_atomic(data: Dict[str, dict], output_file: str) -> None:
    output_dir = os.path.dirname(output_file) or "."
    os.makedirs(output_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output_dir, delete=False) as tmp:
        json.dump(data, tmp, ensure_ascii=False, indent=2)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = tmp.name
    os.replace(tmp_path, output_file)


def call_api_for_alignment(
    episodes: List[dict],
    coord_by_ep_id: Dict[str, list],
    base_url: str,
    model: str,
    api_keys_file: str,
    batch_size: int,
    max_workers: int,
    output_file: str,
    load_existing: bool,
    sleep_seconds: float = 0.0,
) -> Dict[str, dict]:
    if OpenAI is None:
        raise ImportError("openai is required. Please install openai first.")

    if load_existing and os.path.exists(output_file):
        raw_result = _load_existing_result_dict(output_file)
        result = {
            str(ep_id): sanitize_alignment_entry(entry)
            for ep_id, entry in raw_result.items()
        }
    else:
        result = {}

    pending = [ep for ep in episodes if str(ep["episode_id"]) not in result]
    if not pending:
        return result

    api_keys = load_api_keys(api_keys_file)
    batches = [
        pending[batch_idx * batch_size:(batch_idx + 1) * batch_size]
        for batch_idx in range(math.ceil(len(pending) / batch_size))
    ]

    def process_batch(batch_idx: int, batch: List[dict]) -> Dict[str, dict]:
        prompt = build_prompt(batch, coord_by_ep_id)
        parsed = None
        last_failed_parsed = None
        last_error = None
        success_attempt_count = None

        for attempt in range(3):
            api_key = api_keys[(batch_idx + attempt) % len(api_keys)]
            client = OpenAI(api_key=api_key, base_url=base_url)
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": ALIGN_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                )
                text = resp.choices[0].message.content or ""
                parsed, success = parse_response(text, batch, coord_by_ep_id)
                if success:
                    success_attempt_count = attempt + 1
                    break
                last_failed_parsed = parsed
                last_error = "; ".join(
                    f"{ep_id}:{entry.get('error', 'unknown_error')}"
                    for ep_id, entry in sorted(parsed.items())
                )
            except Exception as exc:
                last_error = str(exc)
                time.sleep(1.5)

        if parsed is None:
            if last_failed_parsed is not None:
                parsed = last_failed_parsed
                for entry in parsed.values():
                    entry["api_attempt_count"] = 3
            else:
                parsed = {}
                for item in batch:
                    ep_id = str(item["episode_id"])
                    coord_key = str(item.get("coordinate_key") or ep_id)
                    height_summary = summarize_heights(coord_by_ep_id.get(coord_key, []))
                    fallback = heuristic_fallback(item, height_summary)
                    fallback["error"] = last_error
                    fallback["api_attempt_count"] = 3
                    parsed[ep_id] = fallback
        else:
            for entry in parsed.values():
                entry["api_attempt_count"] = success_attempt_count or 1
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
        return parsed

    max_workers = max(1, min(max_workers, len(batches), len(api_keys)))
    with tqdm(total=len(pending), unit="ep", desc="Stair alignment API") as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {
                executor.submit(process_batch, batch_idx, batch): batch
                for batch_idx, batch in enumerate(batches)
            }
            for future in as_completed(future_to_batch):
                batch = future_to_batch[future]
                parsed = future.result()
                for item in batch:
                    ep_id = str(item["episode_id"])
                    entry = parsed[ep_id]
                    if "sub_instructions" not in entry:
                        coord_key = str(item.get("coordinate_key") or ep_id)
                        sub_instructions = [clean_instruction(x) for x in item.get("sub_instructions", []) if clean_instruction(x)]
                        entry["sub_instructions"] = sub_instructions
                        entry["instruction_labels"] = [classify_instruction(text) for text in sub_instructions]
                        entry["height_summary"] = summarize_heights(coord_by_ep_id.get(coord_key, []))
                    entry = sanitize_alignment_entry(entry)
                    result[ep_id] = entry

                write_json_atomic(result, output_file)

                pbar.update(len(batch))

    return result


def main():
    parser = argparse.ArgumentParser(description="Standalone stair alignment API caller.")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, choices=tuple(sorted(STAIR_DATASET_CONFIGS)))
    parser.add_argument("--input_file", type=str, default="")
    parser.add_argument("--coord_file", type=str, default="")
    parser.add_argument("--output_file", type=str, default="")
    parser.add_argument("--stair_filter_cache", type=str, default="")
    parser.add_argument("--episode", type=str, default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--all_episodes", action="store_true",
                        help="Process all selected episodes instead of filtering by stair cache.")
    parser.add_argument("--ignore_existing", action="store_true",
                        help="Recompute episodes even if they already exist in the output file.")
    parser.add_argument("--base_url", type=str, default=DEFAULT_QWEN_BASE_URL)
    parser.add_argument("--model", type=str, default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--api_keys_file", type=str, default=DEFAULT_QWEN_API_KEYS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max_workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--sleep_seconds", type=float, default=0.0)
    args = parser.parse_args()

    dataset_cfg = get_stair_dataset_config(args.dataset)
    input_file = args.input_file or dataset_cfg["input_file"]
    coord_file = args.coord_file or dataset_cfg["coord_file"]
    output_file = args.output_file or dataset_cfg["stair_alignment_file"]
    stair_filter_cache = args.stair_filter_cache or dataset_cfg["stair_filter_cache"]

    episodes = load_episode_dataset(input_file, args.dataset)
    if args.episode:
        target_ids = {item.strip() for item in args.episode.split(",") if item.strip()}
        episodes = [ep for ep in episodes if str(ep["episode_id"]) in target_ids]

    if not args.all_episodes and os.path.exists(stair_filter_cache):
        with open(stair_filter_cache, "r", encoding="utf-8") as f:
            stair_cache = json.load(f)
        episodes = [
            ep for ep in episodes
            if stair_cache.get(str(ep["episode_id"]), {}).get("contains_vertical_stairs", False)
        ]

    if args.limit > 0:
        episodes = episodes[:args.limit]

    coord_by_ep_id = load_coordinate_map_generic(coord_file)
    print(f"Dataset={args.dataset} selected episodes: {len(episodes)}")
    print(f"Loaded coordinate trajectories: {len(coord_by_ep_id)}")

    result = call_api_for_alignment(
        episodes=episodes,
        coord_by_ep_id=coord_by_ep_id,
        base_url=args.base_url,
        model=args.model,
        api_keys_file=args.api_keys_file,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        output_file=output_file,
        load_existing=(not args.ignore_existing),
        sleep_seconds=args.sleep_seconds,
    )
    print(f"Done. {len(result)} episodes -> {output_file}")


if __name__ == "__main__":
    main()
