"""R2R-train E-SPA segmentation.

The input contains sub-instructions and episode metadata. External trajectory
images, annotations, and action records are supplied by the caller. The
output stores episode-level segmentation results with 1-based frame indices.
"""
import os
import json
import re
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing as mp
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

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
        atomic_json_dump,
        build_cut_points_details,
        build_rgb_dir,
        get_stair_dataset_config,
        infer_data_path,
        infer_scan_id,
        load_annotation_index,
        load_coordinate_map as load_coordinate_map_generic,
        load_episode_dataset,
    )
except ImportError:
    from stair_dataset_utils import (
        STAIR_DATASET_CONFIGS,
        atomic_json_dump,
        build_cut_points_details,
        build_rgb_dir,
        get_stair_dataset_config,
        infer_data_path,
        infer_scan_id,
        load_annotation_index,
        load_coordinate_map as load_coordinate_map_generic,
        load_episode_dataset,
    )

try:
    from transformers import CLIPModel, CLIPProcessor
except ImportError:
    print("[Error] transformers not found.")
    exit(1)

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw): return it

# ===== Paths =====
TRAIN_SEG_FILE  = "data/StreamVLN-Trajectory-Data/R2R/r2r_train_seg.json"
ANN_FILE        = "data/StreamVLN-Trajectory-Data/R2R/annotations.json"
COORD_FILE      = "data/StreamVLN-Trajectory-Data/R2R/all_coordinates.json"
IMAGES_BASE_DIR = "data/StreamVLN-Trajectory-Data/R2R"
CLIP_MODEL_PATH = "model_zoo/clip-vit-large-patch14_0"
ACTION_MAP_FILE = "seg/outputs/actions/sub_instruction_actions_r2r_train.json"
OUTPUT_FILE     = "seg/outputs/segmentation/seg_r2r_train.json"
STAIR_ONLY_OUTPUT_FILE = "seg/outputs/stairs/seg_r2r_train_stairs.json"
STAIR_FILTER_CACHE_FILE = "seg/outputs/stairs/r2r_train_stair_episode_filter.json"
STAIR_COMPARE_MARKDOWN_FILE = "seg/outputs/stairs/seg_r2r_train_stairs_links.md"
STAIR_ALIGNMENT_FILE = "seg/outputs/stairs/r2r_train_stair_alignment_api.json"
DEFAULT_QWEN_BASE_URL   = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_QWEN_MODEL      = "qwen3-vl-flash"
DEFAULT_QWEN_API_KEYS   = "path/to/api_keys.txt"
DEFAULT_STAIR_BATCH     = 1
MAX_WORKERS     = 3
DEFAULT_GPU_ID  = 2
DEFAULT_GPU_IDS = "2,3,4,5"
DEFAULT_DATASET = "r2r"

STAIR_CONTEXT_WORDS = ['stair', 'stairs', 'staircase', 'steps', 'flight']
STAIR_PROGRESS_PHRASES = [
    'halfway', 'half way', 'middle of the stair', 'middle of the stairs',
    'middle of the staircase', 'middle of the stairway', 'on the stair',
    'on the stairs', 'on the staircase', 'on the stairway',
    'on the second stair', 'on the third stair', 'on the fourth stair',
    'on the fifth stair', 'on the first stair', 'on the second step',
    'on the third step', 'on the fourth step', 'on the fifth step',
    'between the winding staircases', 'between another set of stairs',
    'between staircases', 'rest of the stairs', 'rest of the staircase'
]
STAIR_ENDPOINT_PHRASES = [
    'top of the stair', 'top of the stairs', 'top of stair', 'top of stairs',
    'top of the staircase', 'top of staircase', 'top of the stairway', 'top of stairway',
    'bottom of the stair', 'bottom of the stairs', 'bottom of stair', 'bottom of stairs',
    'bottom of the staircase', 'bottom of staircase', 'bottom of the stairway', 'bottom of stairway',
    'base of the stair', 'base of the stairs', 'base of stair', 'base of stairs',
    'foot of the stair', 'foot of the stairs', 'foot of stair', 'foot of stairs',
    'once you get up the stairs', 'once you get down the stairs',
    'when you get to the top of the stairs', 'when you get to the bottom of the stairs',
    'when you arrive on the top of the stairs', 'when you arrive at the bottom of the stairs'
]

STAIR_FILTER_SYSTEM_PROMPT = """You are an indoor navigation analyst.

Your job is to decide whether an episode contains a sub-instruction that requires actual vertical traversal on stairs or steps.

Count as true only when the agent must physically go up or down stairs/steps, such as:
- "go up the stairs"
- "walk down the steps"
- "climb the staircase"
- "descend the stairs"
- "go up two steps"

Count as false when stairs are only landmarks or nearby context, such as:
- "go to the stairs"
- "wait at the top of the stairs"
- "stop at the top of the stairs"
- "turn left at the bottom of the stairs"
- "walk past the stairs"
- "enter the room beside the stairs"

Important:
- Mark true only when at least one sub-instruction explicitly contains real vertical stair traversal such as up/down/ascend/descend/climb stairs or steps.
- Do not mark true for endpoint-only or location-only text such as top/bottom of the stairs unless the same matched sub-instruction itself contains the actual up/down stair motion.
- `matched_sub_instructions` must contain only the explicit stair-traversal sub-instructions that justify `contains_vertical_stairs=true`.

Respond ONLY with a JSON object keyed by episode id:
{
  "123": {
    "contains_vertical_stairs": true,
    "matched_sub_instructions": ["Go up the stairs."]
  }
}
"""


def clean_instruction(text):
    return re.sub(r'^\s*\d+[\d\.\)\s:-]*', '', text).strip()


def has_stair_context(text: str) -> bool:
    t = text.lower()
    return any(word in t for word in STAIR_CONTEXT_WORDS)


def is_up_stair_motion(text: str) -> bool:
    t = text.lower()
    has_context = has_stair_context(t)
    return has_context and is_up_stair_motion_text(t)


def is_down_stair_motion(text: str) -> bool:
    t = text.lower()
    has_context = has_stair_context(t)
    return has_context and is_down_stair_motion_text(t)


def is_vertical_stair_motion(text: str) -> bool:
    return is_up_stair_motion(text) or is_down_stair_motion(text)


def is_stair_progress_state(text: str) -> bool:
    t = text.lower()
    if not has_stair_context(t) or is_vertical_stair_motion(t):
        return False
    if 'landing' in t and 'between' in t:
        return True
    if any(phrase in t for phrase in STAIR_PROGRESS_PHRASES):
        return True
    if any(token in t for token in ['halfway', 'half way', 'three-quarters', 'three quarters']):
        return True
    has_numbered_step = bool(re.search(r'\b(first|second|third|fourth|fifth|\d+)(?:st|nd|rd|th)?\s+(?:step|stair)\b', t))
    return has_numbered_step and not any(phrase in t for phrase in STAIR_ENDPOINT_PHRASES)


def is_stair_endpoint_state(text: str) -> bool:
    t = text.lower()
    if not has_stair_context(t) or is_vertical_stair_motion(t):
        return False
    if 'landing' in t and 'between' in t:
        return False
    return any(phrase in t for phrase in STAIR_ENDPOINT_PHRASES)


def extract_vertical_stair_sub_instructions(item: dict) -> List[str]:
    split_instrs = [clean_instruction(s) for s in item.get('sub_instructions', []) if clean_instruction(s)]
    return [text for text in split_instrs if is_vertical_stair_motion(text)]


def normalize_vertical_stair_matches(item: dict, matched) -> List[str]:
    if isinstance(matched, str):
        matched = [matched]
    if not isinstance(matched, list):
        matched = []

    valid = []
    seen = set()
    for text in matched:
        cleaned = clean_instruction(str(text))
        if not cleaned or not is_vertical_stair_motion(cleaned):
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        valid.append(cleaned)

    if valid:
        return valid

    fallback = extract_vertical_stair_sub_instructions(item)
    deduped = []
    seen = set()
    for text in fallback:
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(text)
    return deduped


def sorted_cut_points(cut_points) -> List[Optional[int]]:
    if isinstance(cut_points, dict):
        try:
            return [int(v) for _, v in sorted(cut_points.items(), key=lambda kv: int(kv[0]))]
        except Exception:
            return []
    if isinstance(cut_points, list):
        try:
            return [int(v) for v in cut_points]
        except Exception:
            return []
    return []




def normalize_cut_points_for_instruction_count(cut_points, num_frames: int, num_instructions: int) -> List[int]:
    points = sorted_cut_points(cut_points)
    if not points:
        return []
    cleaned = []
    for point in points:
        try:
            value = int(point)
        except Exception:
            continue
        if num_frames > 0:
            value = min(max(value, 1), num_frames)
        if not cleaned or value > cleaned[-1]:
            cleaned.append(value)
    if not cleaned:
        return []
    if cleaned[0] != 1:
        cleaned.insert(0, 1)
    if num_frames > 0 and cleaned[-1] != num_frames:
        cleaned.append(num_frames)
    if num_instructions > 0 and len(cleaned) > num_instructions + 1:
        terminal = cleaned[-1]
        cleaned = cleaned[:num_instructions]
        if terminal > cleaned[-1]:
            cleaned.append(terminal)
        elif cleaned[-1] != terminal:
            cleaned[-1] = terminal
    return cleaned

def format_frame_label(frame_idx) -> str:
    if frame_idx in ("", None):
        return ""
    try:
        frame_num = int(frame_idx)
    except Exception:
        return str(frame_idx)
    return f"{frame_num} ({frame_num:03d}.jpg)"


def build_segment_ranges_from_cut_points(cut_points, num_frames: int) -> List[dict]:
    starts = sorted_cut_points(cut_points)
    if not starts or num_frames <= 0:
        return []

    ranges = []
    for idx, start in enumerate(starts):
        if start is None:
            continue
        end = starts[idx + 1] - 1 if idx + 1 < len(starts) else num_frames
        try:
            start = int(start)
            end = int(end)
        except Exception:
            continue
        if start > end:
            end = start
        ranges.append({
            "segment_index": idx,
            "start_frame": start,
            "end_frame": end,
        })
    return ranges


def build_cut_point_ranges(cut_points, num_frames: int) -> Dict[int, str]:
    ranges = build_segment_ranges_from_cut_points(cut_points, num_frames)
    result = {}
    for segment in ranges:
        try:
            idx = int(segment.get("segment_index"))
            start = int(segment.get("start_frame"))
            end = int(segment.get("end_frame"))
        except Exception:
            continue
        result[idx] = f"{start}-{end}"
    return result


def format_segment_range_label(segment: Optional[dict]) -> str:
    if not isinstance(segment, dict):
        return ""
    start = segment.get("start_frame")
    end = segment.get("end_frame")
    if start in ("", None) or end in ("", None):
        return ""
    try:
        start = int(start)
        end = int(end)
    except Exception:
        return f"{start}-{end}"
    return f"{start} ({start:03d}.jpg) - {end} ({end:03d}.jpg)"


TERMINAL_HOLD_WORDS_RE = re.compile(r"\b(stop|wait|stand|halt|stay|remain|pause|hold)\b", re.I)


def is_terminal_hold_instruction(text: str) -> bool:
    return bool(TERMINAL_HOLD_WORDS_RE.search(str(text or "")))


def merge_trailing_unmapped_hold_instructions(split_instructions: List[str], cut_points, num_frames: int) -> List[str]:
    merged = [str(text) for text in split_instructions]
    if not merged:
        return merged

    ranges = build_segment_ranges_from_cut_points(cut_points, num_frames)
    mapped_count = min(len(ranges), len(merged))
    if mapped_count >= len(merged) or mapped_count <= 0:
        return merged

    trailing = merged[mapped_count:]
    if not trailing or not all(is_terminal_hold_instruction(text) for text in trailing):
        return merged

    for text in trailing:
        text = text.strip()
        if not text:
            continue
        base = merged[mapped_count - 1].rstrip()
        merged[mapped_count - 1] = f"{base} {text}".strip()
    return merged[:mapped_count]


def build_instruction_segments(split_instructions: List[str], cut_points, num_frames: int) -> List[dict]:
    ranges = build_segment_ranges_from_cut_points(cut_points, num_frames)
    instruction_segments = []
    for idx, text in enumerate(split_instructions):
        segment = ranges[idx] if idx < len(ranges) else {}
        start_frame = segment.get("start_frame")
        end_frame = segment.get("end_frame")
        frame_range = ""
        if start_frame not in ("", None) and end_frame not in ("", None):
            try:
                frame_range = f"{int(start_frame)}-{int(end_frame)}"
            except Exception:
                frame_range = f"{start_frame}-{end_frame}"
        instruction_segments.append({
            "sub_instruction_index": idx,
            "sub_instruction": text,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "frame_range": frame_range,
        })
    return instruction_segments


def write_stair_compare_markdown(
    markdown_path: str,
    latest_results: Dict[str, dict],
    train_seg_by_id: Dict[str, dict],
    previous_results: Optional[Dict[str, dict]] = None,
):
    previous_results = previous_results or {}
    episode_ids = sorted(latest_results.keys(), key=lambda x: int(x))
    lines = ["# R2R Stairs 001.jpg Links", ""]

    for ep_id in episode_ids:
        latest = latest_results.get(ep_id, {})
        previous = previous_results.get(ep_id, {})
        train_item = train_seg_by_id.get(str(ep_id), {})

        rgb_dir = (
            latest.get("rgb_dir")
            or previous.get("rgb_dir")
            or os.path.join(IMAGES_BASE_DIR, "images", train_item.get("data_path", ""), "rgb")
        )
        link_target = os.path.join(rgb_dir, "001.jpg") if rgb_dir else ""
        rel_link = os.path.relpath(link_target, os.path.dirname(markdown_path)) if link_target else ""
        split_instrs = (
            latest.get("split_instructions")
            or previous.get("split_instructions")
            or train_item.get("sub_instructions", [])
        )

        latest_cut_points = sorted_cut_points(latest.get("cut_points"))
        previous_cut_points = sorted_cut_points(previous.get("cut_points"))
        reference_cut_points = sorted_cut_points(train_item.get("cut_points"))
        lines.append(f"- [{ep_id} 001.jpg]({rel_link})")
        lines.append("  - split_instructions:")
        for instruction in split_instrs:
            lines.append(f'    - "{instruction}"')

        lines.append("  - cut_points:")
        lines.append("    | idx | reference cutpoint | previous cutpoint | latest cutpoint |")
        lines.append("    | --- | --- | --- | --- |")
        max_len = max(len(reference_cut_points), len(previous_cut_points), len(latest_cut_points))
        for idx in range(max_len):
            ref_value = reference_cut_points[idx] if idx < len(reference_cut_points) else ""
            previous_value = previous_cut_points[idx] if idx < len(previous_cut_points) else ""
            latest_value = latest_cut_points[idx] if idx < len(latest_cut_points) else ""
            lines.append(
                f"    | {idx} | {format_frame_label(ref_value)} | {format_frame_label(previous_value)} | {format_frame_label(latest_value)} |"
            )
        lines.append("")

    os.makedirs(os.path.dirname(markdown_path), exist_ok=True)
    with open(markdown_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def load_stair_alignment_cache(path: str) -> Dict[str, dict]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return raw if isinstance(raw, dict) else {}


def load_api_keys(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Qwen API keys file not found: {path}")
    with open(path) as f:
        keys = [line.strip() for line in f if line.strip()]
    if not keys:
        raise ValueError(f"No API keys found in: {path}")
    return keys


def build_stair_filter_prompt(batch: List[dict]) -> str:
    lines = []
    for ep in batch:
        ep_id = str(ep['episode_id'])
        subs = [clean_instruction(s) for s in ep.get('sub_instructions', []) if clean_instruction(s)]
        if not subs:
            subs = ["<empty>"]
        lines.append(f"Episode {ep_id}:")
        lines.extend([f"- {sub}" for sub in subs])
        lines.append("")
    return (
        "For each episode below, determine whether at least one sub-instruction requires going up or down stairs/steps.\n"
        "Only mark true for actual vertical traversal.\n\n"
        f"{chr(10).join(lines)}\n"
        "Output JSON only, keyed by episode id."
    )


def stair_keyword_fallback(item: dict) -> dict:
    matched = extract_vertical_stair_sub_instructions(item)
    return {
        "contains_vertical_stairs": bool(matched),
        "matched_sub_instructions": matched,
        "source": "keyword_fallback",
    }


def parse_stair_filter_response(response_text: str, batch: List[dict]) -> Optional[Dict[str, dict]]:
    try:
        match = re.search(r'\{.*\}', response_text, re.S)
        if not match:
            return None
        obj = json.loads(match.group())
        parsed = {}
        for item in batch:
            ep_id = str(item['episode_id'])
            fallback = stair_keyword_fallback(item)
            raw = obj.get(ep_id)
            if raw is None:
                parsed[ep_id] = fallback
                continue

            if isinstance(raw, bool):
                matched = fallback["matched_sub_instructions"] if raw else []
                parsed[ep_id] = {
                    "contains_vertical_stairs": bool(raw and matched),
                    "matched_sub_instructions": matched,
                    "source": "qwen_bool_strict_motion",
                }
                continue

            if not isinstance(raw, dict):
                parsed[ep_id] = fallback
                continue

            contains = raw.get('contains_vertical_stairs')
            if contains is None:
                contains = raw.get('contains_stairs')
            if contains is None:
                contains = raw.get('is_stair_episode')
            contains = bool(contains)

            matched = raw.get('matched_sub_instructions', [])
            matched = normalize_vertical_stair_matches(item, matched) if contains else []
            contains = bool(contains and matched)

            parsed[ep_id] = {
                "contains_vertical_stairs": contains,
                "matched_sub_instructions": matched,
                "source": "qwen_strict_motion",
            }
        return parsed
    except Exception:
        return None


def classify_stair_episodes_with_qwen(
    episodes: List[dict],
    base_url: str,
    model: str,
    api_keys_file: str,
    batch_size: int,
    cache_file: str,
) -> Dict[str, dict]:
    cache = {}
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            cache = json.load(f)
        print(f"Loaded stair filter cache: {len(cache)} entries from {cache_file}")

    pending = [ep for ep in episodes if str(ep['episode_id']) not in cache]
    if not pending:
        return cache

    if OpenAI is None:
        raise ImportError("openai is required for Qwen stair filtering. Please install openai.")

    api_keys = load_api_keys(api_keys_file)
    clients = [OpenAI(api_key=key, base_url=base_url) for key in api_keys]
    print(f"Qwen stair filter: {len(pending)} uncached episodes, {len(clients)} clients.")

    total_batches = (len(pending) + batch_size - 1) // batch_size
    with tqdm(total=len(pending), unit="ep", desc="Qwen stair filter") as pbar:
        for batch_idx in range(total_batches):
            batch = pending[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            prompt = build_stair_filter_prompt(batch)

            success = False
            for attempt in range(3):
                client = clients[(batch_idx + attempt) % len(clients)]
                try:
                    resp = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": STAIR_FILTER_SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.0,
                        max_tokens=2048,
                    )
                    text = resp.choices[0].message.content.strip()
                    parsed = parse_stair_filter_response(text, batch)
                    if parsed is not None:
                        cache.update(parsed)
                        success = True
                        break
                    print(f"[Stair batch {batch_idx + 1}] Parse failed on attempt {attempt + 1}")
                except Exception as e:
                    print(f"[Stair batch {batch_idx + 1}] API error on attempt {attempt + 1}: {e}")
                    time.sleep(2 ** attempt)

            if not success:
                for item in batch:
                    cache[str(item['episode_id'])] = stair_keyword_fallback(item)

            pbar.update(len(batch))

            if (batch_idx + 1) % 10 == 0 or batch_idx == total_batches - 1:
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                with open(cache_file, 'w') as f:
                    json.dump(cache, f, indent=2, ensure_ascii=False)

    return cache


def load_coordinate_map(path: str) -> Dict[str, List[dict]]:
    return load_coordinate_map_generic(path)


def load_stair_filter_cache(path: str) -> Dict[str, dict]:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def parse_gpu_ids(gpus_arg: str, fallback_gpu: int) -> List[int]:
    values = []
    if gpus_arg:
        for token in str(gpus_arg).split(","):
            token = token.strip()
            if not token:
                continue
            try:
                values.append(int(token))
            except Exception:
                continue
    if not values:
        values = [int(fallback_gpu)]
    return values

# ===== E-SPA alignment =====

class ESPAConfig:
    # Match the current main segmentation weighting used by seg.py.
    lambda_kinetic   = 0.6
    lambda_potential = 0.4
    duration_weight = 0.05
    max_segment_ideal_len_factor = 2.0

class ESPAAligner(nn.Module):
    def __init__(self, config, action_mapping=None):
        super().__init__()
        self.config = config
        self.action_mapping = action_mapping or {}
        self.atomic_actions_3d = {
            'move forward': torch.tensor([0.0, 0.0, -1.0]),
            'forward':      torch.tensor([0.0, 0.0, -1.0]),
            'turn left':    torch.tensor([0.0, 1.0,  0.0]),
            'left':         torch.tensor([0.0, 1.0,  0.0]),
            'turn right':   torch.tensor([0.0, -1.0, 0.0]),
            'right':        torch.tensor([0.0, -1.0, 0.0]),
            'move up':      torch.tensor([0.0, 0.0, -1.0]),
            'up':           torch.tensor([0.0, 0.0, -1.0]),
            'move down':    torch.tensor([0.0, 0.0, -1.0]),
            'down':         torch.tensor([0.0, 0.0, -1.0]),
            'stop':         torch.tensor([0.0, 0.0,  0.0]),
        }
        self.forward_words = [
            'go', 'walk', 'head', 'pass', 'cross', 'enter', 'leave',
            'forward', 'through', 'past', 'towards', 'straight',
            'proceed', 'continue'
        ]
        self.stop_words = ['stop', 'wait', 'end', 'halt', 'stand']

    def _append_action(self, current_set, action_name, atomic_actions):
        action = atomic_actions[action_name]
        if not any(torch.equal(existing, action) for existing in current_set):
            current_set.append(action)

    def _text_to_action_sets(self, sub_instructions: List[str], device):
        atomic_actions = self.atomic_actions_3d
        valid_action_sets = []
        for text in sub_instructions:
            t = text.lower()
            current_set = []

            # The action map is the authoritative, single-primary-action label
            # for a normalized sub-instruction.  Older code appended broad
            # substring matches (e.g. ``left`` in a landmark phrase), which
            # silently turned many R2R single-action instructions into
            # multi-action sets.  Preserve only the explicitly annotated
            # ambiguous turn pair.
            raw_labels = self.action_mapping.get(text)
            if isinstance(raw_labels, str):
                labels = [raw_labels]
            elif isinstance(raw_labels, (list, tuple)):
                labels = list(raw_labels)
            else:
                labels = []
            labels = [label for label in labels if label in atomic_actions]
            if set(labels) == {'turn left', 'turn right'}:
                for label in ('turn left', 'turn right'):
                    self._append_action(current_set, label, atomic_actions)
            elif labels:
                self._append_action(current_set, labels[0], atomic_actions)

            # Keep a deterministic one-action fallback only for texts absent
            # from the action map.  In the released R2R refresh inputs the map
            # has full coverage, so this branch is a robustness fallback, not
            # normal intent expansion.
            if not current_set:
                if any(w in t for w in self.stop_words):
                    fallback_label = 'stop'
                elif 'left' in t:
                    fallback_label = 'left'
                elif 'right' in t:
                    fallback_label = 'right'
                elif any(w in t for w in self.forward_words):
                    fallback_label = 'forward'
                else:
                    fallback_label = 'stop'
                self._append_action(current_set, fallback_label, atomic_actions)
            valid_action_sets.append(torch.stack(current_set).to(device))
        return valid_action_sets

    def compute_semantic_cost_matrix(self, visual_feats, text_feats):
        visual_norm = F.normalize(visual_feats, p=2, dim=1)
        text_norm   = F.normalize(text_feats,   p=2, dim=1)
        similarity  = torch.mm(visual_norm, text_norm.t())
        return -F.log_softmax(similarity / 0.05, dim=1)

    def compute_motion_cost_matrix(self, motion_feats, sub_instructions):
        T = motion_feats.shape[0]
        K = len(sub_instructions)
        device = motion_feats.device
        valid_sets = self._text_to_action_sets(sub_instructions, device)
        K_sem = torch.zeros((T, K), device=device)
        for k in range(K):
            valid_actions = valid_sets[k]
            diff = motion_feats.unsqueeze(1) - valid_actions.unsqueeze(0)
            dists_squared = torch.sum(diff**2, dim=-1)
            min_dists, _ = torch.min(dists_squared, dim=1)
            K_sem[:, k] = 0.5 * min_dists
        return K_sem

    def combine_cost_matrices(self, semantic_cost, motion_cost):
        total = (
            self.config.lambda_kinetic * motion_cost +
            self.config.lambda_potential * semantic_cost
        )
        return total

    def compute_lagrangian_matrix(
        self,
        visual_feats,
        motion_feats,
        text_feats,
        sub_instructions,
    ):
        semantic_cost = self.compute_semantic_cost_matrix(visual_feats, text_feats)
        motion_cost = self.compute_motion_cost_matrix(motion_feats, sub_instructions)
        return self.combine_cost_matrices(semantic_cost, motion_cost)

    def find_optimal_segmentation(self, L_matrix):
        T, K = L_matrix.shape
        cum_L = torch.cumsum(L_matrix, dim=0)

        def get_cost(k, start, end):
            val = cum_L[end, k]
            if start > 0: val = val - cum_L[start-1, k]
            return val.item()

        ideal_len = float(T) / K
        duration_weight = float(getattr(self.config, 'duration_weight', 0.05))
        max_segment_factor = float(getattr(self.config, 'max_segment_ideal_len_factor', 2.0))
        def duration_cost(length):
            return duration_weight * ((length - ideal_len)**2)

        dp     = torch.full((K, T), float('inf'), device=L_matrix.device)
        parent = torch.zeros((K, T), dtype=torch.long, device=L_matrix.device)

        for t in range(T):
            dp[0, t]     = get_cost(0, 0, t) + duration_cost(t+1)
            parent[0, t] = 0

        for k in range(1, K):
            for t in range(k, T):
                search_start = max(k - 1, int(t - max_segment_factor * ideal_len))
                best_cost, best_prev = float('inf'), -1
                for prev_t in range(search_start, t):
                    cost = dp[k-1, prev_t] + get_cost(k, prev_t+1, t) + duration_cost(t - prev_t)
                    if cost < best_cost:
                        best_cost = cost
                        best_prev = prev_t
                dp[k, t]     = best_cost
                parent[k, t] = best_prev + 1

        cut_points, curr_t = [], T - 1
        for k in range(K-1, 0, -1):
            cp = parent[k, curr_t].item() - 1
            cut_points.append(cp)
            curr_t = cp
        return sorted(cut_points)

# ===== Segmenter =====

class R2R_Segmenter:
    def __init__(self, clip_model_path, action_mapping_path=None, gpu_id=DEFAULT_GPU_ID):
        try:
            gpu_id = int(gpu_id)
        except Exception:
            gpu_id = DEFAULT_GPU_ID

        if torch.cuda.is_available() and 0 <= gpu_id < torch.cuda.device_count():
            self.device = f"cuda:{gpu_id}"
        else:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.clip_model     = CLIPModel.from_pretrained(clip_model_path).to(self.device).half()
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_path, use_fast=True)
        if hasattr(torch, 'compile'):
            import os
            os.environ.setdefault("TORCHINDUCTOR_CUDAGRAPH_SKIP_DYNAMIC", "1")
            torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
            self.clip_model.vision_model = torch.compile(
                self.clip_model.vision_model, mode="reduce-overhead"
            )

        action_mapping = {}
        if action_mapping_path and os.path.exists(action_mapping_path):
            with open(action_mapping_path) as f:
                action_mapping = json.load(f)

        self.espa = ESPAAligner(ESPAConfig(), action_mapping=action_mapping).to(self.device)
        # Base motion feature = [reserved, yaw, forward].
        # Stair handling uses API-provided frame-block constraints; action
        # matching always remains 3-D.
        self.action_vectors = {
            -1: [0.0,  0.0,  0.0],
             0: [0.0,  0.0,  0.0],
             1: [0.0,  0.0, -1.0],
             2: [0.0,  1.0,  0.0],
             3: [0.0, -1.0,  0.0],
        }
        print(f"[Segmenter] Loaded on {self.device}")

    def _segment_from_l_matrix(self, l_matrix):
        with torch.inference_mode():
            return self.espa.find_optimal_segmentation(l_matrix)

    def _fallback_cut_points(self, num_frames, num_segments):
        if num_segments <= 0 or num_frames <= 0:
            return []
        if num_segments == 1:
            return [0, num_frames - 1]
        anchors = [1]
        for idx in range(1, num_segments):
            anchors.append(int(round(idx * num_frames / num_segments)))
        anchors.append(num_frames)
        anchors = sorted(min(max(anchor, 1), num_frames) for anchor in anchors)
        deduped = [anchors[0]]
        for anchor in anchors[1:]:
            if anchor > deduped[-1]:
                deduped.append(anchor)
        if deduped[-1] != num_frames:
            deduped.append(num_frames)
        return deduped

    def _safe_segment_from_l_matrix(self, l_matrix):
        num_frames = int(l_matrix.shape[0])
        num_segments = int(l_matrix.shape[1])
        if num_segments == 0 or num_frames == 0:
            return []
        if num_segments == 1:
            return [1, num_frames]
        if num_frames < num_segments:
            return self._fallback_cut_points(num_frames, num_segments)

        local_cut_points = self._segment_from_l_matrix(l_matrix)
        res_cut_points = [1] + [cp + 1 for cp in local_cut_points]
        if num_frames not in res_cut_points:
            res_cut_points.append(num_frames)
        return sorted(set(res_cut_points))

    def _combine_block_cut_points(self, blocks, total_frames):
        combined = [1]
        for block_idx, (start_idx, local_cut_points) in enumerate(blocks):
            if not local_cut_points:
                continue
            cps_to_use = list(local_cut_points)
            is_first = block_idx == 0
            is_last = block_idx == len(blocks) - 1

            # Each local block starts at frame 1 in its own coordinate system.
            # The global sequence already starts at 1, so the first block should
            # never contribute its local leading 1 again.
            if is_first and cps_to_use:
                cps_to_use = cps_to_use[1:]

            # Non-final blocks should not contribute their terminal frame; the
            # next block contributes its own starting cut point instead.
            if not is_last and cps_to_use:
                cps_to_use = cps_to_use[:-1]

            for local_cp in cps_to_use:
                global_cp = start_idx + local_cp
                if 1 <= global_cp <= total_frames:
                    combined.append(global_cp)
        if total_frames not in combined:
            combined.append(total_frames)
        return sorted(set(combined))

    def _normalize_index_list(self, values, upper_bound):
        if not isinstance(values, list):
            return []
        result = []
        for value in values:
            try:
                idx = int(value)
            except Exception:
                continue
            if 0 <= idx < upper_bound:
                result.append(idx)
        return sorted(set(result))

    def _normalize_span_index_list(self, values, spans):
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
                result.append(span_id_to_pos[idx])
            elif 0 <= idx < len(spans):
                result.append(idx)
        return sorted(set(result))

    def _normalize_block_index_list(self, values, candidate_blocks):
        if not isinstance(values, list):
            return []
        valid = {int(block.get("block_index", idx)) for idx, block in enumerate(candidate_blocks)}
        result = []
        for value in values:
            try:
                idx = int(value)
            except Exception:
                continue
            if idx in valid:
                result.append(idx)
        return sorted(set(result))

    def _sanitize_stair_alignment(self, stair_alignment, sub_instructions):
        if not isinstance(stair_alignment, dict):
            return None
        if stair_alignment.get("alignment_type") == "multi_stair_disjoint":
            return None

        num_subs = len(sub_instructions)
        if num_subs == 0:
            return None

        stair_indices = set(self._normalize_index_list(stair_alignment.get("stair_instruction_indices", []), num_subs))
        legacy = set(self._normalize_index_list(stair_alignment.get("stair_core_instruction_indices", []), num_subs))
        if not stair_indices and legacy:
            stair_indices = legacy
        if not stair_indices:
            return None

        height_summary = stair_alignment.get("height_summary", {})
        spans = height_summary.get("spans", []) if isinstance(height_summary, dict) else []
        instruction_labels = [
            "stair_motion_up" if is_up_stair_motion(text) else
            "stair_motion_down" if is_down_stair_motion(text) else
            "stair_endpoint" if is_stair_endpoint_state(text) else
            "stair_progress" if is_stair_progress_state(text) else
            "stair_context" if has_stair_context(text) else
            "non_stair"
            for text in sub_instructions
        ]
        stair_range = trim_noncore_stair_instruction_indices(instruction_labels, stair_indices)
        if not stair_range:
            return None
        stair_start, stair_end = stair_range[0], stair_range[-1]
        pre_indices = list(range(stair_start))
        post_indices = list(range(stair_end + 1, num_subs))
        preferred_labels = infer_stair_directions(instruction_labels, sub_instructions, focus_indices=stair_range)
        candidate_blocks = build_candidate_stair_blocks(height_summary, preferred_labels)
        if not candidate_blocks:
            primary_span_ids = select_primary_stair_span_indices(height_summary, preferred_labels)
            if primary_span_ids:
                candidate_blocks = [{
                    "block_index": 0,
                    "span_indices": primary_span_ids,
                }]
        selected_block_indices = self._normalize_block_index_list(
            stair_alignment.get("selected_block_indices", []),
            candidate_blocks,
        )
        if not selected_block_indices:
            legacy_span_ids = self._normalize_span_index_list(stair_alignment.get("stair_height_span_indices", []), spans)
            legacy_span_set = {int(spans[idx].get("span_index", idx)) for idx in legacy_span_ids}
            if legacy_span_set:
                selected_block_indices = [
                    int(block.get("block_index", idx))
                    for idx, block in enumerate(candidate_blocks)
                    if legacy_span_set & set(int(span_idx) for span_idx in block.get("span_indices", []))
                ]
        if not selected_block_indices and candidate_blocks:
            selected_block_indices = [int(candidate_blocks[0].get("block_index", 0))]

        selected_span_ids = sorted({
            int(span_idx)
            for block in candidate_blocks
            if int(block.get("block_index", -1)) in set(selected_block_indices)
            for span_idx in block.get("span_indices", [])
        })
        span_indices = self._normalize_span_index_list(selected_span_ids, spans)
        valid_span_indices = []
        for span_idx in span_indices:
            span = spans[span_idx]
            if span.get("label") in {"up", "down"}:
                valid_span_indices.append(span_idx)

        if not valid_span_indices:
            valid_span_indices = [
                idx for idx, span in enumerate(spans)
                if isinstance(span, dict) and span.get("label") in {"up", "down"}
            ]
        if not valid_span_indices:
            return None

        merge_selected_blocks = bool(
            stair_alignment.get("merge_selected_blocks", stair_alignment.get("merge_to_single_block", False))
        )
        if merge_selected_blocks and has_strong_opposite_span_between_blocks(spans, candidate_blocks, selected_block_indices):
            selected_block_indices = choose_primary_block_indices(candidate_blocks, selected_block_indices)
            selected_span_ids = sorted({
                int(span_idx)
                for block in candidate_blocks
                if int(block.get("block_index", -1)) in set(selected_block_indices)
                for span_idx in block.get("span_indices", [])
            })
            span_indices = self._normalize_span_index_list(selected_span_ids, spans)
            valid_span_indices = [
                span_idx for span_idx in span_indices
                if spans[span_idx].get("label") in {"up", "down"}
            ]
            merge_selected_blocks = False
        if not valid_span_indices:
            return None

        return {
            "stair_indices": stair_range,
            "pre_indices": pre_indices,
            "post_indices": post_indices,
            "height_spans": spans,
            "height_span_indices": valid_span_indices,
            "candidate_blocks": candidate_blocks,
            "selected_block_indices": selected_block_indices,
            "source": stair_alignment.get("source", "api_alignment"),
        }

    def _segment_stair_with_alignment(
        self,
        visual_feats,
        l_matrix_3d,
        sub_instructions,
        stair_alignment,
    ):
        normalized = self._sanitize_stair_alignment(stair_alignment, sub_instructions)
        if normalized is None:
            return None

        spans = normalized["height_spans"]
        span_indices = normalized["height_span_indices"]
        num_frames = int(visual_feats.shape[0])
        if not spans or not span_indices or num_frames <= 1:
            return None

        core_frame_start = min(int(spans[idx]["start_frame"]) for idx in span_indices)
        core_frame_end = max(int(spans[idx]["end_frame"]) for idx in span_indices)
        if core_frame_start >= core_frame_end:
            return None

        min_pre_frames = len(normalized["pre_indices"])
        min_post_frames = len(normalized["post_indices"])
        stair_frame_start = max(min_pre_frames, core_frame_start)
        stair_frame_end = min(num_frames - 1 - min_post_frames, core_frame_end)
        if stair_frame_start >= stair_frame_end:
            stair_frame_start = max(0, core_frame_start)
            stair_frame_end = min(num_frames - 1, core_frame_end)

        if normalized["pre_indices"] and stair_frame_start < len(normalized["pre_indices"]):
            return None
        if normalized["post_indices"] and (num_frames - 1 - stair_frame_end) < len(normalized["post_indices"]):
            return None

        blocks = []
        if normalized["pre_indices"]:
            pre_cut_points = self._safe_segment_from_l_matrix(
                l_matrix_3d[:stair_frame_start, normalized["pre_indices"]],
            )
            blocks.append((0, pre_cut_points))

        stair_cut_points = self._safe_segment_from_l_matrix(
            l_matrix_3d[stair_frame_start:stair_frame_end + 1, normalized["stair_indices"]],
        )
        if not stair_cut_points:
            return None
        blocks.append((stair_frame_start, stair_cut_points))

        if normalized["post_indices"]:
            post_frame_start = stair_frame_end + 1
            if post_frame_start >= num_frames:
                return None
            post_cut_points = self._safe_segment_from_l_matrix(
                l_matrix_3d[post_frame_start:, normalized["post_indices"]],
            )
            blocks.append((post_frame_start, post_cut_points))

        return self._combine_block_cut_points(blocks, num_frames)

    def _segment_multi_stair_disjoint_with_alignment(
        self,
        visual_feats,
        l_matrix_3d,
        sub_instructions,
        stair_alignment,
    ):
        if not isinstance(stair_alignment, dict):
            return None
        if stair_alignment.get("alignment_type") != "multi_stair_disjoint":
            return None

        raw_instruction_spans = stair_alignment.get("disjoint_stair_instruction_spans", [])
        if not isinstance(raw_instruction_spans, list) or not raw_instruction_spans:
            return None

        num_subs = len(sub_instructions)
        num_frames = int(visual_feats.shape[0])
        if num_subs == 0 or num_frames <= 1:
            return None

        instruction_spans = []
        for raw_span in raw_instruction_spans:
            if not isinstance(raw_span, list) or len(raw_span) != 2:
                return None
            try:
                start = int(raw_span[0])
                end = int(raw_span[1])
            except Exception:
                return None
            if not (0 <= start <= end < num_subs):
                return None
            instruction_spans.append((start, end))
        instruction_spans = sorted(instruction_spans)

        candidate_blocks = stair_alignment.get("candidate_stair_blocks", [])
        selected_block_indices = self._normalize_block_index_list(
            stair_alignment.get("selected_block_indices", []),
            candidate_blocks,
        )
        if len(selected_block_indices) != len(instruction_spans):
            return None

        selected_blocks = []
        for block_idx in selected_block_indices:
            block = next(
                (
                    candidate for candidate in candidate_blocks
                    if int(candidate.get("block_index", -1)) == int(block_idx)
                ),
                None,
            )
            if block is None:
                return None
            try:
                block_start = int(block.get("start_frame", -1))
                block_end = int(block.get("end_frame", -1))
            except Exception:
                return None
            if block_start < 0 or block_end < block_start:
                return None
            block_start = min(max(block_start, 0), num_frames - 1)
            block_end = min(max(block_end, 0), num_frames - 1)
            selected_blocks.append((block_start, block_end))
        selected_blocks = sorted(selected_blocks, key=lambda x: x[0])

        first_sub_start = instruction_spans[0][0]
        first_frame_start = selected_blocks[0][0]
        if first_sub_start > 0:
            if first_frame_start <= 0:
                return None
            pre_cut_points = self._safe_segment_from_l_matrix(
                l_matrix_3d[:first_frame_start, :first_sub_start],
            )
            if not pre_cut_points:
                return None
            blocks = [(0, pre_cut_points)]
        else:
            blocks = []

        num_groups = len(instruction_spans)
        for group_idx, (sub_span, frame_span) in enumerate(zip(instruction_spans, selected_blocks)):
            sub_start, sub_end = sub_span
            frame_start, frame_end = frame_span

            if group_idx == 0 and sub_start == 0:
                frame_start = 0
            if group_idx == num_groups - 1 and sub_end == num_subs - 1:
                frame_end = num_frames - 1
            if frame_start > frame_end:
                return None

            stair_cut_points = self._safe_segment_from_l_matrix(
                l_matrix_3d[frame_start:frame_end + 1, sub_start:sub_end + 1],
            )
            if not stair_cut_points:
                return None
            blocks.append((frame_start, stair_cut_points))

            if group_idx + 1 >= num_groups:
                continue

            next_sub_start, _ = instruction_spans[group_idx + 1]
            next_frame_start, _ = selected_blocks[group_idx + 1]
            gap_sub_start = sub_end + 1
            gap_sub_end = next_sub_start - 1
            gap_frame_start = frame_end + 1
            gap_frame_end = next_frame_start - 1

            if gap_sub_start > gap_sub_end:
                if gap_frame_start <= gap_frame_end:
                    return None
                continue

            if gap_frame_start > gap_frame_end:
                return None

            gap_cut_points = self._safe_segment_from_l_matrix(
                l_matrix_3d[gap_frame_start:gap_frame_end + 1, gap_sub_start:gap_sub_end + 1],
            )
            if not gap_cut_points:
                return None
            blocks.append((gap_frame_start, gap_cut_points))

        last_sub_end = instruction_spans[-1][1]
        last_frame_end = selected_blocks[-1][1]
        if last_sub_end < num_subs - 1:
            post_frame_start = last_frame_end + 1
            if post_frame_start >= num_frames:
                return None
            post_cut_points = self._safe_segment_from_l_matrix(
                l_matrix_3d[post_frame_start:, last_sub_end + 1:],
            )
            if not post_cut_points:
                return None
            blocks.append((post_frame_start, post_cut_points))

        return self._combine_block_cut_points(blocks, num_frames)

    def _project_frame_interval_after_initial_mask(self, start_frame, end_frame, frame_offset: int, local_num_frames: int):
        try:
            start_frame = int(start_frame)
            end_frame = int(end_frame)
        except Exception:
            return None
        if local_num_frames <= 0 or end_frame < frame_offset:
            return None

        local_start = max(0, start_frame - frame_offset)
        local_end = min(local_num_frames - 1, end_frame - frame_offset)
        if local_start > local_end or local_start >= local_num_frames or local_end < 0:
            return None
        return local_start, local_end

    def _project_stair_alignment_after_initial_mask(self, stair_alignment, frame_offset: int, total_frames: int):
        if not isinstance(stair_alignment, dict) or frame_offset <= 0:
            return stair_alignment

        local_num_frames = total_frames - frame_offset
        if local_num_frames <= 0:
            return stair_alignment

        projected = copy.deepcopy(stair_alignment)

        height_summary = projected.get("height_summary")
        if isinstance(height_summary, dict):
            spans = height_summary.get("spans", [])
            if isinstance(spans, list):
                projected_spans = []
                for span in spans:
                    if not isinstance(span, dict):
                        continue
                    interval = self._project_frame_interval_after_initial_mask(
                        span.get("start_frame"),
                        span.get("end_frame"),
                        frame_offset,
                        local_num_frames,
                    )
                    if interval is None:
                        continue
                    span_copy = dict(span)
                    span_copy["start_frame"], span_copy["end_frame"] = interval
                    projected_spans.append(span_copy)
                height_summary["spans"] = projected_spans
                height_summary["num_frames"] = local_num_frames

        candidate_blocks = projected.get("candidate_stair_blocks")
        if isinstance(candidate_blocks, list):
            projected_blocks = []
            for block in candidate_blocks:
                if not isinstance(block, dict):
                    continue
                interval = self._project_frame_interval_after_initial_mask(
                    block.get("start_frame"),
                    block.get("end_frame"),
                    frame_offset,
                    local_num_frames,
                )
                if interval is None:
                    continue
                block_copy = dict(block)
                block_copy["start_frame"], block_copy["end_frame"] = interval
                projected_blocks.append(block_copy)
            projected["candidate_stair_blocks"] = projected_blocks

        return projected

    def _initial_turn_mask_frame_count_from_metadata(self, initial_turn_mask, num_frames: int) -> int:
        if not isinstance(initial_turn_mask, dict) or not initial_turn_mask.get("enabled"):
            return 0
        if num_frames <= 1:
            return 0

        value = initial_turn_mask.get("masked_frame_count")
        if value in ("", None):
            frame_range = initial_turn_mask.get("masked_frame_range")
            if isinstance(frame_range, (list, tuple)) and len(frame_range) == 2:
                try:
                    start = int(frame_range[0])
                    end = int(frame_range[1])
                    value = end - start + 1
                except Exception:
                    value = 0
        try:
            mask_count = int(value)
        except Exception:
            return 0

        if mask_count <= 0 or mask_count >= num_frames:
            return 0
        return mask_count

    def _shift_local_cut_points_after_initial_mask(self, local_cut_points, mask_frame_count: int, total_frames: int):
        if mask_frame_count <= 0:
            return sorted(set(local_cut_points))

        shifted = [1]
        for cp in local_cut_points:
            try:
                local_cp = int(cp)
            except Exception:
                continue
            if local_cp <= 1:
                continue
            global_cp = local_cp + mask_frame_count
            if 1 < global_cp <= total_frames:
                shifted.append(global_cp)
        if total_frames not in shifted:
            shifted.append(total_frames)
        return sorted(set(shifted))

    def segment(
        self,
        image_paths,
        sub_instructions,
        actions,
        use_vertical_motion=False,
        stair_alignment=None,
        initial_turn_mask=None,
    ):
        def _load_image(p):
            return Image.open(p).convert("RGB")

        with ThreadPoolExecutor(max_workers=8) as executor:
            images = list(executor.map(_load_image, image_paths))
        T = len(images)
        total_frames = T

        batch_size = 64
        visual_feats_list = []
        for i in range(0, T, batch_size):
            batch = images[i:i+batch_size]
            inputs_v = self.clip_processor(images=batch, return_tensors="pt", padding=True).to(self.device)
            with torch.inference_mode():
                visual_feats_list.append(self.clip_model.get_image_features(**inputs_v))
        visual_feats = torch.cat(visual_feats_list, dim=0)

        prompts  = [f"A first-person view of {t}" for t in sub_instructions]
        inputs_t = self.clip_processor(
            text=prompts, return_tensors="pt", padding=True, truncation=True, max_length=77
        ).to(self.device)
        with torch.inference_mode():
            text_feats = self.clip_model.get_text_features(**inputs_t)

        if len(actions) < T:
            actions = actions + [0] * (T - len(actions))
        else:
            actions = list(actions[:T])

        initial_turn_mask_frame_count = self._initial_turn_mask_frame_count_from_metadata(initial_turn_mask, T)
        local_stair_alignment = stair_alignment
        if initial_turn_mask_frame_count > 0:
            visual_feats = visual_feats[initial_turn_mask_frame_count:]
            actions = actions[initial_turn_mask_frame_count:]
            images = images[initial_turn_mask_frame_count:]
            T = int(visual_feats.shape[0])
            local_stair_alignment = self._project_stair_alignment_after_initial_mask(
                stair_alignment,
                initial_turn_mask_frame_count,
                total_frames,
            )

        motion_feats_3d = torch.tensor(
            [self.action_vectors.get(a, [0, 0, 0]) for a in actions[:T]],
            dtype=torch.float32, device=self.device
        )
        with torch.inference_mode():
            semantic_cost = self.espa.compute_semantic_cost_matrix(visual_feats, text_feats)
            motion_cost_3d = self.espa.compute_motion_cost_matrix(
                motion_feats_3d,
                sub_instructions,
            )
            l_matrix_3d = self.espa.combine_cost_matrices(semantic_cost, motion_cost_3d)

        l_matrix = l_matrix_3d
        if use_vertical_motion:
            skip_stair_special = isinstance(local_stair_alignment, dict) and (
                local_stair_alignment.get("alignment_type") == "multi_stair_disjoint"
            )
            if skip_stair_special:
                multi_stair_cut_points = self._segment_multi_stair_disjoint_with_alignment(
                    visual_feats,
                    l_matrix_3d,
                    sub_instructions,
                    local_stair_alignment,
                )
                if multi_stair_cut_points is not None:
                    return self._shift_local_cut_points_after_initial_mask(
                        multi_stair_cut_points,
                        initial_turn_mask_frame_count,
                        total_frames,
                    )
            else:
                stair_api_cut_points = self._segment_stair_with_alignment(
                    visual_feats,
                    l_matrix_3d,
                    sub_instructions,
                    local_stair_alignment,
                )
                if stair_api_cut_points is not None:
                    return self._shift_local_cut_points_after_initial_mask(
                        stair_api_cut_points,
                        initial_turn_mask_frame_count,
                        total_frames,
                    )

        cut_points = self._segment_from_l_matrix(l_matrix)
        res_cut_points = [1] + [cp + 1 for cp in cut_points]
        if T not in res_cut_points:
            res_cut_points.append(T)
        res_cut_points = sorted(set(res_cut_points))
        return self._shift_local_cut_points_after_initial_mask(
            res_cut_points,
            initial_turn_mask_frame_count,
            total_frames,
        )

# ===== Worker =====

def worker_func(
    queue,
    lock,
    result_dict,
    clip_path,
    action_map_path,
    images_base_dir,
    ann_by_id,
    coord_by_ep_id,
    gpu_id,
    dataset,
    per_episode_output_filename="",
):
    try:
        segmenter = R2R_Segmenter(clip_path, action_map_path, gpu_id)
    except Exception as e:
        print(f"[Worker] Failed to load model: {e}")
        return

    while True:
        try:
            item = queue.get_nowait()
        except Exception:
            break

        ep_id = str(item['episode_id'])
        data_path = item.get('data_path') or infer_data_path(item, dataset)
        scan_id = item.get('scan_id') or infer_scan_id(item)
        instruction = item.get('original_instruction', item.get('instruction', ''))
        split_instrs = [clean_instruction(s) for s in item.get('sub_instructions', [])]

        if not split_instrs:
            print(f"[Worker] Episode {ep_id}: no sub_instructions, skip.")
            continue

        rgb_dir = item.get("rgb_dir") or build_rgb_dir(item, images_base_dir, dataset)
        if not os.path.isdir(rgb_dir):
            print(f"[Worker] Episode {ep_id}: image dir not found: {rgb_dir}")
            continue

        image_files = sorted(
            [f for f in os.listdir(rgb_dir) if f.lower().endswith(('.jpg','.png')) and os.path.splitext(f)[0].isdigit()],
            key=lambda x: int(os.path.splitext(x)[0])
        )
        image_paths = [os.path.join(rgb_dir, f) for f in image_files]
        if not image_paths:
            print(f"[Worker] Episode {ep_id}: no images in {rgb_dir}")
            continue

        annotation_key = str(item.get("annotation_key") or ep_id)
        coordinate_key = str(item.get("coordinate_key") or ep_id)
        ann = ann_by_id.get(annotation_key) or ann_by_id.get(ep_id)
        actions = ann.get('actions', []) if ann else []
        if actions and actions[0] == -1:
            actions = actions[1:]
        initial_turn_mask = item.get("initial_turn_mask")
        initial_turn_mask_frame_count = segmenter._initial_turn_mask_frame_count_from_metadata(
            initial_turn_mask,
            len(image_paths),
        )
        use_vertical_motion = bool(item.get('use_vertical_motion', False))
        positions = item.get('positions')
        if use_vertical_motion and positions is None and ann:
            positions = ann.get('positions') or ann.get('coordinates') or ann.get('trajectory')
        if use_vertical_motion and positions is None:
            positions = coord_by_ep_id.get(coordinate_key) or coord_by_ep_id.get(ep_id)
        stair_alignment = item.get('stair_alignment')

        scene_id = item.get("scene_id") or (f"mp3d/{scan_id}/{scan_id}.glb" if scan_id else "")
        print(f"[Worker] Episode {ep_id} | {len(image_paths)} frames | {len(split_instrs)} subs")
        try:
            cut_points = segmenter.segment(
                image_paths,
                split_instrs,
                actions,
                use_vertical_motion=use_vertical_motion,
                stair_alignment=stair_alignment,
                initial_turn_mask=initial_turn_mask,
            )
            cut_points = normalize_cut_points_for_instruction_count(
                cut_points,
                len(image_paths),
                len(split_instrs),
            )
            normalized_split_instrs = merge_trailing_unmapped_hold_instructions(
                split_instrs,
                cut_points,
                len(image_paths),
            )
            cut_points = normalize_cut_points_for_instruction_count(
                cut_points,
                len(image_paths),
                len(normalized_split_instrs),
            )
            segment_ranges = build_segment_ranges_from_cut_points(cut_points, len(image_paths))
            cut_points_detail_map = build_cut_points_details(cut_points, positions)
            try:
                episode_value = int(ep_id)
            except Exception:
                episode_value = ep_id
            entry = {
                "episode_id":         episode_value,
                "scan_id":            scan_id,
                "original_instruction": instruction,
                "sub_instructions":   {i: text for i, text in enumerate(normalized_split_instrs)},
                "scene_id":           scene_id,
                "instruction":        instruction,
                "split_instructions": normalized_split_instrs,
                "cut_points":         {i: cp for i, cp in enumerate(cut_points)},
                "cut_points_details": cut_points_detail_map,
                "cut_point_ranges":   build_cut_point_ranges(cut_points, len(image_paths)),
                "segment_ranges":     segment_ranges,
                "instruction_segments": build_instruction_segments(normalized_split_instrs, cut_points, len(image_paths)),
                "num_frames":         len(image_paths),
                "rgb_dir":            rgb_dir,
                "data_path":          data_path,
            }
            if isinstance(initial_turn_mask, dict) and initial_turn_mask.get("enabled"):
                entry["initial_turn_mask"] = {
                    **initial_turn_mask,
                    "masked_frame_count": int(initial_turn_mask_frame_count),
                    "masked_action_count": int(initial_turn_mask_frame_count),
                    "masked_frame_range": [1, int(initial_turn_mask_frame_count)]
                    if initial_turn_mask_frame_count > 0 else [],
                    "masked_frame_range_label": (
                        f"1-{int(initial_turn_mask_frame_count)}"
                        if initial_turn_mask_frame_count > 0 else ""
                    ),
                    "segmentation_start_frame": int(initial_turn_mask_frame_count) + 1,
                    "merged_into_sub_instruction_index": 0,
                }
            if per_episode_output_filename:
                per_episode_path = os.path.join(rgb_dir, per_episode_output_filename)
                atomic_json_dump(entry, per_episode_path)
            with lock:
                result_dict[ep_id] = entry
        except Exception as e:
            print(f"[Worker] Episode {ep_id} error: {e}")
            import traceback; traceback.print_exc()

# ===== Main =====

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, choices=tuple(sorted(STAIR_DATASET_CONFIGS)))
    parser.add_argument("--input_file", type=str, default="",
                        help="Segmentation dataset file containing episode metadata and sub_instructions.")
    parser.add_argument("--annotation_file", type=str, default="")
    parser.add_argument("--coord_file", type=str, default="")
    parser.add_argument("--images_base_dir", type=str, default="")
    parser.add_argument("--action_map_file", type=str, default="")
    parser.add_argument("--gpu", type=int, default=DEFAULT_GPU_ID)
    parser.add_argument("--gpus", type=str, default=DEFAULT_GPU_IDS,
                        help="Comma-separated GPU ids for worker assignment, e.g. 2,3,4,5")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--episode", type=str, default="")
    parser.add_argument("--output_file", type=str, default="",
                        help="Write segmentation results to this file instead of the default output.")
    parser.add_argument("--ignore_existing", action="store_true",
                        help="Do not skip episode ids already present in the chosen output file.")
    parser.add_argument("--stairs_only_output", action="store_true",
                        help="Write a stair-only comparison result file.")
    parser.add_argument("--all_episodes", action="store_true",
                        help="Skip Qwen stair filtering and segment all pending episodes.")
    parser.add_argument("--only_stair_filter", action="store_true",
                        help="Only run the Qwen stair episode analysis and update the cache.")
    parser.add_argument("--base_url", type=str, default=DEFAULT_QWEN_BASE_URL)
    parser.add_argument("--model", type=str, default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--api_keys_file", type=str, default=DEFAULT_QWEN_API_KEYS)
    parser.add_argument("--stair_batch_size", type=int, default=DEFAULT_STAIR_BATCH)
    parser.add_argument("--stair_filter_cache", type=str, default="")
    parser.add_argument("--stair_alignment_file", type=str, default="")
    parser.add_argument("--per_episode_output_filename", type=str, default="")
    args = parser.parse_args()

    dataset_cfg = get_stair_dataset_config(args.dataset)
    input_file = args.input_file or dataset_cfg["input_file"]
    annotation_file = args.annotation_file or dataset_cfg["annotation_file"]
    coord_file = args.coord_file or dataset_cfg["coord_file"]
    images_base_dir = args.images_base_dir or dataset_cfg["images_base_dir"]
    action_map_file = args.action_map_file or dataset_cfg["action_map_file"]
    stair_filter_cache = args.stair_filter_cache or dataset_cfg["stair_filter_cache"]
    stair_alignment_file = args.stair_alignment_file or dataset_cfg["stair_alignment_file"]
    per_episode_output_filename = (
        args.per_episode_output_filename
        if args.per_episode_output_filename
        else dataset_cfg["per_episode_output_filename"]
    )

    gpu_ids = parse_gpu_ids(args.gpus, args.gpu)
    output_file = args.output_file or (
        dataset_cfg["stair_only_output_file"] if args.stairs_only_output else dataset_cfg["segmentation_output_file"]
    )
    should_refresh_markdown = (
        args.dataset == "r2r"
        and args.stairs_only_output
        and (not args.episode)
        and (output_file == dataset_cfg["stair_only_output_file"])
    )

    if args.stairs_only_output and args.episode and not args.output_file:
        raise ValueError(
            "--episode with --stairs_only_output requires --output_file to avoid overwriting "
            f"{dataset_cfg['stair_only_output_file']} and {dataset_cfg['stair_compare_markdown_file']}."
        )

    train_seg = load_episode_dataset(input_file, args.dataset)
    train_seg_by_id = {str(ep['episode_id']): ep for ep in train_seg}

    ann_by_id = load_annotation_index(annotation_file)
    print(f"Loaded {len(train_seg)} train episodes, {len(ann_by_id)} annotations.")

    # Load existing results
    result = {}
    markdown_reference_result = {}
    if should_refresh_markdown and os.path.exists(output_file):
        with open(output_file) as f:
            markdown_reference_result = json.load(f)
    if not args.ignore_existing and os.path.exists(output_file):
        with open(output_file) as f:
            result = json.load(f)
        print(f"Loaded {len(result)} existing results from {output_file}.")
    else:
        print(f"Starting a fresh result set for {output_file}.")

    # Select episodes
    if args.episode:
        target_ids = set(s.strip() for s in args.episode.split(","))
        selected_episodes = [ep for ep in train_seg if str(ep['episode_id']) in target_ids]
    elif args.only_stair_filter:
        selected_episodes = list(train_seg)
    else:
        done = set(result.keys())
        selected_episodes = [ep for ep in train_seg if str(ep['episode_id']) not in done]

    print(f"Selected episodes before stair filter: {len(selected_episodes)}")
    stair_cache = None
    if not args.all_episodes and selected_episodes:
        stair_cache = classify_stair_episodes_with_qwen(
            selected_episodes,
            base_url=args.base_url,
            model=args.model,
            api_keys_file=args.api_keys_file,
            batch_size=args.stair_batch_size,
            cache_file=stair_filter_cache,
        )
        stair_episodes = [
            ep for ep in selected_episodes
            if stair_cache.get(str(ep['episode_id']), {}).get('contains_vertical_stairs', False)
        ]
        print(f"Stair-filtered episodes: {len(stair_episodes)} / {len(selected_episodes)}")
    else:
        stair_episodes = list(selected_episodes)
        stair_cache = load_stair_filter_cache(stair_filter_cache)

    if args.only_stair_filter:
        if stair_cache is None:
            stair_cache = {}
        print(f"Stair filter done. Cached episodes: {len(stair_cache)}")
        print(f"Episodes with vertical stairs: {len(stair_episodes)}")
        return

    if args.limit > 0:
        pending = stair_episodes[:args.limit]
    else:
        pending = stair_episodes

    stair_episode_ids = {
        ep_id for ep_id, info in (stair_cache or {}).items()
        if info.get('contains_vertical_stairs', False)
    }
    stair_alignment_cache = load_stair_alignment_cache(stair_alignment_file)
    pending = [
        dict(
            ep,
            use_vertical_motion=(str(ep['episode_id']) in stair_episode_ids),
            stair_alignment=stair_alignment_cache.get(str(ep['episode_id'])),
        )
        for ep in pending
    ]

    needs_coordinates = any(ep.get('use_vertical_motion', False) for ep in pending)
    coord_by_ep_id = load_coordinate_map(coord_file) if needs_coordinates else {}
    if needs_coordinates:
        print(f"Loaded {len(coord_by_ep_id)} coordinate trajectories for stair episodes.")
    else:
        print("No stair episodes in current pending set; keep original 3D motion features.")

    print(f"Pending: {len(pending)}")
    if not pending:
        print("All done.")
        if should_refresh_markdown and os.path.exists(output_file):
            with open(output_file) as f:
                latest_existing = json.load(f)
            comparison_reference = markdown_reference_result
            if comparison_reference == latest_existing:
                comparison_reference = {}
            write_stair_compare_markdown(
                dataset_cfg["stair_compare_markdown_file"],
                latest_results=latest_existing,
                train_seg_by_id=train_seg_by_id,
                previous_results=comparison_reference,
            )
            print(f"Updated markdown comparison: {dataset_cfg['stair_compare_markdown_file']}")
        return

    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    queue       = manager.Queue()
    lock        = manager.Lock()
    result_dict = manager.dict(result)

    for ep in pending:
        queue.put(ep)

    workers = []
    for i in range(min(args.workers, len(pending))):
        assigned_gpu = gpu_ids[i % len(gpu_ids)]
        p = mp.Process(
            target=worker_func,
            args=(queue, lock, result_dict, CLIP_MODEL_PATH,
                  action_map_file, images_base_dir, ann_by_id, coord_by_ep_id, assigned_gpu,
                  args.dataset, per_episode_output_filename)
        )
        p.start()
        workers.append(p)
        print(f"Started worker {i} on gpu {assigned_gpu}")

    last_save = time.time()
    with tqdm(total=len(pending)) as pbar:
        last_count = 0
        while True:
            alive = any(p.is_alive() for p in workers)
            curr = len(result_dict) - len(result)
            if curr > last_count:
                pbar.update(curr - last_count)
                last_count = curr
            if not alive and queue.empty():
                break
            if time.time() - last_save > 30:
                combined = dict(result)
                combined.update(dict(result_dict))
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with open(output_file, 'w') as f:
                    json.dump(combined, f, indent=2, ensure_ascii=False)
                last_save = time.time()
            time.sleep(1)

    for p in workers:
        p.join()

    combined = dict(result)
    combined.update(dict(result_dict))
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    print(f"Done. {len(combined)} episodes -> {output_file}")

    if should_refresh_markdown:
        comparison_reference = markdown_reference_result
        if comparison_reference == combined:
            comparison_reference = {}
        write_stair_compare_markdown(
            dataset_cfg["stair_compare_markdown_file"],
            latest_results=combined,
            train_seg_by_id=train_seg_by_id,
            previous_results=comparison_reference,
        )
        print(f"Updated markdown comparison: {dataset_cfg['stair_compare_markdown_file']}")
    elif args.stairs_only_output:
        print(
            "Skipped markdown refresh because this was an episode-scoped or custom-output stair run. "
            f"Results were written only to {output_file}"
        )

if __name__ == "__main__":
    main()
