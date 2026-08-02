#!/usr/bin/env python3
"""Evaluate static M1 sub-instruction QA predictions with MiniLM similarity.

Prediction input is JSON, JSONL, or a JSON object keyed by sample id.  The
minimal JSONL schema is:

    {"episode_id": 4, "frame": 17,
     "response": "<answer>Turn left.</answer>"}

``--segmentation`` derives the gold M1 answer from the frame and the segment
boundaries.  Alternatively, provide ``gold_answer`` in every prediction row
(or select another field with ``--gold-field``).  Only the content of
``<answer>...</answer>`` is evaluated; a leading ``Recovering:`` is ignored.

For normal sub-instructions, the main metric is cosine similarity from the
local SentenceTransformer model.  STOP is symbolic, so it is scored exactly
(1 or 0), rather than embedding ``[STOP]``.  Besides the main score, the
report separates exact matches and lexical containment direction:

* prediction_contains_gold: prediction is lexically more detailed;
* gold_contains_prediction: prediction is lexically coarser.

This avoids hiding granularity errors behind a single high embedding score.
"""

import argparse
import base64
import json
import math
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import requests
from PIL import Image


ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
THINK_PATTERN = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
RECOVERING_PATTERN = re.compile(r"^recovering\s*:\s*", re.IGNORECASE)
NON_WORD_PATTERN = re.compile(r"[^a-z0-9]+")
STOP = "[STOP]"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", type=Path, help="JSON/JSONL prediction records (offline scoring mode).")
    parser.add_argument(
        "--segmentation",
        type=Path,
        help="Continuous segmentation JSON; derives gold from episode_id + frame.",
    )
    parser.add_argument(
        "--gold-field",
        default="",
        help="Use this field (supports dotted paths) as gold instead of --segmentation.",
    )
    parser.add_argument(
        "--prediction-field",
        default="",
        help="Prediction field (supports dotted paths). Default auto-detects response/prediction/output/answer.",
    )
    parser.add_argument("--episode-id-field", default="episode_id")
    parser.add_argument("--frame-field", default="frame")
    parser.add_argument("--model", default="model_zoo/all-MiniLM-L6-v2")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--thresholds",
        default="0.5,0.6,0.7,0.8,0.9",
        help="Comma-separated cosine thresholds for pass-rate reporting.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        help="Writes <prefix>.summary.json and <prefix>.details.jsonl. Defaults beside predictions.",
    )
    parser.add_argument("--image-root", type=Path, help="Image root for direct vLLM evaluation.")
    parser.add_argument("--vllm-url", default="", help="OpenAI-compatible vLLM server URL.")
    parser.add_argument("--vllm-model", default="", help="Model name served by vLLM.")
    parser.add_argument("--vllm-workers", type=int, default=8)
    parser.add_argument("--vllm-timeout-s", type=float, default=180.0)
    parser.add_argument("--vllm-max-tokens", type=int, default=256)
    parser.add_argument("--vllm-temperature", type=float, default=0.0)
    parser.add_argument(
        "--task-type",
        choices=("m1_subinstruction", "single_m1"),
        default="m1_subinstruction",
        help=("m1_subinstruction uses the dual-system History trajectory/Current view input; "
              "single_m1 uses the single model's global-uniform Observation history input. "
              "Both use the same M1 output and text metrics."),
    )
    parser.add_argument(
        "--frame-policy",
        choices=("segment_end", "segment_start", "both", "all", "random_per_segment"),
        default="segment_end",
        help="Static frames to query in vLLM mode; segment_end tests next-sub-instruction handoff.",
    )
    parser.add_argument(
        "--sampled-frames",
        type=Path,
        help=("Persistent JSON for --frame-policy random_per_segment. It is created with --sampling-seed "
              "when absent, then reused unchanged by later evaluations."),
    )
    parser.add_argument("--sampling-seed", type=int, default=42,
                        help="Seed used only when creating --sampled-frames (default: 42).")
    parser.add_argument("--sample-target", type=int, default=15000,
                        help="Target number of frames for random_per_segment sampling (default: 15000).")
    parser.add_argument(
        "--sample-focus",
        choices=("uniform_length", "perturbation", "deviated_only"),
        default="uniform_length",
        help=("Frame priority for random_per_segment sampling. deviated_only selects only perturbation/reconnect "
              "frames and does not require coverage of every segment (default: uniform_length)."),
    )
    parser.add_argument("--prepare-samples-only", action="store_true",
                        help="Create or validate --sampled-frames, then exit without inference or scoring.")
    parser.add_argument("--save-every", type=int, default=100,
                        help="Write direct-vLLM predictions every N completed samples (default: 100).")
    parser.add_argument("--boundary-window", type=int, default=2,
                        help="Frames on either side of a step boundary that accept either adjacent step (default: 2).")
    parser.add_argument(
        "--top2-max-similarity-gap",
        type=float,
        default=0.1,
        help=("Include rank 2 in Top-2 only when its route-step cosine similarity is within this gap "
              "of rank 1 (default: 0.1)."),
    )
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all selected static frames.")
    parser.add_argument("--strict", action="store_true", help="Fail if any prediction cannot be scored.")
    return parser.parse_args()


def load_records(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return []
    if path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in content.splitlines() if line.strip()]
    else:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            records = parsed
        elif isinstance(parsed, dict):
            # Support both a single record and {sample_id: prediction} maps.
            if any(key in parsed for key in ("response", "prediction", "output", "answer")):
                records = [parsed]
            else:
                records = []
                for sample_id, value in parsed.items():
                    record = dict(value) if isinstance(value, dict) else {"prediction": value}
                    record.setdefault("id", sample_id)
                    records.append(record)
        else:
            raise ValueError(f"Unsupported prediction payload: {type(parsed).__name__}")
    if not all(isinstance(record, dict) for record in records):
        raise ValueError("Every prediction record must be a JSON object")
    return records


def get_field(record: Dict[str, Any], dotted_path: str) -> Any:
    value: Any = record
    for key in dotted_path.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def first_field(record: Dict[str, Any], requested: str, candidates: Sequence[str]) -> Any:
    if requested:
        return get_field(record, requested)
    for field in candidates:
        value = get_field(record, field)
        if value is not None:
            return value
    return None


def extract_answer(value: Any) -> str:
    text = "" if value is None else str(value)
    match = ANSWER_PATTERN.search(text)
    if match:
        text = match.group(1)
    else:
        text = THINK_PATTERN.sub("", text)
    text = RECOVERING_PATTERN.sub("", text.strip())
    if "[stop]" in text.lower() or text.strip().lower() == "stop":
        return STOP
    return " ".join(text.split()).strip()


def normalized_text(text: str) -> str:
    if text == STOP:
        return STOP
    return " ".join(NON_WORD_PATTERN.sub(" ", text.lower()).split())


def load_segment_index(path: Path) -> Dict[str, Dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    records: Iterable[Dict[str, Any]]
    if isinstance(payload, dict):
        records = payload.values()
    elif isinstance(payload, list):
        records = payload
    else:
        raise ValueError("Segmentation JSON must be a list or dict")
    index = {}
    for episode in records:
        key = str(episode.get("episode_id"))
        if key in index:
            raise ValueError(f"Duplicate episode_id in segmentation: {key}")
        segments = episode.get("instruction_segments")
        if not isinstance(segments, list) or not segments:
            raise ValueError(f"episode {key} has no instruction_segments")
        index[key] = episode
    return index


def derive_gold_answer(episode: Dict[str, Any], frame: int) -> str:
    segments = episode["instruction_segments"]
    for index, segment in enumerate(segments):
        start, end = int(segment["start_frame"]), int(segment["end_frame"])
        if start <= frame <= end:
            if frame < end:
                return str(segment["sub_instruction"])
            if index + 1 < len(segments):
                return str(segments[index + 1]["sub_instruction"])
            return STOP
    raise ValueError(f"frame {frame} is outside all instruction segments")


def final_two_frame_instruction(episode: Dict[str, Any], frame: int) -> Optional[str]:
    """Return the final sub-instruction when ``frame`` is one of the last two frames."""
    valid = [
        segment for segment in episode["instruction_segments"]
        if int(segment["start_frame"]) <= int(segment["end_frame"])
    ]
    if not valid:
        return None
    final_segment = valid[-1]
    final_frame = int(final_segment["end_frame"])
    if final_frame - 1 <= frame <= final_frame:
        return str(final_segment["sub_instruction"])
    return None


def lexical_relation(gold: str, prediction: str) -> str:
    gold_normalized, prediction_normalized = normalized_text(gold), normalized_text(prediction)
    if gold_normalized == prediction_normalized:
        return "exact"
    if not gold_normalized or not prediction_normalized:
        return "empty"
    if gold_normalized == STOP or prediction_normalized == STOP:
        return "stop_mismatch"
    if gold_normalized in prediction_normalized:
        return "prediction_contains_gold"
    if prediction_normalized in gold_normalized:
        return "gold_contains_prediction"
    return "overlap_or_rephrase"


def percent(values: Sequence[bool]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def mean(values: Sequence[float]) -> Optional[float]:
    return float(sum(values) / len(values)) if values else None


def load_encoder(model_path: str, device: str):
    from sentence_transformers import SentenceTransformer

    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    return SentenceTransformer(model_path, device=device), device


def atomic_dump(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    temporary.replace(path)


def write_jsonl(records: Iterable[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def m1_frame_indices(current_frame: int) -> Tuple[List[int], int, int]:
    """Match the default M1 visual sampling in ``agent_dual_qwen2_5_lm.py``.

    The online agent keeps the latest three observations as the current view,
    leaves the two observations immediately before them out of the prompt, and
    selects at most 13 older history observations with denser coverage near the
    end of the trajectory.  Frame numbers here are one-based file names, while
    the agent internally uses zero-based list indices.
    """
    all_frames = list(range(1, current_frame + 1))
    current = all_frames[-3:]
    # Same as: history_end = len(indices) - m1_current_frames - 2.
    # The explicit two-frame gap is part of the agent's normal (non-recursive)
    # M1 path, so the static benchmark must preserve it as well.
    history_end = max(0, len(all_frames) - len(current) - 2)
    history_pool = all_frames[:history_end]
    if len(history_pool) > 13:
        positions = np.linspace(0, 1, 13)
        transformed_positions = 1 - (1 - positions)**1.5
        picked = np.round(transformed_positions * (len(history_pool) - 1)).astype(int).tolist()
        picked = sorted(set(picked))
        while len(picked) < 13:
            available = sorted(set(range(len(history_pool))) - set(picked))
            if not available:
                break
            picked.append(available[-1])
            picked.sort()
        history = [history_pool[index] for index in picked]
    else:
        history = history_pool
    return history + current, len(history), len(current)


def m1_prompt(global_instruction: str, num_history: int, num_current: int) -> str:
    total = num_history + num_current
    image_description = f"Above is 1 image." if total == 1 else f"Above are {total} images."
    if num_history:
        history_noun, history_verb = ("image", "is") if num_history == 1 else ("images", "are")
        current_noun, current_verb = ("image", "is") if num_current == 1 else ("images", "are")
        frame_description = (
            f"{image_description} The first {num_history} {history_noun} {history_verb} the History trajectory, "
            f"and the last {num_current} {current_noun} {current_verb} the Current view."
        )
    elif total == 1:
        frame_description = "Above is 1 image. It is the Current view."
    else:
        frame_description = f"{image_description} All of them are the Current view."
    return (
        f"{'<image>' * total}\n{frame_description}\n"
        f"Global Instruction: {global_instruction}\n"
        "Task: Analyze the history and current view to determine the current progress within the global instruction. "
        "Provide a structured report with the following format: <think> Current Instruction: <instruction> | "
        "Status: <Executing/Completed> | Next Instruction: <instruction> or None </think>\n"
        "<answer> Next Instruction to Execute </answer>"
    )


def global_uniform_frame_indices(current_frame: int, max_images: int = 8) -> List[int]:
    """Match the global-history frame sampling used by the single-model path.

    Frame files are numbered from one, while the training builder applies
    Python ``round`` to zero-based positions.  Keeping that rounding behavior
    (including bankers' rounding) is necessary for exact prompt alignment.
    """
    count = min(int(current_frame), int(max_images))
    if count <= 0:
        return []
    if count == 1:
        return [1]
    return [round(index * (current_frame - 1) / (count - 1)) + 1 for index in range(count)]


def single_m1_prompt(global_instruction: str, num_images: int) -> str:
    """Return the global-history M1 prompt used by the single-model path."""
    if not 1 <= num_images <= 8:
        raise ValueError(f"single_m1 expects 1-8 images, got {num_images}")
    image_description = (
        "Above is 1 image. It is the Observation history."
        if num_images == 1 else
        f"Above are {num_images} images. They form the Observation history and are ordered from earlier "
        "to more recent views."
    )
    return (
        f"{'<image>' * num_images}\n{image_description}\n"
        f"Global Instruction: {global_instruction}\n"
        "Task: Analyze the Observation history to determine the current progress within the global instruction. "
        "Provide a structured report with the following format: <think> Current Instruction: <instruction> | "
        "Status: <Executing/Completed> | Next Instruction: <instruction> or None </think>\n"
        "<answer> Next Instruction to Execute </answer>"
    )


def resolve_image_dir(episode: Dict[str, Any], image_root: Path) -> Path:
    video = episode.get("video")
    if isinstance(video, str) and video:
        video_dir = image_root / video / "rgb"
        if video_dir.is_dir():
            return video_dir
    data_path = str(episode.get("data_path", ""))
    if data_path:
        data_dir = image_root / data_path / "rgb"
        if data_dir.is_dir():
            return data_dir
    if isinstance(video, str) and video:
        # Preserve the most informative missing-path error if neither layout
        # exists. RxR deviation data stores a video path prefixed by the image-root name,
        # whereas data_path is relative to that root.
        return image_root / video / "rgb"
    if not data_path:
        raise ValueError(f"episode {episode.get('episode_id')} has neither video nor data_path")
    return image_root / data_path / "rgb"


def selected_frames(episode: Dict[str, Any], policy: str) -> List[int]:
    # Some generated segmentations contain an empty leading segment such as
    # start_frame=1, end_frame=0. It has no corresponding image/frame and
    # must not become an evaluation sample.
    segments = [
        segment for segment in episode["instruction_segments"]
        if int(segment["start_frame"]) <= int(segment["end_frame"])
    ]
    if policy == "segment_end":
        return [int(segment["end_frame"]) for segment in segments]
    if policy == "segment_start":
        return [int(segment["start_frame"]) for segment in segments]
    if policy == "both":
        return sorted({int(segment[key]) for segment in segments for key in ("start_frame", "end_frame")})
    return list(range(1, int(episode["num_frames"]) + 1))


def valid_segments(episode: Dict[str, Any]) -> List[Tuple[int, int, int]]:
    """Return (segment_index, start_frame, end_frame), omitting empty segments."""
    return [
        (index, int(segment["start_frame"]), int(segment["end_frame"]))
        for index, segment in enumerate(episode["instruction_segments"])
        if int(segment["start_frame"]) <= int(segment["end_frame"])
    ]


def frame_event_map(episode: Dict[str, Any]) -> Dict[int, str]:
    return {
        int(coordinate["frame"]): str(coordinate.get("event", ""))
        for coordinate in episode.get("continuous_gt", {}).get("coordinates", [])
        if coordinate.get("frame") is not None
    }


def sampling_weight(event: str, focus: str) -> int:
    if focus not in ("perturbation", "deviated_only"):
        return 1
    if event.startswith("perturb_"):
        return 6
    if event.startswith("reconnect"):
        return 3
    return 1


def is_deviated_event(event: str) -> bool:
    return event.startswith("perturb_") or event.startswith("reconnect")


def event_category(event: str) -> str:
    if event.startswith("perturb_"):
        return "perturbation"
    if event.startswith("reconnect"):
        return "reconnect"
    return "other"


def route_step_index(episode: Dict[str, Any], frame: int) -> Optional[int]:
    """Return the route-local next-step index used by the M1 handoff task."""
    segments = valid_segments(episode)
    for position, (_, start, end) in enumerate(segments):
        if start <= frame <= end:
            if frame < end:
                return position
            return position + 1 if position + 1 < len(segments) else None
    return None


def boundary_acceptable_indices(episode: Dict[str, Any], frame: int, window: int, gt_index: int) -> List[int]:
    """Accept adjacent step IDs only within ``window`` frames of their boundary."""
    segments = valid_segments(episode)
    accepted = {gt_index}
    for position, (_, _, end) in enumerate(segments[:-1]):
        if abs(frame - end) <= window:
            accepted.update((position, position + 1))
    return sorted(accepted)


def add_step_retrieval_metrics(
    score_rows: List[Dict[str, Any]],
    segmentation: Dict[str, Dict[str, Any]],
    encoder: Any,
    batch_size: int,
    boundary_window: int,
    top2_max_similarity_gap: float,
) -> Optional[Dict[str, Any]]:
    """Add route-local retrieval results with a similarity-aware Top-2 definition."""
    grouped: Dict[str, List[Tuple[int, List[str], int, List[int]]]] = defaultdict(list)
    candidate_texts = set()
    prediction_items = []
    for row_index, row in enumerate(score_rows):
        if row["terminal_two_frame"] or row["episode_id"] is None:
            continue
        episode_id = str(row["episode_id"])
        episode = segmentation.get(episode_id)
        if episode is None:
            continue
        segments = valid_segments(episode)
        gt_index = route_step_index(episode, int(row["frame"]))
        if gt_index is None:
            continue
        candidates = [str(episode["instruction_segments"][segment_index]["sub_instruction"])
                      for segment_index, _, _ in segments]
        accepted = boundary_acceptable_indices(episode, int(row["frame"]), boundary_window, gt_index)
        row["gt_step_index"] = gt_index
        row["candidate_step_count"] = len(candidates)
        row["boundary_acceptable_step_indices"] = accepted
        grouped[episode_id].append((row_index, candidates, gt_index, accepted))
        candidate_texts.update(candidates)
        if row["prediction_answer"] != STOP:
            prediction_items.append((row_index, row["prediction_answer"]))
    if not grouped:
        return None

    ordered_candidates = sorted(candidate_texts)
    candidate_embeddings = encoder.encode(
        ordered_candidates, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
    candidate_embedding_by_text = dict(zip(ordered_candidates, candidate_embeddings))
    prediction_embedding_by_row = {}
    if prediction_items:
        prediction_embeddings = encoder.encode(
            [text for _, text in prediction_items], batch_size=batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=True)
        prediction_embedding_by_row = dict(zip((row_index for row_index, _ in prediction_items), prediction_embeddings))

    retrieval_rows = []
    for episode_rows in grouped.values():
        candidates = episode_rows[0][1]
        candidate_matrix = np.stack([candidate_embedding_by_text[text] for text in candidates])
        for row_index, _, gt_index, accepted in episode_rows:
            row = score_rows[row_index]
            prediction_embedding = prediction_embedding_by_row.get(row_index)
            if prediction_embedding is None:
                ranked_indices, ranked_similarities = [], []
            else:
                similarities = candidate_matrix @ prediction_embedding
                ranked_indices = np.argsort(-similarities, kind="stable").astype(int).tolist()
                ranked_similarities = [float(similarities[index]) for index in ranked_indices]
            top_similarity = ranked_similarities[0] if ranked_similarities else None
            second_similarity = ranked_similarities[1] if len(ranked_similarities) > 1 else None
            similarity_gap = (
                top_similarity - second_similarity
                if top_similarity is not None and second_similarity is not None else None
            )
            top_indices = ranked_indices[:1]
            second_included = (
                len(ranked_indices) > 1
                and similarity_gap is not None
                and similarity_gap <= top2_max_similarity_gap
            )
            if second_included:
                top_indices.append(ranked_indices[1])
            predicted_index = top_indices[0] if top_indices else None
            row["predicted_step_index"] = predicted_index
            row["ranked_top2_step_indices"] = ranked_indices[:2]
            row["top2_step_indices"] = top_indices
            row["best_candidate_similarity"] = top_similarity
            row["top2_second_candidate_similarity"] = second_similarity
            row["top2_similarity_gap"] = similarity_gap
            row["top2_second_included"] = second_included
            row["top2_max_similarity_gap"] = top2_max_similarity_gap
            row["strict_step_correct"] = predicted_index == gt_index
            row["top2_step_correct"] = gt_index in top_indices
            row["boundary_aware_step_correct"] = predicted_index in accepted if predicted_index is not None else False
            retrieval_rows.append(row)

    return {
        "count": len(retrieval_rows),
        "candidate_scope": "all valid sub-instructions in the same route",
        "top2_definition": (
            "rank 1 always qualifies; rank 2 qualifies only when its cosine similarity "
            "is no more than top2_max_similarity_gap below rank 1"
        ),
        "top2_max_similarity_gap": top2_max_similarity_gap,
        "top2_second_inclusion_rate": percent([row["top2_second_included"] for row in retrieval_rows]),
        "boundary_window_frames": boundary_window,
        "strict_step_accuracy": percent([row["strict_step_correct"] for row in retrieval_rows]),
        "top2_step_recall": percent([row["top2_step_correct"] for row in retrieval_rows]),
        "boundary_aware_accuracy": percent([row["boundary_aware_step_correct"] for row in retrieval_rows]),
        "mean_best_candidate_similarity": mean([
            row["best_candidate_similarity"] for row in retrieval_rows
            if row["best_candidate_similarity"] is not None
        ]),
        "terminal_two_frame_excluded": True,
    }


def current_segment_index(episode: Dict[str, Any], frame: int) -> Optional[int]:
    """Return the active segment index before applying the next-step handoff rule."""
    for position, (_, start, end) in enumerate(valid_segments(episode)):
        if start <= frame <= end:
            return position
    return None


def add_progress_transition_metrics(
    score_rows: List[Dict[str, Any]],
    segmentation: Dict[str, Dict[str, Any]],
    encoder: Any,
    batch_size: int,
    boundary_window: int,
) -> Optional[Dict[str, Any]]:
    """Measure local current-step retention and transitions around segment boundaries."""
    work_items = []
    candidate_texts = set()
    prediction_items = []
    for row_index, row in enumerate(score_rows):
        if row["terminal_two_frame"] or row["episode_id"] is None:
            continue
        episode = segmentation.get(str(row["episode_id"]))
        if episode is None:
            continue
        active_index = current_segment_index(episode, int(row["frame"]))
        if active_index is None:
            continue
        segments = valid_segments(episode)
        local_indices = list(range(max(0, active_index - 1), min(len(segments), active_index + 2)))
        local_texts = [str(episode["instruction_segments"][segments[index][0]]["sub_instruction"])
                       for index in local_indices]
        row["active_step_index"] = active_index
        row["local_candidate_step_indices"] = local_indices
        work_items.append((row_index, episode, segments, active_index, local_indices, local_texts))
        candidate_texts.update(local_texts)
        if row["prediction_answer"] != STOP:
            prediction_items.append((row_index, row["prediction_answer"]))
    if not work_items:
        return None

    ordered_candidates = sorted(candidate_texts)
    candidate_embeddings = encoder.encode(
        ordered_candidates, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
    candidate_embedding_by_text = dict(zip(ordered_candidates, candidate_embeddings))
    prediction_embedding_by_row = {}
    if prediction_items:
        prediction_embeddings = encoder.encode(
            [text for _, text in prediction_items], batch_size=batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=True)
        prediction_embedding_by_row = dict(zip((row_index for row_index, _ in prediction_items), prediction_embeddings))

    interior_rows, before_events, after_events = [], [], []
    for row_index, episode, segments, active_index, local_indices, local_texts in work_items:
        row = score_rows[row_index]
        prediction_embedding = prediction_embedding_by_row.get(row_index)
        if prediction_embedding is None:
            predicted_index = None
        else:
            local_matrix = np.stack([candidate_embedding_by_text[text] for text in local_texts])
            predicted_index = local_indices[int(np.argmax(local_matrix @ prediction_embedding))]
        row["local_predicted_step_index"] = predicted_index
        boundaries = [(index, end) for index, (_, _, end) in enumerate(segments[:-1])]
        is_interior = all(abs(int(row["frame"]) - end) > boundary_window for _, end in boundaries)
        row["segment_interior"] = is_interior
        row["current_step_retained"] = predicted_index == active_index if is_interior else None
        if is_interior:
            interior_rows.append(row)
        for boundary_index, boundary_frame in boundaries:
            offset = int(row["frame"]) - boundary_frame
            if -boundary_window <= offset <= -1:
                before_events.append(predicted_index == boundary_index + 1)
            elif 0 <= offset <= boundary_window:
                after_events.append(predicted_index == boundary_index)

    return {
        "candidate_scope": "local [previous, current, next] valid route steps",
        "trajectory_condition": "clean GT trajectory only; perturbation/recovery labels are unavailable",
        "boundary_window_frames": boundary_window,
        "segment_interior": {
            "count": len(interior_rows),
            "current_step_retention": percent([row["current_step_retained"] for row in interior_rows]),
        },
        "boundary": {
            "before_completion_count": len(before_events),
            "premature_transition_rate": percent(before_events),
            "after_completion_count": len(after_events),
            "delayed_transition_rate": percent(after_events),
        },
        "terminal_two_frame_excluded": True,
    }


def load_or_create_sampled_frames(
    segmentation: Dict[str, Dict[str, Any]], args: argparse.Namespace
) -> List[Tuple[str, int, int]]:
    """Persist a fixed sample, with per-segment coverage except in deviated-only mode."""
    if not args.sampled_frames:
        raise ValueError("--frame-policy random_per_segment requires --sampled-frames")
    path = args.sampled_frames
    expected = {
        (episode_id, segment_index): (start, end)
        for episode_id, episode in segmentation.items()
        for segment_index, start, end in valid_segments(episode)
    }
    if path.is_file():
        with path.open(encoding="utf-8") as f:
            payload = json.load(f)
        selections = payload.get("segments") if isinstance(payload, dict) else None
        if not isinstance(selections, list):
            raise ValueError(f"{path} must contain a JSON object with a 'segments' list")
        stored_focus = payload.get("sampling_focus", "uniform_length")
        if stored_focus != args.sample_focus:
            raise ValueError(
                f"{path} uses sampling_focus={stored_focus!r}, expected {args.sample_focus!r}; "
                "choose a new path to resample"
            )
        selected, seen = [], set()
        for selection in selections:
            if not isinstance(selection, dict):
                raise ValueError(f"{path} contains a non-object segment selection")
            key = (str(selection.get("episode_id")), int(selection.get("segment_index")))
            if key not in expected or key in seen:
                raise ValueError(f"{path} contains an unknown or duplicate segment selection: {key}")
            frames = selection.get("frames")
            start, end = expected[key]
            if not isinstance(frames, list) or not frames or len(set(frames)) != len(frames):
                raise ValueError(f"{path} has invalid frames for segment {key}")
            if any(not isinstance(frame, int) or frame < start or frame > end for frame in frames):
                raise ValueError(f"{path} selects an out-of-range frame for segment {key}")
            if stored_focus == "deviated_only":
                events = frame_event_map(segmentation[key[0]])
                if any(not is_deviated_event(events.get(frame, "")) for frame in frames):
                    raise ValueError(f"{path} contains a non-deviated frame for segment {key}")
            selected.extend((key[0], key[1], frame) for frame in frames)
            seen.add(key)
        missing, extra = set(expected) - seen, seen - set(expected)
        if (stored_focus != "deviated_only" and missing) or extra:
            raise ValueError(
                f"{path} does not match the current segmentation "
                f"(missing={len(missing)}, extra={len(extra)}); choose a new path to resample"
            )
        print(f"reusing {len(selected)} fixed sampled frames from {path}", flush=True)
        return selected

    rng = random.Random(args.sampling_seed)
    candidates = [
        (episode_id, segment_index, start, end)
        for episode_id in sorted(segmentation, key=int)
        for segment_index, start, end in valid_segments(segmentation[episode_id])
    ]
    if args.sample_focus != "deviated_only" and args.sample_target < len(candidates):
        raise ValueError(
            f"--sample-target ({args.sample_target}) is smaller than the {len(candidates)} valid segments"
        )
    event_maps = {episode_id: frame_event_map(episode) for episode_id, episode in segmentation.items()}
    if args.sample_focus == "deviated_only":
        pool = [
            (episode_id, segment_index, frame)
            for episode_id, segment_index, start, end in candidates
            for frame in range(start, end + 1)
            if is_deviated_event(event_maps[episode_id].get(frame, ""))
        ]
        if not pool:
            raise ValueError("No perturbation/reconnect frames are available for deviated_only sampling")
        target = min(args.sample_target, len(pool))
        weighted_pool = [
            (rng.random()**(1.0 / sampling_weight(event_maps[episode_id].get(frame, ""), args.sample_focus)),
             episode_id, segment_index, frame)
            for episode_id, segment_index, frame in pool
        ]
        selected_by_segment = defaultdict(list)
        for _, episode_id, segment_index, frame in sorted(weighted_pool, reverse=True)[:target]:
            selected_by_segment[(episode_id, segment_index)].append(frame)
        selections, selected = [], []
        for episode_id, segment_index in sorted(selected_by_segment, key=lambda key: (int(key[0]), key[1])):
            frames = sorted(selected_by_segment[(episode_id, segment_index)])
            selections.append({"episode_id": int(episode_id), "segment_index": segment_index, "frames": frames})
            selected.extend((episode_id, segment_index, frame) for frame in frames)
    else:
        total_capacity = sum(end - start + 1 for _, _, start, end in candidates)
        target = min(args.sample_target, total_capacity)
        if args.sample_focus == "perturbation":
            # First guarantee one frame per segment, preferring perturbation frames
            # and then reconnect frames. The remaining budget is sampled globally
            # without replacement using the same priority weights.
            selected_by_segment = {}
            selected_keys = set()
            for episode_id, segment_index, start, end in candidates:
                frames = list(range(start, end + 1))
                weights = [sampling_weight(event_maps[episode_id].get(frame, ""), args.sample_focus) for frame in frames]
                frame = rng.choices(frames, weights=weights, k=1)[0]
                selected_by_segment[(episode_id, segment_index)] = [frame]
                selected_keys.add((episode_id, segment_index, frame))
            remaining = target - len(candidates)
            priority_pool = []
            for episode_id, segment_index, start, end in candidates:
                for frame in range(start, end + 1):
                    key = (episode_id, segment_index, frame)
                    if key in selected_keys:
                        continue
                    weight = sampling_weight(event_maps[episode_id].get(frame, ""), args.sample_focus)
                    priority_pool.append((rng.random()**(1.0 / weight), key))
            for _, (episode_id, segment_index, frame) in sorted(priority_pool, reverse=True)[:remaining]:
                selected_by_segment[(episode_id, segment_index)].append(frame)
            selections, selected = [], []
            for episode_id, segment_index, _, _ in candidates:
                frames = sorted(selected_by_segment[(episode_id, segment_index)])
                selections.append({"episode_id": int(episode_id), "segment_index": segment_index, "frames": frames})
                selected.extend((episode_id, segment_index, frame) for frame in frames)
        else:
            extra_target = target - len(candidates)
            extra_capacity = total_capacity - len(candidates)
            allocations = [1] * len(candidates)
            fractional = []
            for index, (_, _, start, end) in enumerate(candidates):
                capacity = end - start
                exact_extra = extra_target * capacity / extra_capacity if extra_capacity else 0.0
                allocated_extra = int(exact_extra)
                allocations[index] += allocated_extra
                fractional.append(exact_extra - allocated_extra)
            remainder = extra_target - sum(allocation - 1 for allocation in allocations)
            ranked = sorted(
                ((rng.random()**(1.0 / weight), index) for index, weight in enumerate(fractional) if weight > 0),
                reverse=True,
            )
            for _, index in ranked[:remainder]:
                allocations[index] += 1
            selections, selected = [], []
            for (episode_id, segment_index, start, end), count in zip(candidates, allocations):
                frames = sorted(rng.sample(range(start, end + 1), count))
                selections.append({"episode_id": int(episode_id), "segment_index": segment_index, "frames": frames})
                selected.extend((episode_id, segment_index, frame) for frame in frames)
    event_counts = Counter(
        event_category(frame_event_map(segmentation[episode_id]).get(frame, ""))
        for episode_id, _, frame in selected
    )
    atomic_dump({
        "format": "m1_static_qa_random_per_segment_v1",
        "segmentation": str(args.segmentation),
        "sampling_seed": args.sampling_seed,
        "sampling_focus": args.sample_focus,
        "sample_target": args.sample_target,
        "num_samples": len(selected),
        "event_category_counts": dict(event_counts),
        "segments": selections,
    }, path)
    print(f"created {len(selected)} fixed sampled frames at {path} (seed={args.sampling_seed})", flush=True)
    return selected


def image_data_url(path: Path) -> str:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((640, 480))
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=85)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def vllm_predict_sample(sample: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    if args.task_type == "single_m1":
        frame_indices = global_uniform_frame_indices(sample["frame"])
        num_history, num_current = len(frame_indices), 0
        prompt = single_m1_prompt(sample["global_instruction"], len(frame_indices))
    else:
        frame_indices, num_history, num_current = m1_frame_indices(sample["frame"])
        prompt = m1_prompt(sample["global_instruction"], num_history, num_current)
    image_paths = [sample["image_dir"] / f"{frame:03d}.jpg" for frame in frame_indices]
    missing = [str(path) for path in image_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} evaluation images, e.g. {missing[0]}")
    content = [{"type": "image_url", "image_url": {"url": image_data_url(path)}} for path in image_paths]
    content.append({"type": "text", "text": prompt.replace("<image>", "").strip()})
    payload = {
        "model": args.vllm_model,
        "messages": [
            {"role": "system", "content": "You are an intelligent navigation robot."},
            {"role": "user", "content": content},
        ],
        "max_tokens": args.vllm_max_tokens,
        "temperature": args.vllm_temperature,
    }
    url = args.vllm_url.rstrip("/") + "/v1/chat/completions"
    response = requests.post(url, json=payload, timeout=args.vllm_timeout_s)
    if response.status_code >= 400:
        raise RuntimeError(f"vLLM HTTP {response.status_code}: {response.text[:1000]}")
    raw_response = response.json()["choices"][0]["message"]["content"].strip()
    record = {
        "id": sample["id"],
        "episode_id": sample["episode_id"],
        "frame": sample["frame"],
        "response": raw_response,
        "image_frames": frame_indices,
        "num_history": num_history,
        "num_current": num_current,
        "task_type": args.task_type,
    }
    record["gold_answer"] = sample["gold_answer"]
    return record


def generate_vllm_predictions(
    segmentation: Dict[str, Dict[str, Any]], args: argparse.Namespace, checkpoint_path: Optional[Path] = None
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    samples = []
    random_samples = load_or_create_sampled_frames(segmentation, args) if args.frame_policy == "random_per_segment" else []
    if random_samples:
        for episode_id, segment_index, frame in random_samples:
            episode = segmentation[episode_id]
            sample = {
                "id": f"{episode_id}:{segment_index}:{frame}",
                "episode_id": int(episode_id),
                "frame": frame,
                "global_instruction": episode.get("original_instruction") or episode.get("instruction", ""),
                "image_dir": resolve_image_dir(episode, args.image_root),
            }
            sample["gold_answer"] = derive_gold_answer(episode, frame)
            samples.append(sample)
    else:
        for episode_id, episode in segmentation.items():
            image_dir = resolve_image_dir(episode, args.image_root)
            for frame in selected_frames(episode, args.frame_policy):
                sample = {
                    "id": f"{episode_id}:{frame}",
                    "episode_id": int(episode_id),
                    "frame": frame,
                    "global_instruction": episode.get("original_instruction") or episode.get("instruction", ""),
                    "image_dir": image_dir,
                }
                sample["gold_answer"] = derive_gold_answer(episode, frame)
                samples.append(sample)
    if args.max_samples:
        samples = samples[:args.max_samples]
    if not samples:
        raise ValueError("No static samples selected")
    print(f"vLLM static samples={len(samples)} policy={args.frame_policy}", flush=True)
    records, errors = [], []
    if checkpoint_path:
        write_jsonl(records, checkpoint_path)
        print(f"initialized prediction checkpoint: {checkpoint_path}", flush=True)
    with ThreadPoolExecutor(max_workers=args.vllm_workers) as executor:
        futures = {executor.submit(vllm_predict_sample, sample, args): sample for sample in samples}
        for completed, future in enumerate(as_completed(futures), start=1):
            sample = futures[future]
            try:
                records.append(future.result())
            except Exception as exc:
                errors.append({"id": sample["id"], "error": str(exc)})
            if checkpoint_path and completed % args.save_every == 0:
                records.sort(key=lambda record: (record["episode_id"], record["frame"]))
                write_jsonl(records, checkpoint_path)
                print(
                    f"checkpointed {completed}/{len(samples)} samples "
                    f"({len(records)} predictions, {len(errors)} errors) to {checkpoint_path}",
                    flush=True,
                )
    records.sort(key=lambda record: (record["episode_id"], record["frame"]))
    if checkpoint_path:
        write_jsonl(records, checkpoint_path)
        print(f"final prediction checkpoint: {len(records)} records at {checkpoint_path}", flush=True)
    return records, errors


def main() -> None:
    args = parse_args()
    args.vllm_url = args.vllm_url.strip()
    if not args.segmentation and not args.gold_field:
        raise ValueError("Provide either --segmentation or --gold-field")
    if not args.prepare_samples_only and not args.predictions and not args.vllm_url:
        raise ValueError("Provide --predictions for offline scoring or --vllm-url for direct evaluation")
    if args.prepare_samples_only:
        if not args.segmentation:
            raise ValueError("--prepare-samples-only requires --segmentation")
        if args.frame_policy != "random_per_segment":
            raise ValueError("--prepare-samples-only requires --frame-policy random_per_segment")
        if not args.sampled_frames:
            raise ValueError("--prepare-samples-only requires --sampled-frames")
    if args.vllm_url:
        if not args.segmentation or not args.image_root:
            raise ValueError("Direct vLLM evaluation requires --segmentation and --image-root")
        if not args.vllm_model:
            raise ValueError("Direct vLLM evaluation requires --vllm-model")
        if args.vllm_workers < 1:
            raise ValueError("--vllm-workers must be positive")
        if args.save_every < 1:
            raise ValueError("--save-every must be positive")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.boundary_window < 0:
        raise ValueError("--boundary-window must be non-negative")
    if args.top2_max_similarity_gap < 0:
        raise ValueError("--top2-max-similarity-gap must be non-negative")
    thresholds = [float(item) for item in args.thresholds.split(",") if item.strip()]
    if any(item < -1.0 or item > 1.0 for item in thresholds):
        raise ValueError("thresholds must be within [-1, 1]")

    segmentation = load_segment_index(args.segmentation) if args.segmentation else {}
    frame_events = {
        episode_id: frame_event_map(episode)
        for episode_id, episode in segmentation.items()
    }
    if args.prepare_samples_only:
        selected = load_or_create_sampled_frames(segmentation, args)
        print(f"prepared {len(selected)} fixed sampled frames: {args.sampled_frames}")
        return
    if args.output_prefix:
        prefix = args.output_prefix
    elif args.predictions:
        prefix = args.predictions.with_suffix("")
    elif args.segmentation:
        prefix = args.segmentation.with_suffix("").with_name(args.segmentation.stem + "_m1_static_eval")
    else:  # Kept for static type checkers; validation above makes this unreachable.
        raise AssertionError("missing output prefix")

    generation_errors = []
    if args.vllm_url:
        prediction_path = Path(f"{prefix}.predictions.jsonl")
        predictions, generation_errors = generate_vllm_predictions(segmentation, args, prediction_path)
    else:
        predictions = load_records(args.predictions)
        prediction_path = args.predictions

    score_rows, errors = [], []
    for row_index, record in enumerate(predictions):
        raw_prediction = first_field(
            record,
            args.prediction_field,
            ("response", "raw_response", "prediction", "output", "answer", "assistant_response"),
        )
        prediction = extract_answer(raw_prediction)
        try:
            if args.vllm_url:
                gold = extract_answer(record.get("gold_answer"))
                episode_id = record.get("episode_id")
                frame = record.get("frame")
            elif args.gold_field:
                gold = extract_answer(get_field(record, args.gold_field))
                episode_id, frame = None, None
            else:
                episode_id = first_field(record, args.episode_id_field, ("episode_id", "metadata.episode_id"))
                frame = first_field(record, args.frame_field, ("frame", "frame_id", "metadata.frame", "metadata.current_frame"))
                if episode_id is None or frame is None:
                    raise ValueError("missing episode_id or frame")
                gold = derive_gold_answer(segmentation[str(int(episode_id))], int(frame))
            if not prediction:
                raise ValueError("empty prediction answer")
            if not gold:
                raise ValueError("empty gold answer")
        except (KeyError, TypeError, ValueError) as exc:
            errors.append({"row_index": row_index, "error": str(exc), "record": record})
            continue
        terminal_reference = None
        if episode_id is not None and frame is not None and segmentation:
            terminal_reference = final_two_frame_instruction(
                segmentation[str(int(episode_id))], int(frame)
            )
        scoring_reference = terminal_reference or gold
        terminal_stop_accepted = terminal_reference is not None and prediction == STOP
        trajectory_event = None
        if episode_id is not None and frame is not None:
            trajectory_event = frame_events.get(str(int(episode_id)), {}).get(int(frame), "")
        score_rows.append({
            "row_index": row_index,
            "id": record.get("id"),
            "episode_id": int(episode_id) if episode_id is not None else None,
            "frame": int(frame) if frame is not None else None,
            "gold_answer": gold,
            "scoring_reference_answer": scoring_reference,
            "terminal_two_frame": terminal_reference is not None,
            "terminal_stop_accepted": terminal_stop_accepted,
            "prediction_answer": prediction,
            "trajectory_event": trajectory_event,
            "trajectory_event_category": event_category(trajectory_event or "") if trajectory_event is not None else None,
            "relation": "terminal_stop" if terminal_stop_accepted else lexical_relation(scoring_reference, prediction),
        })

    all_errors = generation_errors + errors
    if args.strict and all_errors:
        raise RuntimeError(f"{len(all_errors)} unscorable rows; first error: {all_errors[0]['error']}")
    similarity_indices = [
        index for index, row in enumerate(score_rows)
        if row["scoring_reference_answer"] != STOP and row["prediction_answer"] != STOP
    ]
    if similarity_indices or segmentation:
        encoder, resolved_device = load_encoder(args.model, args.device)
    else:
        encoder, resolved_device = None, args.device
    if similarity_indices:
        golds = [score_rows[index]["scoring_reference_answer"] for index in similarity_indices]
        predictions_text = [score_rows[index]["prediction_answer"] for index in similarity_indices]
        gold_embeddings = encoder.encode(
            golds, batch_size=args.batch_size, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
        prediction_embeddings = encoder.encode(
            predictions_text, batch_size=args.batch_size, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
        similarities = np.sum(gold_embeddings * prediction_embeddings, axis=1)
        for index, similarity in zip(similarity_indices, similarities):
            score_rows[index]["cosine_similarity"] = float(similarity)
    for row in score_rows:
        if row["terminal_stop_accepted"]:
            row["cosine_similarity"] = 1.0
        elif row["scoring_reference_answer"] == STOP or row["prediction_answer"] == STOP:
            row["cosine_similarity"] = (
                1.0 if row["scoring_reference_answer"] == row["prediction_answer"] else 0.0
            )
        row["score"] = row["cosine_similarity"]

    scores = [row["score"] for row in score_rows]
    relations = Counter(row["relation"] for row in score_rows)
    relation_scores = defaultdict(list)
    event_rows = defaultdict(list)
    for row in score_rows:
        relation_scores[row["relation"]].append(row["score"])
        if row["trajectory_event_category"] is not None:
            event_rows[row["trajectory_event_category"]].append(row)
    terminal_rows = [row for row in score_rows if row["terminal_two_frame"]]
    normal_rows = [row for row in score_rows if not row["terminal_two_frame"]]
    step_retrieval = add_step_retrieval_metrics(
        score_rows, segmentation, encoder, args.batch_size, args.boundary_window,
        args.top2_max_similarity_gap,
    ) if segmentation else None
    progress_transition = add_progress_transition_metrics(
        score_rows, segmentation, encoder, args.batch_size, args.boundary_window
    ) if segmentation else None
    summary = {
        "predictions": str(prediction_path),
        "segmentation": str(args.segmentation) if args.segmentation else None,
        "vllm": {
            "url": args.vllm_url or None,
            "model": args.vllm_model or None,
            "frame_policy": args.frame_policy if args.vllm_url else None,
        },
        "model": args.model,
        "device": resolved_device,
        "num_input_rows": len(predictions),
        "num_scored": len(score_rows),
        "num_unscorable": len(all_errors),
        "mean_score": mean(scores),
        "median_score": float(np.median(scores)) if scores else None,
        "exact_match": percent([row["relation"] == "exact" for row in score_rows]),
        "threshold_pass_rate": {str(threshold): percent([score >= threshold for score in scores]) for threshold in thresholds},
        "terminal_two_frame": {
            "count": len(terminal_rows),
            "stop_prediction_rate": percent([row["terminal_stop_accepted"] for row in terminal_rows]),
            "mean_score": mean([row["score"] for row in terminal_rows]),
            "policy": "[STOP] scores 1.0; all other predictions are compared with the final sub-instruction",
        },
        "non_terminal": {
            "count": len(normal_rows),
            "mean_cosine": mean([row["score"] for row in normal_rows]),
        },
        "trajectory_event": {
            category: {
                "count": len(rows),
                "mean_score": mean([row["score"] for row in rows]),
                "exact_match": percent([row["relation"] == "exact" for row in rows]),
                "threshold_pass_rate": {
                    str(threshold): percent([row["score"] >= threshold for row in rows])
                    for threshold in thresholds
                },
            }
            for category, rows in sorted(event_rows.items())
        },
        "step_retrieval": step_retrieval,
        "progress_transition": progress_transition,
        "granularity": {
            "relation_counts": dict(relations),
            "relation_rates": {name: count / len(score_rows) if score_rows else 0.0 for name, count in relations.items()},
            "mean_score_by_relation": {name: mean(values) for name, values in relation_scores.items()},
            "meaning": {
                "prediction_contains_gold": "prediction is lexically more detailed than gold",
                "gold_contains_prediction": "prediction is lexically coarser than gold",
            },
        },
        "errors_preview": all_errors[:20],
    }
    summary_path = Path(f"{prefix}.summary.json")
    details_path = Path(f"{prefix}.details.jsonl")
    atomic_dump(summary, summary_path)
    write_jsonl(score_rows, details_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote: {summary_path}")
    print(f"wrote: {details_path}")


if __name__ == "__main__":
    main()
