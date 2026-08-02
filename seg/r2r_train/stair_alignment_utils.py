import re
from typing import Iterable, List, Optional


STAIR_STRONG_PEAK_THRESHOLD = 0.08
STAIR_STRONG_DELTA_THRESHOLD = 0.08
STAIR_STRONG_DELTA_MAX_LENGTH = 5
STAIR_VERTICAL_LABELS = {"up", "down"}
_STAIR_OBJECT_PATTERN = r"(?:stairs?|staircase|stairway|steps?|flights?(?:\s+of\s+(?:stairs?|steps?))?)"
_STAIR_COUNT_PATTERN = (
    r"(?:one|two|three|four|five|six|seven|eight|nine|ten|"
    r"first|second|third|fourth|fifth|\d+)"
)
_NONCORE_EDGE_LABELS = {"stair_context", "stair_endpoint"}
_CORE_STAIR_LABELS = {"stair_motion_up", "stair_motion_down", "stair_progress"}
_DIRECTION_GAP_BLOCKLIST = (
    "hall",
    "hallway",
    "corridor",
    "past",
    "beside",
    "near",
    "toward",
    "towards",
    "around",
    "parallel",
    "bottom",
    "top",
    "under",
)


def _has_approach_only_stair_motion(text: str, direction: str) -> bool:
    return bool(re.search(rf"\b{direction}\s+to\b[^\n\r]*\b{_STAIR_OBJECT_PATTERN}\b", text))


def _gap_mentions_landmark_context(gap: str) -> bool:
    return any(re.search(rf"\b{re.escape(word)}\b", gap) for word in _DIRECTION_GAP_BLOCKLIST)


def _has_directional_stair_motion(text: str, direction: str) -> bool:
    pattern = re.compile(
        rf"\b{direction}\b(?P<gap>(?:\s+\w+){{0,8}})\s+{_STAIR_OBJECT_PATTERN}\b"
    )
    for match in pattern.finditer(text):
        if _gap_mentions_landmark_context(match.group("gap")):
            continue
        return True
    return False


def _has_motion_verb_stair_object(text: str, verbs: str) -> bool:
    pattern = re.compile(
        rf"\b(?:{verbs})\b(?P<gap>(?:\s+\w+){{0,8}})\s+{_STAIR_OBJECT_PATTERN}\b"
    )
    for match in pattern.finditer(text):
        if _gap_mentions_landmark_context(match.group("gap")):
            continue
        return True
    return False


def is_up_stair_motion_text(text: str) -> bool:
    t = str(text).lower()
    if _has_approach_only_stair_motion(t, "up"):
        return False
    if re.search(r"\b(?:go|walk|head|move|continue|proceed|climb)\s+upstairs\b", t):
        return True
    if re.fullmatch(r"\s*(?:ascend|ascending)\.?\s*", t):
        return True
    if _has_motion_verb_stair_object(t, r"ascend|ascending|climb|climbing"):
        return True
    return _has_directional_stair_motion(t, "up")


def is_down_stair_motion_text(text: str) -> bool:
    t = str(text).lower()
    if _has_approach_only_stair_motion(t, "down"):
        return False
    if re.search(r"\b(?:go|walk|head|move|continue|proceed)\s+downstairs\b", t):
        return True
    if re.fullmatch(r"\s*(?:descend|descending)\.?\s*", t):
        return True
    if _has_motion_verb_stair_object(t, r"descend|descending"):
        return True
    return _has_directional_stair_motion(t, "down")


def infer_stair_directions(
    instruction_labels: Optional[List[str]],
    sub_instructions: List[str],
    focus_indices: Optional[Iterable[int]] = None,
) -> List[str]:
    if focus_indices is None:
        focus_indices = range(len(sub_instructions))

    focused_indices = [idx for idx in focus_indices if 0 <= idx < len(sub_instructions)]
    if not focused_indices:
        focused_indices = list(range(len(sub_instructions)))

    directions = []
    for idx in focused_indices:
        label = instruction_labels[idx] if instruction_labels and idx < len(instruction_labels) else ""
        text = str(sub_instructions[idx]).lower()
        if label == "stair_motion_up" or is_up_stair_motion_text(text):
            directions.append("up")
        if label == "stair_motion_down" or is_down_stair_motion_text(text):
            directions.append("down")
    return sorted(set(directions))


def trim_noncore_stair_instruction_indices(
    instruction_labels: Optional[List[str]],
    stair_indices: Iterable[int],
) -> List[int]:
    if not instruction_labels:
        return []

    normalized = sorted({
        int(idx) for idx in stair_indices
        if isinstance(idx, int) and 0 <= idx < len(instruction_labels)
    })
    if not normalized:
        return []

    start_pos = 0
    end_pos = len(normalized) - 1
    while start_pos <= end_pos and instruction_labels[normalized[start_pos]] in _NONCORE_EDGE_LABELS:
        start_pos += 1
    while end_pos >= start_pos and instruction_labels[normalized[end_pos]] in _NONCORE_EDGE_LABELS:
        end_pos -= 1
    if start_pos > end_pos:
        return []

    trimmed = normalized[start_pos:end_pos + 1]
    if not any(instruction_labels[idx] in _CORE_STAIR_LABELS for idx in trimmed):
        return []
    return trimmed


def cluster_candidate_spans(spans: List[dict], max_frame_gap: int = 4) -> List[List[dict]]:
    clusters = []
    current = []
    for span in spans:
        if not current:
            current = [span]
            continue

        prev = current[-1]
        frame_gap = int(span.get("start_frame", 0)) - int(prev.get("end_frame", 0))
        if frame_gap <= max_frame_gap:
            current.append(span)
            continue

        clusters.append(current)
        current = [span]

    if current:
        clusters.append(current)
    return clusters


def is_strong_stair_span(span: dict) -> bool:
    if not isinstance(span, dict):
        return False
    if span.get("label") not in STAIR_VERTICAL_LABELS:
        return False
    peak_abs = abs(float(span.get("peak_abs_diff", 0.0)))
    abs_delta = abs(float(span.get("delta", 0.0)))
    length = int(span.get("length", 1))
    if peak_abs >= STAIR_STRONG_PEAK_THRESHOLD:
        return True
    return abs_delta >= STAIR_STRONG_DELTA_THRESHOLD and length <= STAIR_STRONG_DELTA_MAX_LENGTH


def select_primary_stair_span_indices(height_summary: dict, preferred_labels: List[str]) -> List[int]:
    spans = height_summary.get("spans", []) if isinstance(height_summary, dict) else []
    if not spans:
        return []

    preferred = set(preferred_labels) & STAIR_VERTICAL_LABELS
    candidates = [
        span for span in spans
        if isinstance(span, dict) and span.get("label") in (preferred or STAIR_VERTICAL_LABELS)
    ]
    if not candidates:
        return []

    strong_candidates = [span for span in candidates if is_strong_stair_span(span)]
    clustered = cluster_candidate_spans(strong_candidates or candidates)
    if not clustered:
        return []

    def cluster_score(cluster: List[dict]):
        strong = [span for span in cluster if is_strong_stair_span(span)]
        strong_count = len(strong)
        strong_delta = sum(abs(float(span.get("delta", 0.0))) for span in strong)
        total_delta = sum(abs(float(span.get("delta", 0.0))) for span in cluster)
        peak_abs = max(abs(float(span.get("peak_abs_diff", 0.0))) for span in cluster)
        total_length = sum(int(span.get("length", 1)) for span in cluster)
        end_frame = max(int(span.get("end_frame", 0)) for span in cluster)
        return (strong_count, strong_delta, total_delta, peak_abs, total_length, end_frame)

    best_cluster = max(clustered, key=cluster_score)
    if strong_candidates:
        strong_positions = [idx for idx, span in enumerate(best_cluster) if is_strong_stair_span(span)]
        if strong_positions:
            best_cluster = best_cluster[strong_positions[0]:strong_positions[-1] + 1]

    return [int(span.get("span_index", idx)) for idx, span in enumerate(best_cluster)]


def build_candidate_stair_blocks(height_summary: dict, preferred_labels: List[str]) -> List[dict]:
    spans = height_summary.get("spans", []) if isinstance(height_summary, dict) else []
    if not spans:
        return []

    preferred = set(preferred_labels) & STAIR_VERTICAL_LABELS
    candidates = [
        span for span in spans
        if isinstance(span, dict) and span.get("label") in (preferred or STAIR_VERTICAL_LABELS)
    ]
    if not candidates:
        return []

    strong_candidates = [span for span in candidates if is_strong_stair_span(span)]
    blocks = cluster_candidate_spans(strong_candidates or candidates)

    result = []
    for block_index, block in enumerate(blocks):
        span_indices = [int(span.get("span_index", idx)) for idx, span in enumerate(block)]
        if not span_indices:
            continue
        labels = sorted({str(span.get("label", "")) for span in block if span.get("label")})
        result.append({
            "block_index": block_index,
            "span_indices": span_indices,
            "start_frame": min(int(span.get("start_frame", 0)) for span in block),
            "end_frame": max(int(span.get("end_frame", 0)) for span in block),
            "labels": labels,
            "total_abs_delta": round(sum(abs(float(span.get("delta", 0.0))) for span in block), 4),
            "max_peak_abs_diff": round(max(abs(float(span.get("peak_abs_diff", 0.0))) for span in block), 4),
        })
    return result


def score_stair_block(block: dict) -> tuple:
    if not isinstance(block, dict):
        return (0.0, 0.0, 0, 0)
    total_abs_delta = abs(float(block.get("total_abs_delta", 0.0)))
    peak_abs = abs(float(block.get("max_peak_abs_diff", 0.0)))
    span_count = len(block.get("span_indices", []) or [])
    end_frame = int(block.get("end_frame", 0))
    return (total_abs_delta, peak_abs, span_count, end_frame)


def choose_primary_block_indices(candidate_blocks: List[dict], selected_block_indices: List[int]) -> List[int]:
    selected_set = set(int(idx) for idx in selected_block_indices)
    selected_blocks = [
        block for block in candidate_blocks
        if int(block.get("block_index", -1)) in selected_set
    ]
    if not selected_blocks:
        return []
    best_block = max(selected_blocks, key=score_stair_block)
    return [int(best_block.get("block_index", 0))]


def has_strong_opposite_span_between_blocks(
    spans: List[dict],
    candidate_blocks: List[dict],
    selected_block_indices: List[int],
) -> bool:
    if len(selected_block_indices) <= 1:
        return False

    span_by_id = {}
    for pos, span in enumerate(spans):
        try:
            span_id = int(span.get("span_index", pos))
        except Exception:
            span_id = pos
        span_by_id[span_id] = span

    block_by_id = {
        int(block.get("block_index", idx)): block
        for idx, block in enumerate(candidate_blocks)
    }
    ordered_blocks = [
        block_by_id[idx]
        for idx in sorted(set(int(idx) for idx in selected_block_indices))
        if idx in block_by_id
    ]
    if len(ordered_blocks) <= 1:
        return False

    first_labels = set(ordered_blocks[0].get("labels", []) or [])
    direction = next((label for label in ("up", "down") if label in first_labels), None)
    if direction not in STAIR_VERTICAL_LABELS:
        return False
    opposite = "down" if direction == "up" else "up"

    for prev_block, next_block in zip(ordered_blocks, ordered_blocks[1:]):
        prev_end = int(prev_block.get("end_frame", 0))
        next_start = int(next_block.get("start_frame", 0))
        for span in spans:
            if not isinstance(span, dict):
                continue
            if span.get("label") != opposite:
                continue
            span_start = int(span.get("start_frame", 0))
            span_end = int(span.get("end_frame", 0))
            if span_end <= prev_end or span_start >= next_start:
                continue
            if is_strong_stair_span(span):
                return True
            peak_abs = abs(float(span.get("peak_abs_diff", 0.0)))
            abs_delta = abs(float(span.get("delta", 0.0)))
            if peak_abs >= STAIR_STRONG_PEAK_THRESHOLD or abs_delta >= STAIR_STRONG_DELTA_THRESHOLD:
                return True
    return False
