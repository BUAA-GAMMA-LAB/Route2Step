"""
Fix turn action distribution at segment boundaries.

API identifies what turn (left/right/none) appears at the tail of seg A
and the head of seg B based on instruction text. Then rules allocate
the actual trajectory turn frames.

Rules:
- Only A has turn → turn frames go to A
- Only B has turn → turn frames go to B
- Both same turn → split evenly
- Both different turns → each gets their own (split at midpoint)
- Neither has turn → no change
"""
import json
import os
import copy
import time
import signal
import sys
import re
import threading
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw): return it


MAP_FILE = "data/StreamVLN-Trajectory-Data/RxR/seg/rxr-landmark/rxr_landmark_frame_mapping.json"
ANNOTATIONS_FILE = "data/StreamVLN-Trajectory-Data/RxR/annotations_with_coordinates.json"
OUTPUT_FILE = "data/StreamVLN-Trajectory-Data/RxR/seg/rxr-landmark/rxr_landmark_frame_mapping.json"
CACHE_FILE = "seg/outputs/split/turn_boundary_cache.json"

DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_API_KEYS = "path/to/api_keys.txt"
DEFAULT_MAX_TOKENS = 512

TURN_ACTIONS = {2, 3}
ACTION_NAMES = {2: 'left', 3: 'right'}

SYSTEM_PROMPT = """A_tail = last action of segment A (turn direction or none).
B_head = first action of segment B (turn direction or none).
A: "turn right and climb stairs" -> A_tail: none (last = climb)
A: "walk then turn left" -> A_tail: left
A: "turn around, can see stairs" -> A_tail: around
B: "turn left and walk" -> B_head: left
B: "now walk, then turn right" -> B_head: none (first = walk)

Turn types: left, right, around.

Reply exactly:
A_tail: <left/right/around/none>
B_head: <left/right/around/none>"""


def load_api_keys(path: str) -> List[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def load_coords(ann_path: str) -> Dict[str, List[dict]]:
    with open(ann_path) as f:
        ann_data = json.load(f)
    return {ann['video'].replace('images/', ''): ann.get('coordinates', []) for ann in ann_data}


def find_turn_span_at_boundary(
    frame_coord: dict, boundary_frame: int, total_frames: int,
) -> Tuple[int, int]:
    """Find same-position frames around boundary that contain turns."""
    if boundary_frame not in frame_coord:
        return (boundary_frame, boundary_frame)

    boundary_pos = frame_coord[boundary_frame].get('position', [0, 0, 0])

    def same_pos(p1, p2) -> bool:
        if not p1 or not p2:
            return False
        return abs(p1[0] - p2[0]) + abs(p1[2] - p2[2]) < 0.05

    start = boundary_frame
    end = boundary_frame
    while start > 1 and start - 1 in frame_coord:
        if same_pos(boundary_pos, frame_coord[start - 1].get('position')):
            start -= 1
        else:
            break
    while end < total_frames and end + 1 in frame_coord:
        if same_pos(boundary_pos, frame_coord[end + 1].get('position')):
            end += 1
        else:
            break

    actions = [frame_coord.get(f, {}).get('action', -1) for f in range(start, end + 1)]
    if not any(a in TURN_ACTIONS for a in actions):
        return (boundary_frame, boundary_frame)

    return (start, end)


def collect_boundary_issues(mapping: dict, coord_lookup: dict) -> List[dict]:
    """Collect all boundary turn issues (multi-frame same-pos turn spans)."""
    issues = []
    for ep_id, item in mapping.items():
        dp = item.get('data_path', '')
        coords = coord_lookup.get(dp, [])
        if not coords:
            continue
        nframes = item.get('num_frames', 0)
        frame_coord = {c['step']: c for c in coords}
        segments = item.get('instruction_segments', [])

        for i in range(len(segments) - 1):
            seg_a = segments[i]
            seg_b = segments[i + 1]
            boundary = seg_a['end_frame']
            turn_start, turn_end = find_turn_span_at_boundary(frame_coord, boundary, nframes)
            if turn_start == turn_end:
                continue

            issues.append({
                'cache_key': f"{ep_id}_{i}",
                'ep_id': ep_id,
                'dp': dp,
                'seg_idx': i,
                'si_a': seg_a['sub_instruction'],
                'si_b': seg_b['sub_instruction'],
                'turn_start': turn_start,
                'turn_end': turn_end,
                'current_boundary': boundary,
            })

    return issues


def parse_turn_response(text: str) -> Tuple[str, str]:
    """Parse 'A_tail: left\\nB_head: none' into (a_tail, b_head)."""
    a_tail = b_head = 'none'
    for line in text.strip().split('\n'):
        line = line.strip().lower()
        if line.startswith('a_tail'):
            for d in ('left', 'right', 'none'):
                if d in line:
                    a_tail = d
                    break
        elif line.startswith('b_head'):
            for d in ('left', 'right', 'none'):
                if d in line:
                    b_head = d
                    break
    return a_tail, b_head


def compute_new_boundary(
    a_tail: str, b_head: str,
    turn_start: int, turn_end: int,
    current_boundary: int,
    seg_a_start: int, seg_b_end: int,
) -> int:
    """
    Given A_tail and B_head turn directions, compute where the boundary should be.
    Returns new boundary frame (last frame of seg A).
    """
    if a_tail == 'none' and b_head == 'none':
        return current_boundary

    if a_tail != 'none' and b_head == 'none':
        # Turn belongs to A: move boundary to end of turn span
        return turn_end

    if a_tail == 'none' and b_head != 'none':
        # Turn belongs to B: move boundary to before turn span
        return turn_start - 1

    # Both have turns
    if a_tail == b_head:
        # Same turn → split evenly
        return (turn_start + turn_end) // 2
    else:
        # Different turns → split at midpoint
        return (turn_start + turn_end) // 2

    return current_boundary


def fix_boundaries_with_api(
    mapping: dict, coord_lookup: dict,
    base_url: str, model: str, api_keys_file: str,
    cache_file: str, max_tokens: int,
):
    issues = collect_boundary_issues(mapping, coord_lookup)
    print(f"Boundary turn issues: {len(issues)}")

    cache = {}
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            cache = json.load(f)
        print(f"Loaded cache: {len(cache)} entries")

    pending = [iss for iss in issues if iss['cache_key'] not in cache]
    print(f"Pending API calls: {len(pending)}")

    if not pending:
        print("All done!")
        _apply_and_save(mapping, coord_lookup, cache, OUTPUT_FILE)
        return

    if OpenAI is None:
        raise ImportError("openai required")

    api_keys = load_api_keys(api_keys_file)
    num_workers = len(api_keys)
    print(f"Using {num_workers} workers, model={model}")

    success_count = 0
    fail_count = 0
    _interrupted = False
    _lock = threading.RLock()

    def _save_cache():
        with _lock:
            tmp = cache_file + ".tmp"
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(tmp, 'w') as f:
                json.dump(cache, f, indent=2, ensure_ascii=False)
            os.replace(tmp, cache_file)

    def _on_signal(signum, frame):
        nonlocal _interrupted
        if _interrupted:
            os._exit(1)
        _interrupted = True
        print(f"\n[Signal {signum}] Saving and exiting...")

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)
    _save_cache()

    def _process_one(iss: dict, worker_id: int) -> Tuple[str, str, str]:
        if _interrupted:
            return (iss['cache_key'], 'none', 'none')
        client = OpenAI(api_key=api_keys[worker_id % num_workers], base_url=base_url)
        prompt = f'Segment A: "{iss["si_a"]}"\nSegment B: "{iss["si_b"]}"'

        for attempt in range(3):
            if _interrupted:
                return (iss['cache_key'], 'none', 'none')
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=max_tokens,
                )
                text = resp.choices[0].message.content.strip()
                a_tail, b_head = parse_turn_response(text)
                return (iss['cache_key'], a_tail, b_head)
            except Exception:
                if attempt < 2:
                    time.sleep(2 ** attempt)
        return (iss['cache_key'], 'none', 'none')

    SAVE_INTERVAL = 200

    with tqdm(total=len(pending), unit="boundary", desc="Turn API") as pbar:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            pending_futures = {}
            submit_idx = 0

            while submit_idx < len(pending) and len(pending_futures) < num_workers * 2:
                if _interrupted:
                    break
                iss = pending[submit_idx]
                fut = executor.submit(_process_one, iss, submit_idx % num_workers)
                pending_futures[fut] = iss['cache_key']
                submit_idx += 1

            while pending_futures and not _interrupted:
                try:
                    for fut in as_completed(list(pending_futures.keys()), timeout=3):
                        key = pending_futures.pop(fut, None)
                        try:
                            _, a_tail, b_head = fut.result()
                            with _lock:
                                cache[key] = {'a_tail': a_tail, 'b_head': b_head}
                                success_count += 1
                                total = success_count + fail_count
                                pbar.update(1)
                                pbar.set_postfix(success=success_count)
                                if total % SAVE_INTERVAL == 0:
                                    _save_cache()
                        except Exception:
                            fail_count += 1
                            pass
                        if submit_idx < len(pending) and not _interrupted:
                            iss = pending[submit_idx]
                            fut = executor.submit(_process_one, iss, submit_idx % num_workers)
                            pending_futures[fut] = iss['cache_key']
                            submit_idx += 1
                except FuturesTimeoutError:
                    pass

    _save_cache()
    print(f"\nDone! Success: {success_count}, Fail: {fail_count}")
    _apply_and_save(mapping, coord_lookup, cache, OUTPUT_FILE)


def _apply_and_save(mapping, coord_lookup, cache, output_file):
    """Apply cached decisions to mapping."""
    total_changes = 0
    output = {}

    for ep_id, item in mapping.items():
        dp = item.get('data_path', '')
        coords = coord_lookup.get(dp, [])
        if not coords:
            output[ep_id] = item
            continue

        nframes = item.get('num_frames', 0)
        frame_coord = {c['step']: c for c in coords}
        segments = copy.deepcopy(item.get('instruction_segments', []))

        for i in range(len(segments) - 1):
            key = f"{ep_id}_{i}"
            decision = cache.get(key, {})
            a_tail = decision.get('a_tail', 'none')
            b_head = decision.get('b_head', 'none')

            if a_tail == 'none' and b_head == 'none':
                continue

            seg_a = segments[i]
            seg_b = segments[i + 1]
            boundary = seg_a['end_frame']
            turn_start, turn_end = find_turn_span_at_boundary(frame_coord, boundary, nframes)
            if turn_start == turn_end:
                continue

            new_boundary = compute_new_boundary(
                a_tail, b_head, turn_start, turn_end, boundary,
                seg_a['start_frame'], seg_b['end_frame'],
            )

            if new_boundary != boundary:
                seg_a['end_frame'] = new_boundary
                seg_a['frame_range'] = f"{seg_a['start_frame']}-{new_boundary}"
                seg_b['start_frame'] = new_boundary + 1
                seg_b['frame_range'] = f"{new_boundary + 1}-{seg_b['end_frame']}"
                total_changes += 1

        # Rebuild cut_points
        cut_points = {}
        cut_point_ranges = {}
        for idx, seg in enumerate(segments):
            cut_points[str(idx)] = seg['start_frame']
            cut_point_ranges[str(idx)] = f"{seg['start_frame']}-{seg['end_frame']}"
        last_seg = segments[-1]
        cut_points[str(len(segments))] = last_seg['end_frame']
        cut_point_ranges[str(len(segments))] = f"{last_seg['end_frame']}-{last_seg['end_frame']}"

        fixed = dict(item)
        fixed['instruction_segments'] = segments
        fixed['segment_ranges'] = [
            {'segment_index': s['sub_instruction_index'],
             'start_frame': s['start_frame'], 'end_frame': s['end_frame']}
            for s in segments
        ]
        fixed['cut_points'] = cut_points
        fixed['cut_point_ranges'] = cut_point_ranges
        output[ep_id] = fixed

    bak = output_file.replace('.json', '_before_turn_api_fix.json')
    if not os.path.exists(bak):
        import shutil
        shutil.copy2(output_file, bak)
        print(f"Backup: {bak}")

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Applied {total_changes} boundary changes, saved to {output_file}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api_keys_file", default=DEFAULT_API_KEYS)
    parser.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    with open(MAP_FILE) as f:
        mapping = json.load(f)
    print(f"Loaded {len(mapping)} episodes")

    if args.limit > 0:
        mapping = {k: v for i, (k, v) in enumerate(mapping.items()) if i < args.limit}

    coord_lookup = load_coords(ANNOTATIONS_FILE)
    issues = collect_boundary_issues(mapping, coord_lookup)
    print(f"Boundary turn issues: {len(issues)}")

    if args.dry_run:
        # Count by keyword (for comparison)
        a_only = b_only = both = neither = 0
        for iss in issues:
            a = any(w in iss['si_a'].lower() for w in ['turn', 'take a left', 'take a right'])
            b = any(w in iss['si_b'].lower() for w in ['turn', 'take a left', 'take a right'])
            if a and b: both += 1
            elif a: a_only += 1
            elif b: b_only += 1
            else: neither += 1
        print(f"  Only A: {a_only}, Only B: {b_only}, Both: {both}, Neither: {neither}")
        print(f"  Pending API (all): {len(issues)}")
        return

    fix_boundaries_with_api(
        mapping=mapping, coord_lookup=coord_lookup,
        base_url=args.base_url, model=args.model,
        api_keys_file=args.api_keys_file, cache_file=CACHE_FILE,
        max_tokens=args.max_tokens,
    )


if __name__ == '__main__':
    main()
