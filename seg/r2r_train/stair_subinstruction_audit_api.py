#!/usr/bin/env python3
"""
Audit stair-related sub-instructions with an LLM and propose rewrites
that isolate true vertical stair traversal into standalone sub-instructions.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
import re
import time
from typing import Dict, List, Optional

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
    from seg.r2r_train.stair_alignment_utils import is_down_stair_motion_text, is_up_stair_motion_text
except ImportError:
    from stair_alignment_utils import is_down_stair_motion_text, is_up_stair_motion_text

try:
    from seg.r2r_train.stair_dataset_utils import get_stair_dataset_config, load_episode_dataset
except ImportError:
    from stair_dataset_utils import get_stair_dataset_config, load_episode_dataset

DEFAULT_QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_QWEN_MODEL = "qwen3-vl-flash"
DEFAULT_QWEN_API_KEYS = "path/to/api_keys.txt"
DEFAULT_BATCH_SIZE = 20
DEFAULT_MAX_WORKERS = 8
DEFAULT_DATASET = "r2r"
STAIR_CONTEXT_WORDS = ["stair", "stairs", "staircase", "stairway", "steps", "flight", "landing"]

AUDIT_SYSTEM_PROMPT = """You are an indoor-navigation sub-instruction auditor.

You will receive short sub-instructions that mention stairs or stair traversal.
Your goal is to determine whether each instruction contains real vertical stair traversal,
and if so, isolate the true stair traversal as its own standalone sub-instruction.

Important rules:
- True vertical stair traversal means physically moving up/down stairs, steps, a staircase, or a stairway.
- "walk up to the stairs", "go to the stairs", "turn toward the stairs", and similar approach-only text are NOT vertical traversal.
- "walk up the stairs", "go down the steps", "climb the staircase", "descend the stairs", and "go upstairs/downstairs" ARE vertical traversal.
- If an instruction mixes approach/reorientation/other motion with vertical stair traversal, split it into 2-3 short ordered sub-instructions.
- If one original instruction contains multiple separate stair traversals, isolate each traversal as its own pure stair-traversal sub-instruction.
- Every rewritten sub-instruction listed in `vertical_sub_instruction_indices` must contain only the actual stair traversal, without approach-only text.
- Never drop an explicit action that already appears in the original instruction.
- Preserve all original actions, including approach, turning, entering, passing, stopping, waiting, and standing, as separate non-vertical sub-instructions when needed.
- A stair traversal may include its own endpoint, such as reaching the top, bottom, or landing of that same staircase.
- Do not demote a stair traversal to non-vertical just because it ends at the top/bottom/landing before the next action begins.
- Keep wording close to the original. Do not invent new objects, destinations, or actions.
- Preserve execution order.
- If no split is needed, return the original instruction as the only rewritten sub-instruction.

Examples:
- "Walk up to the stairs." -> no vertical traversal, ["Walk up to the stairs."]
- "Walk up the stairs." -> vertical traversal, ["Walk up the stairs."]
- "Turn left and walk up the stairs." -> vertical traversal, ["Turn left.", "Walk up the stairs."]
- "Walk past the couch and go down the stairs." -> vertical traversal, ["Walk past the couch.", "Go down the stairs."]
- "Ascend two steps and then stop." -> vertical traversal, ["Ascend two steps.", "Stop."]
- "Walk up the stairs and enter the room." -> vertical traversal, ["Walk up the stairs.", "Enter the room."]
- "After climbing four steps and reaching the top, turn left and climb five more steps." -> vertical traversal, ["Climb four steps and reach the top.", "Turn left.", "Climb five more steps."]
- "Turn right towards the stairs." -> no vertical traversal, ["Turn right towards the stairs."]

Return JSON only, mapping each number to:
{
  "1": {
    "contains_vertical_stair_traversal": true,
    "rewritten_sub_instructions": ["Turn left.", "Walk up the stairs."],
    "vertical_sub_instruction_indices": [1],
    "reason": "..."
  }
}
"""


def clean_instruction(text: str) -> str:
    return re.sub(r"^\s*\d+[\d\.\)\s:-]*", "", str(text)).strip()


def normalize_sentence(text: str) -> str:
    text = clean_instruction(text)
    text = re.sub(r"\s+", " ", text).strip(" ,;")
    if not text:
        return ""
    if text[-1] not in ".!?":
        text += "."
    return text


def contains_stop_like_action(text: str) -> bool:
    t = str(text).lower()
    return any(token in t for token in ("stop", "wait", "stand", "halt"))


def is_missing_explicit_stop_like_action(original_text: str, rewritten_sub_instructions: List[str]) -> bool:
    if not contains_stop_like_action(original_text):
        return False
    return not any(contains_stop_like_action(text) for text in rewritten_sub_instructions)
def has_stair_context(text: str) -> bool:
    t = str(text).lower()
    return any(word in t for word in STAIR_CONTEXT_WORDS)


def contains_vertical_stair_traversal(text: str) -> bool:
    t = str(text).lower()
    return has_stair_context(t) and (is_up_stair_motion_text(t) or is_down_stair_motion_text(t))


def load_api_keys(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"API keys file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        keys = [line.strip() for line in f if line.strip()]
    if not keys:
        raise ValueError(f"No API keys found in {path}")
    return keys
def build_prompt(batch: List[str]) -> str:
    lines = [f'- [{idx}] "{text}"' for idx, text in enumerate(batch)]
    return (
        "Audit the following stair-related sub-instructions.\n"
        "Rewrite only when needed to isolate the true stair traversal as its own sub-instruction.\n\n"
        f"{chr(10).join(lines)}\n\n"
        "Return JSON only."
    )


def normalize_index_list(values, upper_bound: int) -> List[int]:
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


def heuristic_audit(text: str) -> dict:
    normalized = normalize_sentence(text)
    contains_vertical = contains_vertical_stair_traversal(normalized)
    return {
        "original_text": normalized,
        "contains_vertical_stair_traversal": contains_vertical,
        "rewritten_sub_instructions": [normalized] if normalized else [],
        "vertical_sub_instruction_indices": [0] if contains_vertical and normalized else [],
        "split_required": False,
        "reason": "heuristic_fallback",
        "source": "heuristic_fallback",
    }


def canonicalize_audit_entry(raw: dict, original_text: str) -> dict:
    fallback = heuristic_audit(original_text)
    if not isinstance(raw, dict):
        return fallback

    rewritten = raw.get("rewritten_sub_instructions", [])
    if not isinstance(rewritten, list):
        rewritten = []
    rewritten = [normalize_sentence(text) for text in rewritten]
    rewritten = [text for text in rewritten if text]
    if not rewritten:
        rewritten = fallback["rewritten_sub_instructions"]

    provided_vertical_indices = normalize_index_list(raw.get("vertical_sub_instruction_indices", []), len(rewritten))
    inferred_vertical_indices = [
        idx for idx, text in enumerate(rewritten)
        if contains_vertical_stair_traversal(text)
    ]
    valid_provided_vertical_indices = [
        idx for idx in provided_vertical_indices
        if idx in inferred_vertical_indices
    ]

    contains_vertical = bool(raw.get("contains_vertical_stair_traversal"))
    if inferred_vertical_indices:
        vertical_indices = sorted(set(valid_provided_vertical_indices) | set(inferred_vertical_indices))
    else:
        vertical_indices = provided_vertical_indices
    if contains_vertical and not vertical_indices:
        vertical_indices = inferred_vertical_indices
    if not contains_vertical and inferred_vertical_indices:
        contains_vertical = True
        vertical_indices = inferred_vertical_indices
    if contains_vertical and not vertical_indices:
        return fallback

    if not contains_vertical:
        vertical_indices = []

    split_required = bool(raw.get("split_required", len(rewritten) > 1))
    if len(rewritten) > 1:
        split_required = True

    return {
        "original_text": normalize_sentence(original_text),
        "contains_vertical_stair_traversal": contains_vertical,
        "rewritten_sub_instructions": rewritten,
        "vertical_sub_instruction_indices": vertical_indices,
        "split_required": split_required,
        "reason": str(raw.get("reason", "")).strip(),
        "source": raw.get("source", "api"),
    }


def parse_response(response_text: str, batch: List[str]) -> Optional[Dict[str, dict]]:
    try:
        match = re.search(r"\{.*\}", response_text, re.S)
        if not match:
            return None
        obj = json.loads(match.group())
    except Exception:
        return None

    parsed = {}
    for idx, original_text in enumerate(batch):
        raw = obj.get(str(idx))
        entry = canonicalize_audit_entry(raw, original_text)
        if is_missing_explicit_stop_like_action(original_text, entry.get("rewritten_sub_instructions", [])):
            return None
        parsed[original_text] = entry
    return parsed


def collect_candidate_texts(episodes: List[dict]) -> List[str]:
    seen = set()
    candidates = []
    for item in episodes:
        for text in item.get("sub_instructions", []):
            normalized = normalize_sentence(text)
            if not normalized or normalized in seen:
                continue
            if contains_vertical_stair_traversal(normalized):
                seen.add(normalized)
                candidates.append(normalized)
    return sorted(candidates)


def call_api_for_audit(
    candidate_texts: List[str],
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
        with open(output_file, "r", encoding="utf-8") as f:
            raw_result = json.load(f)
        result = {
            normalize_sentence(text): canonicalize_audit_entry(entry, text)
            for text, entry in raw_result.items()
        }
    else:
        result = {}

    pending = [text for text in candidate_texts if text not in result]
    if not pending:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return result

    api_keys = load_api_keys(api_keys_file)
    batches = [
        pending[batch_idx * batch_size:(batch_idx + 1) * batch_size]
        for batch_idx in range(math.ceil(len(pending) / batch_size))
    ]

    def process_batch(batch_idx: int, batch: List[str]) -> Dict[str, dict]:
        prompt = build_prompt(batch)
        parsed = None
        last_error = None

        for attempt in range(3):
            api_key = api_keys[(batch_idx + attempt) % len(api_keys)]
            client = OpenAI(api_key=api_key, base_url=base_url)
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": AUDIT_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                )
                text = resp.choices[0].message.content or ""
                parsed = parse_response(text, batch)
                if parsed is not None:
                    break
                last_error = "failed_to_parse_response"
            except Exception as exc:
                last_error = str(exc)
                time.sleep(1.5)

        if parsed is None:
            parsed = {}
            for text in batch:
                entry = heuristic_audit(text)
                entry["error"] = last_error
                parsed[text] = entry
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
        return parsed

    max_workers = max(1, min(max_workers, len(batches), len(api_keys)))
    with tqdm(total=len(pending), unit="sub", desc="Stair sub-instruction audit") as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {
                executor.submit(process_batch, batch_idx, batch): batch
                for batch_idx, batch in enumerate(batches)
            }
            for future in as_completed(future_to_batch):
                batch = future_to_batch[future]
                parsed = future.result()
                for text in batch:
                    result[text] = canonicalize_audit_entry(parsed.get(text), text)

                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)

                pbar.update(len(batch))

    return result


def build_episode_rewrite_proposals(episodes: List[dict], audit_result: Dict[str, dict]) -> Dict[str, dict]:
    proposals = {}
    for item in episodes:
        episode_id = str(item.get("episode_id"))
        original_subs = [normalize_sentence(text) for text in item.get("sub_instructions", []) if normalize_sentence(text)]
        if not original_subs:
            continue

        proposed_subs = []
        changed_instruction_indices = []
        proposed_vertical_indices = []

        for original_idx, text in enumerate(original_subs):
            entry = audit_result.get(text)
            rewrite = list(entry.get("rewritten_sub_instructions", [])) if entry else [text]
            if not rewrite:
                rewrite = [text]
            start_pos = len(proposed_subs)
            proposed_subs.extend(rewrite)
            if rewrite != [text]:
                changed_instruction_indices.append(original_idx)
            if entry and entry.get("contains_vertical_stair_traversal"):
                for rel_idx in entry.get("vertical_sub_instruction_indices", []):
                    proposed_vertical_indices.append(start_pos + rel_idx)

        if not changed_instruction_indices:
            continue

        proposals[episode_id] = {
            "episode_id": int(item.get("episode_id", 0)),
            "original_sub_instructions": original_subs,
            "proposed_sub_instructions": proposed_subs,
            "changed_instruction_indices": changed_instruction_indices,
            "proposed_vertical_sub_instruction_indices": proposed_vertical_indices,
            "requires_resegmentation": True,
        }
    return proposals


def main():
    parser = argparse.ArgumentParser(description="Audit and rewrite stair-related sub-instructions with an API.")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, choices=("r2r",))
    parser.add_argument("--input_file", type=str, default="")
    parser.add_argument("--audit_output_file", type=str, default="")
    parser.add_argument("--episode_output_file", type=str, default="")
    parser.add_argument("--base_url", type=str, default=DEFAULT_QWEN_BASE_URL)
    parser.add_argument("--model", type=str, default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--api_keys_file", type=str, default=DEFAULT_QWEN_API_KEYS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max_workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--sleep_seconds", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0, help="Limit the number of unique candidate texts.")
    parser.add_argument("--ignore_existing", action="store_true")
    args = parser.parse_args()

    dataset_cfg = get_stair_dataset_config(args.dataset)
    input_file = args.input_file or dataset_cfg["input_file"]
    audit_output_file = args.audit_output_file or dataset_cfg["audit_output_file"]
    episode_output_file = args.episode_output_file or dataset_cfg["rewrite_file"]

    episodes = load_episode_dataset(input_file, args.dataset)
    candidate_texts = collect_candidate_texts(episodes)
    if args.limit > 0:
        candidate_texts = candidate_texts[:args.limit]

    print(f"Dataset={args.dataset} loaded episodes: {len(episodes)}")
    print(f"Unique stair-traversal candidate sub-instructions: {len(candidate_texts)}")

    result = call_api_for_audit(
        candidate_texts=candidate_texts,
        base_url=args.base_url,
        model=args.model,
        api_keys_file=args.api_keys_file,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        output_file=audit_output_file,
        load_existing=(not args.ignore_existing),
        sleep_seconds=args.sleep_seconds,
    )
    print(f"Done audit. {len(result)} entries -> {audit_output_file}")

    proposals = build_episode_rewrite_proposals(episodes, result)
    os.makedirs(os.path.dirname(episode_output_file), exist_ok=True)
    with open(episode_output_file, "w", encoding="utf-8") as f:
        json.dump(proposals, f, ensure_ascii=False, indent=2)

    split_count = sum(1 for entry in result.values() if entry.get("split_required"))
    vertical_count = sum(1 for entry in result.values() if entry.get("contains_vertical_stair_traversal"))
    print(f"Vertical stair traversal entries: {vertical_count}")
    print(f"Split-required entries: {split_count}")
    print(f"Changed episodes needing re-segmentation: {len(proposals)} -> {episode_output_file}")


if __name__ == "__main__":
    main()
