"""
Use Qwen API to map R2R sub-instructions to action labels.

Supports both the original episode-level sub-instruction files and the newer
split-existing-subinstruction outputs.
"""
import os
import json
import re
import time
import argparse
from openai import OpenAI
from tqdm import tqdm

BATCH_SIZE = 50
VALID_ACTIONS = {"move forward", "turn left", "turn right", "stop"}
API_KEYS_FILE = "path/to/api_keys.txt"

DEFAULT_INPUT_FILE = "data/StreamVLN-Trajectory-Data/R2R/r2r_train_seg_fix_stairs.json"
DEFAULT_OUTPUT_FILE = "seg/outputs/actions/sub_instruction_actions_r2r_train.json"

SYSTEM_PROMPT = """You are a navigation action classifier for an embodied robot.

For each sub-instruction, usually output EXACTLY ONE action from: move forward, turn left, turn right, stop.
Choose the single closest action label for the sub-instruction.

Priority rules (when the instruction contains multiple actions, pick by priority):
1. stop - if the agent arrives at / stands at the final destination ("stand", "stop", "halt", "final destination", "end point", "you're done", "that's your end point")
2. turn left - if the instruction involves a left turn ("turn left", "bear left", "make a left")
3. turn right - if the instruction involves a right turn ("turn right", "bear right", "make a right")
4. move forward - for all forward movement verbs: walk, go, head, proceed, pass, enter, exit, continue, ascend, descend, climb, move, approach, cross

Special cases:
- "Turn around" / "U-turn" -> ["turn left", "turn right"]
- Reorientation without explicit left/right direction, such as "Turn to face the doorway" or "Turn and face the sink" -> ["turn left", "turn right"]
- "Face X" or "Stand facing X" without walking -> stop (orientation-only, arrival)
- Compound "Turn left and walk" -> turn left (turn is the primary navigation decision)
- Compound "Walk forward then turn right" -> turn right (the turn is the key change)
- "Ascend/descend steps" -> move forward (vertical movement is still forward)
- "Click between X and Y" -> stop (click = arrive at waypoint)
- "Enter the room" -> move forward
- "Locate / identify / find" (no movement verb) -> move forward (searching while walking)

Respond ONLY with a JSON object.
Values are usually strings.
Only the two ambiguous-turn exceptions above may use ["turn left", "turn right"]."""

TURN_AROUND_RE = re.compile(r"\bturn\s+around\b|\bu[- ]?turn\b", re.I)
LEFT_RE = re.compile(r"\b(?:turn|bear|veer|make|take|slide)\b.*\bleft\b|\bleft\s+turn\b", re.I)
RIGHT_RE = re.compile(r"\b(?:turn|bear|veer|make|take|slide)\b.*\bright\b|\bright\s+turn\b", re.I)
TURN_TO_FACE_AMBIG_RE = re.compile(
    r"\bturn\s+to\s+face\b|\bturn\s+and\s+face\b|\bturn\b(?:[^.!?;:]*)\bface\b",
    re.I,
)
STOP_RE = re.compile(
    r"\b(?:stand|stop|wait|halt|final destination|end point|endpoint|destination)\b",
    re.I,
)
MOVE_RE = re.compile(
    r"\b(?:walk|go|head|proceed|pass|enter|exit|continue|straight|forward|through|towards?|"
    r"cross|climb|descend|ascend|move|approach|step|skirt|circle|curve|loop)\b",
    re.I,
)
FACE_ONLY_RE = re.compile(r"^\s*(?:face|facing)\b", re.I)


def load_api_keys():
    with open(API_KEYS_FILE) as f:
        return [line.strip() for line in f if line.strip()]


def build_prompt(batch):
    lines = []
    for i, text in enumerate(batch, 1):
        lines.append(f'{i}. "{text}"')
    instructions_str = "\n".join(lines)
    return (
        "Classify the single closest navigation action for each sub-instruction below.\n"
        "Output one action per item: move forward / turn left / turn right / stop.\n"
        "Exception: output [\"turn left\", \"turn right\"] only for turn-around / U-turn "
        "or turn-to-face-without-direction cases.\n\n"
        f"{instructions_str}\n\n"
        'Output JSON only, like: {"1": "turn left", "2": "move forward", '
        '"3": ["turn left", "turn right"], "4": "stop"}'
    )


def keyword_fallback(text):
    t = text.lower()
    if STOP_RE.search(t):
        return "stop"
    if TURN_AROUND_RE.search(t):
        return ["turn left", "turn right"]
    if TURN_TO_FACE_AMBIG_RE.search(t) and not LEFT_RE.search(t) and not RIGHT_RE.search(t):
        return ["turn left", "turn right"]
    if LEFT_RE.search(t):
        return "turn left"
    if RIGHT_RE.search(t):
        return "turn right"
    if FACE_ONLY_RE.search(t):
        return "stop"
    if MOVE_RE.search(t):
        return "move forward"
    return "move forward"


def canonicalize_existing_action(raw):
    if isinstance(raw, str) and raw in VALID_ACTIONS:
        return raw
    if isinstance(raw, list):
        valid = [item for item in raw if item in VALID_ACTIONS]
        if not valid:
            return None
        if set(valid) == {"turn left", "turn right"}:
            return ["turn left", "turn right"]
        return valid[0]
    return None


def parse_response(response_text, batch):
    try:
        match = re.search(r"\{.*\}", response_text, re.S)
        if not match:
            return None
        obj = json.loads(match.group())
        result = {}
        for i, text in enumerate(batch, 1):
            key = str(i)
            raw = obj.get(key)
            if isinstance(raw, str) and raw in VALID_ACTIONS:
                result[text] = raw
            elif isinstance(raw, list):
                valid = [item for item in raw if item in VALID_ACTIONS]
                if set(valid) == {"turn left", "turn right"}:
                    result[text] = ["turn left", "turn right"]
                else:
                    result[text] = keyword_fallback(text)
            else:
                result[text] = keyword_fallback(text)
        return result
    except Exception:
        return None


def ordered_subinstructions(raw):
    if isinstance(raw, dict):
        try:
            keys = sorted(raw.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))
        except Exception:
            keys = sorted(raw.keys())
        values = [raw[key] for key in keys]
    elif isinstance(raw, list):
        values = list(raw)
    else:
        values = []
    return [str(text).strip() for text in values if str(text).strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default=DEFAULT_INPUT_FILE)
    parser.add_argument("--output_file", type=str, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--base_url", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--model", type=str, default="qwen3-vl-flash")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    clients = [OpenAI(api_key=k, base_url=args.base_url) for k in load_api_keys()]
    print(f"Using {len(clients)} API keys in rotation.")

    with open(args.input_file, encoding="utf-8") as f:
        data = json.load(f)

    all_subs = set()
    for ep in data:
        for s in ordered_subinstructions(ep.get("sub_instructions", [])):
            if s:
                all_subs.add(s)
    all_subs = sorted(all_subs)
    print(f"Total unique sub-instructions: {len(all_subs)}")

    result = {}
    if os.path.exists(args.output_file):
        with open(args.output_file, encoding="utf-8") as f:
            saved = json.load(f)
        for key, value in saved.items():
            normalized = canonicalize_existing_action(value)
            if normalized is not None:
                result[key] = normalized
        print(f"Loaded {len(result)} existing entries from {args.output_file}")

    pending = [s for s in all_subs if s not in result]
    print(f"Pending: {len(pending)}")

    batch_size = args.batch_size
    total_batches = (len(pending) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(total_batches), desc="Batches", unit="batch"):
        batch = pending[batch_idx * batch_size:(batch_idx + 1) * batch_size]
        prompt = build_prompt(batch)

        success = False
        for attempt in range(3):
            client = clients[batch_idx % len(clients)]
            try:
                resp = client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=1024,
                )
                text = resp.choices[0].message.content.strip()
                parsed = parse_response(text, batch)
                if parsed:
                    result.update(parsed)
                    success = True
                    break
                else:
                    print(f"[Batch {batch_idx+1}] Parse failed attempt {attempt+1}, retrying")
            except Exception as e:
                print(f"[Batch {batch_idx+1}] API error attempt {attempt+1}: {e}")
                time.sleep(2 ** attempt)

        if not success:
            for text in batch:
                result[text] = keyword_fallback(text)

        if (batch_idx + 1) % 10 == 0 or batch_idx == total_batches - 1:
            os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
            with open(args.output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Done. Total entries: {len(result)} -> {args.output_file}")


if __name__ == "__main__":
    main()
