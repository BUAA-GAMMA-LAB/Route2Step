"""Generate an action mapping for R2R val-unseen sub-instructions.

The output maps each sub-instruction text to one or more atomic actions.
"""
import os
import json
import re
import time
import argparse
from openai import OpenAI

INPUT_FILE = "data/datasets/R2R_VLNCE_v1-3/val_unseen/val_unseen_split.json"
OUTPUT_FILE = "seg/outputs/actions/sub_instruction_actions_r2r_vlnce_v1_3_val_unseen.json"
DEFAULT_API_KEYS = "path/to/api_keys.txt"
BATCH_SIZE = 50
VALID_ACTIONS = {"move forward", "turn left", "turn right", "stop"}

SYSTEM_PROMPT = """You are a navigation action classifier. For each sub-instruction, identify which atomic navigation actions it involves.
Available actions: move forward, turn left, turn right, stop.
Rules:
- "walk/go/head/proceed/pass/enter/exit/continue" → move forward
- "turn/make a left/go left/bear left" → turn left
- "turn/make a right/go right/bear right" → turn right
- "stop/wait/stand/halt" → stop
- "face/orient/pivot/rotate toward ..." → turn left or turn right if the direction is explicit; otherwise move forward
- "ascend/descend/climb stairs/steps" → move forward
- Most instructions involve "move forward" as the main action
- An instruction can have multiple actions
Respond ONLY with a JSON object mapping each number to its action list."""


def read_api_keys(path):
    with open(path, "r", encoding="utf-8") as f:
        keys = [line.strip() for line in f if line.strip()]
    if not keys:
        raise ValueError(f"No API keys found in {path}")
    return keys

def build_prompt(batch):
    lines = []
    for i, text in enumerate(batch, 1):
        lines.append(f'{i}. "{text}"')
    instructions_str = "\n".join(lines)
    return (
        f"Classify the navigation actions for each sub-instruction below.\n\n"
        f"{instructions_str}\n\n"
        f'Output JSON only, like: {{"1": ["move forward", "turn left"], "2": ["move forward"]}}'
    )

def keyword_fallback(text):
    t = text.lower()
    actions = []
    if any(w in t for w in ['left', 'l turn']):
        actions.append('turn left')
    if any(w in t for w in ['right', 'r turn']):
        actions.append('turn right')
    if any(w in t for w in ['stop', 'wait', 'halt', 'stand']):
        actions.append('stop')
    if any(w in t for w in ['walk', 'go', 'head', 'proceed', 'pass', 'enter', 'exit',
                              'continue', 'straight', 'forward', 'through', 'towards',
                              'toward', 'cross', 'up', 'down', 'climb', 'descend',
                              'ascend', 'approach', 'step', 'travel', 'move']):
        actions.append('move forward')
    if not actions:
        actions = ['move forward']
    return actions

def parse_response(response_text, batch):
    try:
        match = re.search(r'\{.*\}', response_text, re.S)
        if not match:
            return None
        obj = json.loads(match.group())
        result = {}
        for i, text in enumerate(batch, 1):
            key = str(i)
            if key in obj:
                raw = obj[key]
                if isinstance(raw, list):
                    filtered = [a for a in raw if a in VALID_ACTIONS]
                    result[text] = filtered if filtered else ['move forward']
                else:
                    result[text] = ['move forward']
            else:
                result[text] = keyword_fallback(text)
        return result
    except Exception:
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=INPUT_FILE)
    parser.add_argument("--output", type=str, default=OUTPUT_FILE)
    parser.add_argument("--api_keys_file", type=str, default=DEFAULT_API_KEYS)
    parser.add_argument("--base_url", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--model", type=str, default="qwen-plus")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--request_timeout", type=float, default=60.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    keys = read_api_keys(args.api_keys_file)
    clients = [OpenAI(api_key=k, base_url=args.base_url, timeout=args.request_timeout) for k in keys]
    print(f"Using {len(clients)} API keys in rotation from {args.api_keys_file}.")

    # Collect unique sub-instructions.
    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)
    all_subs = set()
    for ep in data['episodes']:
        for s in ep.get('split_instructions', []):
            cleaned = re.sub(r'^\s*\d+[\.\)\s:-]*', '', s).strip()
            if cleaned:
                all_subs.add(cleaned)
    all_subs = sorted(all_subs)
    print(f"Total unique sub-instructions: {len(all_subs)}")

    # Load any existing mapping so completed entries can be reused.
    result = {}
    if os.path.exists(args.output) and not args.overwrite:
        with open(args.output, encoding="utf-8") as f:
            result = json.load(f)
        print(f"Loaded {len(result)} existing entries from {args.output}")

    pending = [s for s in all_subs if s not in result]
    print(f"Pending: {len(pending)}")

    batch_size = args.batch_size
    total_batches = (len(pending) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        batch = pending[batch_idx * batch_size:(batch_idx + 1) * batch_size]
        prompt = build_prompt(batch)

        success = False
        for attempt in range(args.max_retries):
            client = clients[batch_idx % len(clients)]
            try:
                print(f"[Batch {batch_idx+1}/{total_batches}] attempt {attempt+1}/{args.max_retries}", flush=True)
                resp = client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
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
                    print(f"[Batch {batch_idx+1}] Parse failed, attempt {attempt+1}, using fallback")
            except Exception as e:
                print(f"[Batch {batch_idx+1}] API error attempt {attempt+1}: {e}")
                time.sleep(2 ** attempt)

        if not success:
            for text in batch:
                result[text] = keyword_fallback(text)

        if (batch_idx + 1) % 10 == 0 or batch_idx == total_batches - 1:
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"[{batch_idx+1}/{total_batches}] Saved {len(result)} entries.")

    print(f"Done. Total entries: {len(result)} -> {args.output}")

if __name__ == "__main__":
    main()
