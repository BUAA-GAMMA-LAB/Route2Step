import base64
import json
import numpy as np
import os
from collections import OrderedDict
from io import BytesIO
import re
import time
import torch
import cv2
from habitat.core.agent import Agent
from habitat.utils.visualizations import maps
from PIL import Image

# Set image token limits for the visual model interface.
os.environ['MAX_PIXELS'] = '1003520'
os.environ['IMAGE_MAX_TOKEN_NUM'] = '2048'

try:
    from swift.infer_engine import TransformersEngine, InferRequest, RequestConfig
except ImportError:
    print("Error: ms-swift not found. Please install it to ensure alignment.")
    raise


SYSTEM_PROMPT = "You are an intelligent navigation robot."
M1_RECOVERING_PREFIX_RE = re.compile(r'^\s*(?:Recovering|Recovring)\s*[:：]\s*', re.I)
M1_RESEGMENT_CHAR_LIMIT = 30
M1_RESEGMENT_CONTEXT_WINDOW = 60
M1_RESEGMENT_HISTORY_FRAMES = 13
M1_RESEGMENT_CURRENT_FRAMES = 3


def _format_image_count_desc(num_total):
    if num_total == 1:
        return "Above is 1 image."
    return f"Above are {num_total} images."


def build_m2_prompt(global_inst, sub_inst, num_total):
    if num_total == 1:
        desc = "It is the Current view."
    else:
        desc = "They are ordered from earlier to more recent views."

    return (
        f"{'<image>' * num_total}\n"
        f"{_format_image_count_desc(num_total)} {desc}\n"
        f"Global Instruction: {global_inst}\n"
        f"Current Sub-instruction: {sub_inst}\n"
        f"Task: Provide the next 3 actions to execute for the current sub-instruction. "
        f"Available actions: 1) Move forward 25 cm, 2) Turn left 15 degrees, 3) Turn right 15 degrees, 4) stop. "
        f"Output stop only when the entire task is completed."
    )


def _denser_at_end_sampling(pool_size, num_to_select, power=1.5):
    if num_to_select <= 0:
        return []
    if num_to_select >= pool_size:
        return list(range(pool_size))
    positions = np.linspace(0, 1, num_to_select)
    transformed_positions = 1 - (1 - positions) ** power
    sampled_indices = np.round(transformed_positions * (pool_size - 1)).astype(int)
    unique_indices = sorted(list(set(sampled_indices)))
    while len(unique_indices) < num_to_select:
        available = sorted(list(set(range(pool_size)) - set(unique_indices)))
        if not available:
            break
        unique_indices.append(available[-1])
        unique_indices = sorted(unique_indices)
    return unique_indices


def _uniform_sparse_sampling(pool_size, num_to_select):
    if num_to_select <= 0:
        return []
    if num_to_select >= pool_size:
        return list(range(pool_size))
    sampled_indices = np.linspace(0, pool_size - 1, num_to_select + 1)
    return np.round(sampled_indices).astype(int).tolist()[:num_to_select]


def _extract_m1_sub_instruction(response):
    """Extract the executable sub-instruction from an M1 response."""
    ans_match = re.search(r'<answer>(.*?)</answer>', response, re.S | re.I)
    if ans_match:
        return ans_match.group(1).strip()

    sub_inst = re.sub(r'<think>.*?</think>', '', response, flags=re.S | re.I).strip()
    return re.sub(
        r'^(Analyze|Reasoning|Instruction|Next Instruction to Execute):\s*',
        '',
        sub_inst,
        flags=re.I,
    )


def _instruction_char_count(instruction):
    """Count visible instruction characters, ignoring formatting whitespace."""
    return len(re.sub(r'\s+', '', instruction or ''))


def get_images_for_modules_from_pil_history(
    rgb_pil_history,
    mode="module1",
    m1_history_frames=13,
    m1_current_frames=3,
    m2_recent_window=40,
    m2_num_frames=8,
):
    if not rgb_pil_history:
        return [], 0, 0

    indices = list(range(len(rgb_pil_history)))

    if mode == "module1":
        if len(indices) >= m1_current_frames:
            current_indices = indices[-m1_current_frames:]
            # 2-frame gap between history and current (history ends at t-5, current starts at t-2)
            history_end = max(0, len(indices) - m1_current_frames - 2)
            history_pool = indices[:history_end]
        else:
            current_indices = indices
            history_pool = []

        if len(history_pool) > m1_history_frames:
            selected_h = _denser_at_end_sampling(len(history_pool), m1_history_frames, power=1.5)
            history_indices = [history_pool[i] for i in selected_h]
        else:
            history_indices = history_pool

        final_indices = history_indices + current_indices
        num_h, num_c = len(history_indices), len(current_indices)
    else:
        recent_window = indices[-m2_recent_window:]
        selected_indices = _denser_at_end_sampling(len(recent_window), m2_num_frames, power=1.5)
        final_indices = [recent_window[i] for i in selected_indices]
        num_h, num_c = 0, len(final_indices)

    return [rgb_pil_history[i] for i in final_indices], num_h, num_c


class DualReason_Agent(Agent):
    def __init__(
        self,
        model1_path,
        model2_path,
        result_path,
        require_map=True,
        device_map="auto",
        model_type="qwen2_5_vl",
        torch_dtype="bfloat16",
        image_size=(640, 480),
        m1_max_tokens=256,
        m2_max_tokens=32,
        action_horizon=3,
        m1_history_frames=13,
        m1_current_frames=3,
        m1_history_window=0,
        m1_turn_aware_sampling=False,
        m1_turn_frame_budget=3,
        enable_m1_recursive_split=False,
        m2_recent_window=40,
        m2_num_frames=8,
        jpeg_quality=90,  # JPEG quality for image serialization.
        http_timeout_s=120.0,
        http_max_connections=32,
        http_max_keepalive_connections=8,
        http_retry_count=3,
        http_retry_backoff_s=2.0,
        http_retry_max_backoff_s=8.0,
        http_force_connection_close=False,
        enable_image_data_url_cache=True,
        image_data_url_cache_size=160,
        stop_on_m1_stop_token=True,
        enable_align_turn_rule=False,
        filter_align_memory_for_m2=False,
        strip_m1_recovering_prefix_for_m2=False,
        m2_recover_max_consecutive=0,
        m2_recover_cooldown=0,
        use_single_model=False,
            # Optional vLLM server mode. When set, local model loading is skipped.
        m1_server_url=None,
        m1_server_model="m1",
        m2_server_url=None,
        m2_server_model="m2",
    ):
        self.use_single_model = bool(use_single_model)
        mode_name = "Single-Model" if self.use_single_model else "Dual-Model"
        print(f"Initializing {mode_name} Agent (Architecture: {model_type})...")

        self.result_path = result_path
        self.require_map = require_map
        os.makedirs(self.result_path, exist_ok=True)

        self.image_size = image_size
        self.action_horizon = max(1, int(action_horizon))
        self.m1_history_frames = max(0, int(m1_history_frames))
        self.m1_current_frames = max(1, int(m1_current_frames))
        self.m1_history_window = max(0, int(m1_history_window))
        self.m1_turn_aware_sampling = bool(m1_turn_aware_sampling)
        self.m1_turn_frame_budget = max(0, int(m1_turn_frame_budget))
        self.enable_m1_recursive_split = bool(enable_m1_recursive_split)
        if self.enable_m1_recursive_split:
            # Keep the test-time re-segmentation setup fixed and reproducible.
            self.m1_history_frames = M1_RESEGMENT_HISTORY_FRAMES
            self.m1_current_frames = M1_RESEGMENT_CURRENT_FRAMES
        self.m2_recent_window = max(1, int(m2_recent_window))
        self.m2_num_frames = max(1, int(m2_num_frames))
        self.history_buffer_size = None
        if self.m1_history_window > 0 or self.enable_m1_recursive_split:
            self.history_buffer_size = max(
                100,
                self.m1_history_window + self.m1_current_frames,
                M1_RESEGMENT_CONTEXT_WINDOW,
                self.m2_recent_window,
            )
        self.jpeg_quality = max(1, min(int(jpeg_quality), 100))
        http_timeout_s = float(http_timeout_s)
        self.http_timeout_s = None if http_timeout_s <= 0 else http_timeout_s
        self.http_max_connections = max(1, int(http_max_connections))
        self.http_max_keepalive_connections = max(0, int(http_max_keepalive_connections))
        self.http_retry_count = int(http_retry_count)
        if self.http_retry_count < -1:
            raise ValueError("http_retry_count must be >= -1; use -1 for unlimited retries.")
        self.http_retry_backoff_s = max(0.0, float(http_retry_backoff_s))
        self.http_retry_max_backoff_s = max(self.http_retry_backoff_s, float(http_retry_max_backoff_s))
        self.http_force_connection_close = bool(http_force_connection_close)
        self.enable_image_data_url_cache = bool(enable_image_data_url_cache)
        self.image_data_url_cache_size = max(1, int(image_data_url_cache_size))
        self.stop_on_m1_stop_token = bool(stop_on_m1_stop_token)
        self.enable_align_turn_rule = bool(enable_align_turn_rule)
        self.filter_align_memory_for_m2 = bool(filter_align_memory_for_m2)
        self.strip_m1_recovering_prefix_for_m2 = bool(strip_m1_recovering_prefix_for_m2)
        self.m2_recover_max_consecutive = max(0, int(m2_recover_max_consecutive))
        self.m2_recover_cooldown = max(0, int(m2_recover_cooldown))
        if (self.m2_recover_max_consecutive > 0) != (self.m2_recover_cooldown > 0):
            raise ValueError(
                "m2_recover_max_consecutive and m2_recover_cooldown must both be > 0 to enable limiting, "
                "or both be 0 to disable it."
            )
        self.enable_m2_recover_limit = self.m2_recover_max_consecutive > 0
        if self.enable_m2_recover_limit:
            print(
                "M2 recover limit enabled: "
                f"max_consecutive={self.m2_recover_max_consecutive}, cooldown={self.m2_recover_cooldown}"
            )
        if self.enable_m1_recursive_split:
            print(
                "M1 one-shot re-segmentation enabled: "
                f"char_limit={M1_RESEGMENT_CHAR_LIMIT}, context_window={M1_RESEGMENT_CONTEXT_WINDOW}, "
                f"images={M1_RESEGMENT_HISTORY_FRAMES}+{M1_RESEGMENT_CURRENT_FRAMES}"
            )

        self.m1_server_url = m1_server_url.rstrip("/") if m1_server_url else None
        self.m1_server_model = m1_server_model
        self.m2_server_url = m2_server_url.rstrip("/") if m2_server_url else None
        self.m2_server_model = m2_server_model

        self._http_clients = {}
        self._image_data_url_cache = OrderedDict()

        if self.m1_server_url:
            print(f"M1: vllm server mode -> {self.m1_server_url} (model={self.m1_server_model})")
            self.engine1 = None
            self._http_clients[self.m1_server_url] = self._build_http_client()
        else:
            print(f"Loading Module 1 from: {model1_path}")
            self.engine1 = TransformersEngine(
                model=model1_path,
                model_type=model_type,
                torch_dtype=torch_dtype,
                device_map={"": 0} if device_map == "auto" else device_map,
            )

        self.engine2 = None
        if not self.use_single_model:
            if self.m2_server_url:
                print(f"M2: vllm server mode -> {self.m2_server_url} (model={self.m2_server_model})")
                if self.m2_server_url not in self._http_clients:
                    self._http_clients[self.m2_server_url] = self._build_http_client()
            else:
                print(f"Loading Module 2 from: {model2_path}")
                self.engine2 = TransformersEngine(
                    model=model2_path,
                    model_type=model_type,
                    torch_dtype=torch_dtype,
                    device_map="auto",
                )
        else:
            if model2_path:
                print("Single-model mode enabled: model2_path is ignored.")
            if self.m2_server_url:
                print("Single-model mode enabled: m2_server_url is ignored.")

        self.request_config_m1 = RequestConfig(max_tokens=m1_max_tokens, temperature=0.0)
        self.request_config_m2 = RequestConfig(max_tokens=m2_max_tokens, temperature=0.0)

        self.action_mapping = {
            "move forward 25 cm": 1,
            "turn left 15 degrees": 2,
            "turn right 15 degrees": 3,
            "stop": 0,
        }

        self.rgb_history = []
        self.rgb_pil_history = []
        self.rgb_uid_history = []
        self._next_frame_uid = 0
        self._action_to_frame_uid = {}
        self.topdown_map_list = []
        self.pending_action_list = []
        self.episode_id = None
        self.output_video_id = None

        self.last_reasoning = ""
        self.last_action_str = ""
        self.current_m1_answer = ""
        self.current_m2_sub_instruction = ""
        self.just_predicted = False
        self.align_phase_active = True
        self.m2_history_start_uid = None
        self.m2_recover_consecutive = 0
        self.m2_recover_cooldown_remaining = 0

    def _build_http_client(self):
        import httpx

        return httpx.Client(
            timeout=httpx.Timeout(self.http_timeout_s),
            headers={"Connection": "close"} if self.http_force_connection_close else None,
            limits=httpx.Limits(
                max_connections=self.http_max_connections,
                max_keepalive_connections=self.http_max_keepalive_connections,
            ),
        )

    def _reset_http_client(self, server_url):
        client = self._http_clients.get(server_url)
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
        self._http_clients[server_url] = self._build_http_client()

    def close(self):
        for client in self._http_clients.values():
            try:
                client.close()
            except Exception:
                pass

    def __del__(self):
        self.close()

    def _save_video_segment(self, video_id=None):
        if not self.require_map or len(self.topdown_map_list) == 0:
            return
        video_id = video_id or self.output_video_id or self.episode_id or 'unknown'
        os.makedirs(self.result_path, exist_ok=True)
        output_video_path = os.path.join(self.result_path, f"{video_id}.mp4")
        try:
            import imageio
            imageio.mimsave(output_video_path, self.topdown_map_list, fps=10, quality=9, macro_block_size=None)
            print(f"Successfully saved video to: {output_video_path}")
        except Exception as e:
            print(f"Failed to save video: {e}")

    def flush_video(self, video_id=None):
        self._save_video_segment(video_id=video_id)
        self.topdown_map_list = []

    def reset(self):
        self._save_video_segment()
        self.topdown_map_list = []

        self.rgb_history = []
        self.rgb_pil_history = []
        self.rgb_uid_history = []
        self._next_frame_uid = 0
        self._action_to_frame_uid = {}
        self._image_data_url_cache.clear()
        self.topdown_map_list = []
        self.pending_action_list = []
        self.last_reasoning = ""
        self.last_action_str = ""
        self.current_m1_answer = ""
        self.just_predicted = False
        self.align_phase_active = True
        self.m2_history_start_uid = None
        self.m2_recover_consecutive = 0
        self.m2_recover_cooldown_remaining = 0

    def _denser_at_end_sampling(self, pool_size, num_to_select, power=1.5):
        return _denser_at_end_sampling(pool_size, num_to_select, power=power)

    def _uniform_sparse_sampling(self, pool_size, num_to_select):
        return _uniform_sparse_sampling(pool_size, num_to_select)

    def _is_turn_result_frame(self, frame_index):
        if frame_index <= 0 or frame_index >= len(self.rgb_uid_history):
            return False
        frame_uid = self.rgb_uid_history[frame_index]
        return self._action_to_frame_uid.get(frame_uid) in {2, 3}

    def _motion_aware_history_sampling(self, history_pool):
        if not history_pool or self.m1_history_frames <= 0:
            return []

        candidate_pool = history_pool
        if self.m1_history_window > 0:
            candidate_pool = candidate_pool[-self.m1_history_window:]

        num_to_select = min(self.m1_history_frames, len(candidate_pool))
        if len(candidate_pool) <= num_to_select:
            return list(candidate_pool)

        uniform_rel = self._denser_at_end_sampling(len(candidate_pool), num_to_select, power=1.5)
        selected = {candidate_pool[i] for i in uniform_rel}

        if not self.m1_turn_aware_sampling or self.m1_turn_frame_budget <= 0:
            return sorted(selected)

        candidate_set = set(candidate_pool)
        turn_candidates = []
        for idx in candidate_pool:
            if not self._is_turn_result_frame(idx):
                continue
            for neighbor_idx in (idx - 1, idx, idx + 1):
                if neighbor_idx in candidate_set and neighbor_idx not in turn_candidates:
                    turn_candidates.append(neighbor_idx)
        turn_candidates = sorted(turn_candidates)
        if not turn_candidates:
            return sorted(selected)

        turn_budget = min(
            self.m1_turn_frame_budget,
            max(0, num_to_select // 4),
            max(0, num_to_select - 2),
            len(turn_candidates),
        )
        if turn_budget <= 0:
            return sorted(selected)

        turn_rel = self._uniform_sparse_sampling(len(turn_candidates), turn_budget)
        turn_selected = {turn_candidates[i] for i in turn_rel}
        selected.update(turn_selected)

        protected = {candidate_pool[0], candidate_pool[-1]}
        while len(selected) > num_to_select:
            removable = [idx for idx in selected if idx not in turn_selected and idx not in protected]
            if not removable:
                removable = [idx for idx in selected if idx not in turn_selected]
            if not removable:
                break
            selected.remove(min(removable, key=lambda idx: min(abs(idx - t) for t in turn_selected)))

        return sorted(selected)[:num_to_select]

    def _record_action_for_next_frame(self, action):
        try:
            action = int(action)
        except Exception:
            return
        self._action_to_frame_uid[self._next_frame_uid] = action

    def _pop_next_action(self):
        action = self.pending_action_list.pop(0)
        self._record_action_for_next_frame(action)
        return {"action": action}

    def get_images_for_modules(self, mode="module1"):
        if not self.rgb_history:
            return [], 0, 0, []

        indices = list(range(len(self.rgb_history)))

        if mode == "module1" or self.use_single_model:
            if len(indices) >= self.m1_current_frames:
                current_indices = indices[-self.m1_current_frames:]
                if self.enable_m1_recursive_split:
                    # Use only the newest window for the complete M1 visual
                    # context, including both history and current views.
                    history_start = max(0, len(indices) - M1_RESEGMENT_CONTEXT_WINDOW)
                    history_end = len(indices) - self.m1_current_frames
                    history_pool = indices[history_start:history_end]
                else:
                    # 2-frame gap between history and current
                    history_end = max(0, len(indices) - self.m1_current_frames - 2)
                    history_pool = indices[:history_end]
            else:
                current_indices = indices
                history_pool = []

            history_indices = self._motion_aware_history_sampling(history_pool)

            final_indices = history_indices + current_indices
            num_h, num_c = len(history_indices), len(current_indices)
        else:
            if self.m2_history_start_uid is not None:
                indices = [i for i in indices if self.rgb_uid_history[i] >= self.m2_history_start_uid]
                if not indices:
                    indices = [len(self.rgb_history) - 1]
            recent_window = indices[-self.m2_recent_window:]
            selected_indices = self._denser_at_end_sampling(len(recent_window), self.m2_num_frames, power=1.5)
            final_indices = [recent_window[i] for i in selected_indices]
            num_h, num_c = 0, len(final_indices)

        final_frames = [self.rgb_pil_history[i] for i in final_indices]
        frame_keys = [self.rgb_uid_history[i] for i in final_indices]
        return final_frames, num_h, num_c, frame_keys

    def predict_swift(self, engine, query_text, images, request_config):
        messages = [
            {"role": "system", "content": "You are an intelligent navigation robot."},
            {"role": "user", "content": query_text},
        ]
        resp_list = engine.infer([InferRequest(messages=messages, images=images)], request_config=request_config)
        return resp_list[0].choices[0].message.content.strip()

    def _get_cached_image_data_url(self, image, frame_key=None):
        cache_key = None if frame_key is None else int(frame_key)
        if self.enable_image_data_url_cache and cache_key is not None:
            cached = self._image_data_url_cache.get(cache_key)
            if cached is not None:
                self._image_data_url_cache.move_to_end(cache_key)
                return cached

        buf = BytesIO()
        image.save(
            buf,
            format="JPEG",
            quality=self.jpeg_quality,

            # Use a fixed JPEG subsampling mode for consistent serialization.
            subsampling=2,
            optimize=False,
            progressive=False,
        )
        data_url = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"

        if self.enable_image_data_url_cache and cache_key is not None:
            self._image_data_url_cache[cache_key] = data_url
            self._image_data_url_cache.move_to_end(cache_key)
            while len(self._image_data_url_cache) > self.image_data_url_cache_size:
                self._image_data_url_cache.popitem(last=False)
        return data_url

    def predict_openai(self, server_url, model_name, query_text, images, image_keys, max_tokens, temperature):
        import httpx

        content = []
        image_keys = image_keys or [None] * len(images)
        for img, frame_key in zip(images, image_keys):
            content.append({
                "type": "image_url",
                "image_url": {"url": self._get_cached_image_data_url(img, frame_key=frame_key)},
            })
        clean_text = re.sub(r'<image>', '', query_text).strip()
        content.append({"type": "text", "text": clean_text})

        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are an intelligent navigation robot."},
                {"role": "user", "content": content},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        url = f"{server_url}/v1/chat/completions"
        retryable_status_codes = {408, 409, 425, 429, 500, 502, 503, 504}
        total_attempts = None if self.http_retry_count == -1 else self.http_retry_count + 1
        attempt = 0
        while total_attempts is None or attempt < total_attempts:
            attempt += 1
            attempt_label = f"{attempt}/{'infinite' if total_attempts is None else total_attempts}"
            client = self._http_clients[server_url]
            try:
                resp = client.post(url, json=payload)
                if resp.status_code >= 400:
                    body = resp.text
                    if len(body) > 2000:
                        body = body[:2000] + "...(truncated)"
                    if resp.status_code in retryable_status_codes and (
                        total_attempts is None or attempt < total_attempts
                    ):
                        delay = min(self.http_retry_backoff_s * (2 ** (attempt - 1)), self.http_retry_max_backoff_s)
                        print(
                            f"Retryable vLLM response from {url} "
                            f"(status={resp.status_code}, attempt={attempt_label}). "
                            f"Retrying in {delay:.1f}s."
                        )
                        self._reset_http_client(server_url)
                        if delay > 0:
                            time.sleep(delay)
                        continue
                    raise RuntimeError(
                        f"vLLM request failed: status={resp.status_code}, url={url}, body={body}"
                    )
                return resp.json()["choices"][0]["message"]["content"].strip()
            except (httpx.TimeoutException, httpx.NetworkError, httpx.ProtocolError) as exc:
                if total_attempts is not None and attempt >= total_attempts:
                    raise RuntimeError(
                        f"vLLM request failed after {total_attempts} attempts: url={url}, error={exc}"
                    ) from exc
                delay = min(self.http_retry_backoff_s * (2 ** (attempt - 1)), self.http_retry_max_backoff_s)
                print(
                    f"Transient vLLM transport error from {url} "
                    f"(attempt={attempt_label}): {exc}. Retrying in {delay:.1f}s."
                )
                self._reset_http_client(server_url)
                if delay > 0:
                    time.sleep(delay)

        raise RuntimeError(f"vLLM request loop exited unexpectedly: url={url}")

    def parse_action_string(self, action_str):
        actions = []
        parts = [p.strip().lower() for p in re.split(r'[,，\n]', action_str)]
        for p in parts:
            if not p:
                continue
            matched = False
            for key, val in self.action_mapping.items():
                if key in p:
                    actions.append(val)
                    matched = True
                    break
            if not matched:
                num_match = re.search(r'\b(0|1|2|3|4)\b', p)
                if num_match:
                    act_num = int(num_match.group(1))
                    actions.append(0 if act_num == 4 else act_num)

        return actions

    def _format_image_count_desc(self, num_total):
        return _format_image_count_desc(num_total)

    def _build_m1_desc(self, num_total, num_h, num_c):
        prefix = self._format_image_count_desc(num_total)
        if num_h > 0:
            history_noun = "image" if num_h == 1 else "images"
            history_verb = "is" if num_h == 1 else "are"
            current_noun = "image" if num_c == 1 else "images"
            current_verb = "is" if num_c == 1 else "are"
            return (
                f"{prefix} The first {num_h} {history_noun} {history_verb} the History trajectory, "
                f"and the last {num_c} {current_noun} {current_verb} the Current view."
            )

        if num_total == 1:
            return f"{prefix} It is the Current view."
        return f"{prefix} All of them are the Current view."

    def _build_m1_prompt(self, global_inst, num_total, num_h, num_c):
        return (
            f"{'<image>' * num_total}\n"
            f"{self._build_m1_desc(num_total, num_h, num_c)}\n"
            f"Global Instruction: {global_inst}\n"
            f"Task: Analyze the history and current view to determine the current progress within the global "
            f"instruction. Provide a structured report with the following format: <think> Current Instruction: "
            f"<instruction> | Status: <Executing/Completed> | Next Instruction: <instruction> or None </think>\n"
            f"<answer> Next Instruction to Execute </answer>"
        )

    def _build_m2_prompt(self, global_inst, sub_inst, num_total):
        return build_m2_prompt(global_inst, sub_inst, num_total)

    def _format_m2_sub_instruction(self, sub_inst):
        sub_inst = sub_inst or ""
        if self.strip_m1_recovering_prefix_for_m2:
            return M1_RECOVERING_PREFIX_RE.sub('', sub_inst, count=1).strip()
        return sub_inst

    def _apply_m2_recover_limit(self, sub_inst):
        sub_inst = sub_inst or ""
        is_recovering = bool(M1_RECOVERING_PREFIX_RE.search(sub_inst))
        if not self.enable_m2_recover_limit:
            return sub_inst

        if self.m2_recover_cooldown_remaining > 0:
            self.m2_recover_cooldown_remaining -= 1
            self.m2_recover_consecutive = 0
            if is_recovering:
                return M1_RECOVERING_PREFIX_RE.sub('', sub_inst, count=1).strip()
            return sub_inst

        if not is_recovering:
            self.m2_recover_consecutive = 0
            return sub_inst

        self.m2_recover_consecutive += 1
        if self.m2_recover_consecutive >= self.m2_recover_max_consecutive:
            self.m2_recover_cooldown_remaining = self.m2_recover_cooldown
        return sub_inst

    def _build_single_model_prompt(self, global_inst, num_total, num_h, num_c):
        desc = f"Above are {num_total} images. "
        if num_h > 0:
            desc += (
                f"The first {num_h} images are the History trajectory, and the last {num_c} images are the "
                f"Current view."
            )
        else:
            desc += "All of them are the Current view."

        return (
            f"{'<image>' * num_total}\n"
            f"{desc}\n"
            f"Global Instruction: {global_inst}\n"
            f"Task: Provide the next 3 actions to execute based on the global instruction and visual observations. "
            f"Available actions: 1) Move forward 25 cm, 2) Turn left 15 degrees, 3) Turn right 15 degrees, 4) stop. "
            f"Output stop only when the entire task is completed."
        )

    def _prepare_frame_for_video(self, frame):
        if frame is None:
            return frame
        if frame.ndim == 2:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if frame.ndim == 3 and frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        if frame.ndim == 3 and frame.shape[2] == 3:
            return frame
        raise ValueError(f"Unsupported frame shape for video rendering: {frame.shape}")

    def act(self, observations, info):
        self.episode_id = info.get("episode_id", "unknown")
        rgb = self._prepare_frame_for_video(observations["rgb"])

        if self.require_map:
            target_h, target_w = rgb.shape[:2]
            map_key = "top_down_map_vlnce" if "top_down_map_vlnce" in info else "top_down_map"
            if map_key in info and info[map_key] is not None:
                map_data = info[map_key]
                if isinstance(map_data, dict):
                    top_down_map = maps.colorize_topdown_map(
                        map_data["map"], map_data.get("fog_of_war_mask")
                    )
                else:
                    top_down_map = maps.colorize_topdown_map(map_data)

                top_down_map = maps.colorize_draw_agent_and_fit_to_height(map_data, target_h)

                h, w = top_down_map.shape[:2]
                scale = target_h / h
                new_h, new_w = target_h, int(w * scale)

                if new_w > target_w:
                    scale = target_w / new_w
                    new_w = target_w
                    new_h = int(new_h * scale)

                top_down_map = cv2.resize(top_down_map, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

                pad_v = (target_h - new_h) // 2
                pad_h = (target_w - new_w) // 2
                top_down_map = cv2.copyMakeBorder(
                    top_down_map,
                    pad_v, target_h - new_h - pad_v,
                    pad_h, target_w - new_w - pad_h,
                    cv2.BORDER_CONSTANT, value=[0, 0, 0]
                )
                combined_frame = np.concatenate((rgb, top_down_map), axis=1)
            else:
                combined_frame = np.concatenate((rgb, np.zeros_like(rgb)), axis=1)

            if self.use_single_model:
                if self.last_action_str:
                    text = f"Actions: {self.last_action_str}"
                    cv2.putText(combined_frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(combined_frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                if self.current_m1_answer:
                    text = f"Sub-Inst: {self.current_m1_answer}"
                    cv2.putText(combined_frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(combined_frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            self.topdown_map_list.append(combined_frame)

        self.rgb_history.append(rgb)
        self.rgb_pil_history.append(Image.fromarray(rgb).convert('RGB').resize(self.image_size))
        self.rgb_uid_history.append(self._next_frame_uid)
        self._next_frame_uid += 1
        if self.history_buffer_size is not None:
            if len(self.rgb_history) > self.history_buffer_size:
                self.rgb_history.pop(0)
            if len(self.rgb_pil_history) > self.history_buffer_size:
                self.rgb_pil_history.pop(0)
            if len(self.rgb_uid_history) > self.history_buffer_size:
                self.rgb_uid_history.pop(0)
            if self.rgb_uid_history:
                min_uid = self.rgb_uid_history[0]
                self._action_to_frame_uid = {
                    uid: action for uid, action in self._action_to_frame_uid.items() if uid >= min_uid
                }

        global_inst = observations["instruction"]["text"]
        self.just_predicted = False

        if len(self.pending_action_list) == 0:
            if self.use_single_model:
                images1, num_h1, num_c1, image_keys1 = self.get_images_for_modules(mode="module1")
                num_total1 = len(images1)
                q1 = self._build_single_model_prompt(global_inst, num_total1, num_h1, num_c1)

                if self.m1_server_url:
                    res1 = self.predict_openai(
                        self.m1_server_url,
                        self.m1_server_model,
                        q1,
                        images1,
                        image_keys1,
                        self.request_config_m1.max_tokens,
                        self.request_config_m1.temperature,
                    )
                else:
                    res1 = self.predict_swift(self.engine1, q1, images1, self.request_config_m1)

                self.last_action_str = res1
                self.current_m1_answer = ""
                self.current_m2_sub_instruction = ""
                self.pending_action_list = self.parse_action_string(res1)[:self.action_horizon]
                if not self.pending_action_list:
                    self.pending_action_list = [0]
                self.just_predicted = True
            else:
                images1, num_h1, num_c1, image_keys1 = self.get_images_for_modules(mode="module1")
                num_total1 = len(images1)

                m1_global_inst = global_inst
                q1 = self._build_m1_prompt(m1_global_inst, num_total1, num_h1, num_c1)
                if self.m1_server_url:
                    res1 = self.predict_openai(
                        self.m1_server_url,
                        self.m1_server_model,
                        q1,
                        images1,
                        image_keys1,
                        self.request_config_m1.max_tokens,
                        self.request_config_m1.temperature,
                    )
                else:
                    res1 = self.predict_swift(self.engine1, q1, images1, self.request_config_m1)
                sub_inst = _extract_m1_sub_instruction(res1)
                m1_responses = [res1]

                # At most one re-segmentation pass. A Recovering answer is a
                # recovery-control signal rather than a normal sub-instruction,
                # so it must be passed through to M2 without a second M1 call.
                should_resegment = (
                    self.enable_m1_recursive_split
                    and not M1_RECOVERING_PREFIX_RE.search(sub_inst)
                    and "[STOP]" not in sub_inst.upper()
                    and _instruction_char_count(sub_inst) > M1_RESEGMENT_CHAR_LIMIT
                )
                if should_resegment:
                    m1_global_inst = sub_inst
                    q1 = self._build_m1_prompt(m1_global_inst, num_total1, num_h1, num_c1)
                    if self.m1_server_url:
                        res1 = self.predict_openai(
                            self.m1_server_url,
                            self.m1_server_model,
                            q1,
                            images1,
                            image_keys1,
                            self.request_config_m1.max_tokens,
                            self.request_config_m1.temperature,
                        )
                    else:
                        res1 = self.predict_swift(self.engine1, q1, images1, self.request_config_m1)
                    sub_inst = _extract_m1_sub_instruction(res1)
                    m1_responses.append(res1)

                if len(m1_responses) == 2:
                    self.last_reasoning = (
                        "[M1 initial segmentation]\n"
                        f"{m1_responses[0]}\n\n"
                        "[M1 one-shot re-segmentation]\n"
                        f"{m1_responses[1]}"
                    )
                else:
                    self.last_reasoning = m1_responses[0]

                self.current_m1_answer = sub_inst

                if self.stop_on_m1_stop_token and "[STOP]" in sub_inst.upper():
                    self.current_m2_sub_instruction = ""
                    self.pending_action_list = [0]
                    self.just_predicted = True
                    return self._pop_next_action()

                align_turn_action = None
                if re.search(r'\balign\s+left\b', sub_inst, re.I):
                    align_turn_action = ("Turn left 15 degrees", 2)
                elif re.search(r'\balign\s+right\b', sub_inst, re.I):
                    align_turn_action = ("Turn right 15 degrees", 3)
                is_align_answer = align_turn_action is not None or "align" in sub_inst.lower()
                if self.enable_align_turn_rule and align_turn_action is not None:
                    self.current_m2_sub_instruction = ""
                    action_text, action_id = align_turn_action
                    self.pending_action_list = [action_id] * self.action_horizon
                    self.last_action_str = ", ".join([action_text] * len(self.pending_action_list))
                    self.just_predicted = True
                    return self._pop_next_action()
                if self.align_phase_active and not is_align_answer:
                    self.align_phase_active = False
                    if self.filter_align_memory_for_m2 and self.rgb_uid_history:
                        self.m2_history_start_uid = self.rgb_uid_history[-1]

                images2, num_h2, num_c2, image_keys2 = self.get_images_for_modules(mode="module2")
                num_total2 = len(images2)

                m2_sub_inst = self._format_m2_sub_instruction(self._apply_m2_recover_limit(sub_inst))
                self.current_m2_sub_instruction = m2_sub_inst
                q2 = self._build_m2_prompt(global_inst, m2_sub_inst, num_total2)
                if self.m2_server_url:
                    res2 = self.predict_openai(
                        self.m2_server_url,
                        self.m2_server_model,
                        q2,
                        images2,
                        image_keys2,
                        self.request_config_m2.max_tokens,
                        self.request_config_m2.temperature,
                    )
                else:
                    res2 = self.predict_swift(self.engine2, q2, images2, self.request_config_m2)
                self.last_action_str = res2
                self.pending_action_list = self.parse_action_string(res2)[:self.action_horizon]
                if not self.pending_action_list:
                    self.pending_action_list = [0]
                self.just_predicted = True

        return self._pop_next_action()
