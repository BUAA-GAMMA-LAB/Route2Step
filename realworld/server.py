import argparse
import base64
import io
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import requests
from flask import Flask, jsonify, request
from PIL import Image, ImageDraw, ImageFont


SYSTEM_PROMPT = 'You are an intelligent navigation robot.'
M1_MAX_HISTORY_IMAGES = 9
M1_MAX_CURRENT_IMAGES = 3
M2_RECENT_WINDOW = 40
M2_NUM_IMAGES = 8
M2_DENSER_AT_END_POWER = 1.5
DEFAULT_HISTORY_LIMIT = 1000


def denser_at_end_sampling(pool_size: int, num_to_select: int, power: float = M2_DENSER_AT_END_POWER) -> List[int]:
    if num_to_select <= 0:
        return []
    if num_to_select >= pool_size:
        return list(range(pool_size))
    positions = np.linspace(0, 1, num_to_select)
    transformed_positions = 1 - (1 - positions) ** power
    sampled_indices = np.round(transformed_positions * (pool_size - 1)).astype(int)
    unique_indices = sorted(list(set(sampled_indices.tolist())))
    while len(unique_indices) < num_to_select:
        available = sorted(list(set(range(pool_size)) - set(unique_indices)))
        if not available:
            break
        unique_indices.append(available[-1])
        unique_indices = sorted(unique_indices)
    return unique_indices


def uniform_sparse_sampling(pool_size: int, num_to_select: int) -> List[int]:
    if num_to_select <= 0:
        return []
    if num_to_select >= pool_size:
        return list(range(pool_size))
    sampled_indices = np.linspace(0, pool_size - 1, num_to_select)
    return np.round(sampled_indices).astype(int).tolist()


def get_images_for_modules_from_pil_history(
    rgb_pil_history: List[Image.Image],
    mode: str = 'module1',
) -> Tuple[List[Image.Image], int, int]:
    if not rgb_pil_history:
        return [], 0, 0

    indices = list(range(len(rgb_pil_history)))

    if mode == 'module1':
        if len(indices) >= M1_MAX_CURRENT_IMAGES:
            current_indices = indices[-M1_MAX_CURRENT_IMAGES:]
            history_pool = indices[:-M1_MAX_CURRENT_IMAGES]
        else:
            current_indices = indices
            history_pool = []

        if len(history_pool) > M1_MAX_HISTORY_IMAGES:
            selected_h = uniform_sparse_sampling(len(history_pool), M1_MAX_HISTORY_IMAGES)
            history_indices = [history_pool[i] for i in selected_h]
        else:
            history_indices = history_pool

        final_indices = history_indices + current_indices
        num_h, num_c = len(history_indices), len(current_indices)
    else:
        recent_window = indices[-M2_RECENT_WINDOW:]
        selected_indices = denser_at_end_sampling(
            len(recent_window),
            M2_NUM_IMAGES,
            power=M2_DENSER_AT_END_POWER,
        )
        final_indices = [recent_window[i] for i in selected_indices]
        num_h, num_c = 0, len(final_indices)

    final_frames = [rgb_pil_history[i] for i in final_indices]
    return final_frames, num_h, num_c


def build_m1_prompt(global_instruction: str, num_history: int, num_current: int) -> str:
    num_images = num_history + num_current
    image_tags = '<image>' * num_images
    if num_history > 0:
        frame_desc = (
            f'Above are {num_images} images. The first {num_history} images are the History trajectory, '
            f'and the last {num_current} images are the Current view.'
        )
    else:
        frame_desc = f'Above are {num_images} images. All of them are the Current view.'

    return (
        f'{image_tags}\n'
        f'{frame_desc}\n'
        f'Global Instruction: {global_instruction}\n'
        f'Task: Analyze the history and current view to determine the current progress within the global instruction. '
        f'Provide a structured report with the following format: <think> Current Instruction: <instruction> | '
        f'Status: <Executing/Completed> | Next Instruction: <instruction> or None </think>\n'
        f'<answer> Next Instruction to Execute </answer>'
    )


def build_m2_prompt(global_instruction: str, current_sub_instruction: str, num_images: int) -> str:
    return (
        f'{"<image>" * num_images}\n'
        f'Above are {num_images} images. They are ordered from earlier to more recent views.\n'
        f'Global Instruction: {global_instruction}\n'
        f'Current Sub-instruction: {current_sub_instruction}\n'
        f'Task: Provide the next 3 actions to execute for the current sub-instruction. '
        f'Available actions: 1) Move forward 25 cm, 2) Turn left 15 degrees, 3) Turn right 15 degrees, 4) stop. '
        f'Output stop only when the entire task is completed.'
    )


def extract_sub_instruction(raw_text: str) -> str:
    answer_match = re.search(r'<answer>(.*?)</answer>', raw_text, re.S | re.I)
    if answer_match:
        return answer_match.group(1).strip()

    sub_inst = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.S | re.I).strip()
    sub_inst = re.sub(
        r'^(Analyze|Reasoning|Instruction|Next Instruction to Execute):\s*',
        '',
        sub_inst,
        flags=re.I,
    )
    return sub_inst.strip()


def parse_action_string(action_str: str, action_mapping: Dict[str, int]) -> List[int]:
    actions = []
    parts = [p.strip().lower() for p in re.split(r'[,，\n]', action_str)]
    for part in parts:
        if not part:
            continue
        matched = False
        for key, val in action_mapping.items():
            if key in part:
                actions.append(val)
                matched = True
                break
        if not matched:
            num_match = re.search(r'\b(0|1|2|3|4)\b', part)
            if num_match:
                act_num = int(num_match.group(1))
                actions.append(0 if act_num == 4 else act_num)
    return actions


def format_action_sequence(actions: List[int]) -> str:
    label_map = {
        0: 'STOP',
        1: 'FORWARD',
        2: 'LEFT',
        3: 'RIGHT',
    }
    if not actions:
        return 'STOP'
    return ' | '.join(label_map.get(action, str(action)) for action in actions)


@dataclass
class SessionState:
    session_id: str
    run_id: str = ''
    instruction: str = ''
    rgb_pil_history: List[Image.Image] = field(default_factory=list)
    request_count: int = 0
    frame_count: int = 0
    terminated: bool = False
    last_reasoning: str = ''
    last_sub_instruction: str = ''
    last_m2_sub_instruction: str = ''
    last_action_text: str = ''
    async_m2_sub_instruction: str = ''
    total_latency_ms: float = 0.0
    start_time: float = field(default_factory=time.time)
    output_dir: str = ''


class RealWorldDualVLNServer:
    def __init__(
        self,
        m1_server_url: str,
        m1_server_model: str,
        m2_server_url: str,
        m2_server_model: str,
        output_dir: str,
        host: str,
        port: int,
        image_size: Tuple[int, int],
        action_horizon: int,
        request_timeout: float,
        history_limit: int,
        save_annotated: bool,
        inference_mode: str,
    ):
        self.m1_server_url = m1_server_url.rstrip('/')
        self.m1_server_model = m1_server_model
        self.m2_server_url = m2_server_url.rstrip('/')
        self.m2_server_model = m2_server_model
        self.output_dir = output_dir
        self.host = host
        self.port = port
        self.image_size = image_size
        self.action_horizon = max(1, int(action_horizon))
        self.request_timeout = request_timeout
        self.history_limit = max(1, int(history_limit))
        self.save_annotated = save_annotated
        self.inference_mode = inference_mode
        self.sessions: Dict[str, SessionState] = {}
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='realworld_dual_vln')
        self.action_mapping = {
            'move forward 25 cm': 1,
            'turn left 15 degrees': 2,
            'turn right 15 degrees': 3,
            'stop': 0,
        }
        os.makedirs(self.output_dir, exist_ok=True)

    def _sanitize_session_id(self, session_id: str) -> str:
        return re.sub(r'[^A-Za-z0-9_.-]+', '_', session_id) or 'default'

    def _new_output_dir(self, session_id: str) -> str:
        ts = datetime.now().strftime('%Y%m%d-%H%M%S')
        session_name = self._sanitize_session_id(session_id)
        path = os.path.join(self.output_dir, f'{session_name}_{ts}')
        os.makedirs(path, exist_ok=True)
        return path

    def _create_session(self, session_id: str, instruction: str = '', run_id: str = '') -> SessionState:
        return SessionState(
            session_id=session_id,
            run_id=run_id,
            instruction=instruction.strip(),
            output_dir=self._new_output_dir(session_id),
        )

    def _get_session(self, session_id: str, reset: bool, instruction: str, run_id: str = '') -> SessionState:
        session = self.sessions.get(session_id)
        same_client_run = bool(session and run_id and session.run_id == run_id)
        run_id_changed = bool(session and run_id and session.run_id and session.run_id != run_id)
        should_create = session is None or run_id_changed or (reset and not same_client_run)

        if should_create:
            self.sessions[session_id] = self._create_session(session_id, instruction, run_id=run_id)

        session = self.sessions[session_id]
        if run_id and not session.run_id:
            session.run_id = run_id
        if instruction:
            session.instruction = instruction.strip()
        return session

    def _decode_request_image(self, image_storage) -> Image.Image:
        image = Image.open(image_storage.stream).convert('RGB')
        return image.resize(self.image_size)

    def _pil_to_content_image(self, image: Image.Image) -> Dict[str, Dict[str, str]]:
        buf = io.BytesIO()
        image.save(buf, format='JPEG', quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{b64}'}}

    def _predict_openai(
        self,
        server_url: str,
        model_name: str,
        query_text: str,
        images: List[Image.Image],
        max_tokens: int,
        temperature: float,
    ) -> Tuple[str, float]:
        clean_text = re.sub(r'<image>', '', query_text).strip()
        content = [self._pil_to_content_image(img) for img in images]
        content.append({'type': 'text', 'text': clean_text})

        payload = {
            'model': model_name,
            'messages': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': content},
            ],
            'max_tokens': max_tokens,
            'temperature': temperature,
        }

        t0 = time.time()
        response = requests.post(
            f'{server_url}/v1/chat/completions',
            json=payload,
            timeout=self.request_timeout,
        )
        latency_ms = (time.time() - t0) * 1000.0
        if response.status_code >= 400:
            body = response.text
            if len(body) > 2000:
                body = body[:2000] + '...(truncated)'
            raise RuntimeError(
                f'vLLM request failed: status={response.status_code}, '
                f'url={server_url}/v1/chat/completions, body={body}'
            )
        response_json = response.json()
        output_text = response_json['choices'][0]['message']['content'].strip()
        return output_text, latency_ms

    def _is_stop_sub_instruction(self, sub_inst: str) -> bool:
        normalized_sub_inst = sub_inst.strip().lower()
        return '[stop]' in normalized_sub_inst or normalized_sub_inst in {'stop', 'none'}

    def _run_m1_inference(self, instruction: str, rgb_pil_history: List[Image.Image]) -> Tuple[str, str, float]:
        images1, num_h1, num_c1 = get_images_for_modules_from_pil_history(rgb_pil_history, mode='module1')
        q1 = build_m1_prompt(instruction, num_h1, num_c1)
        res1, latency_ms = self._predict_openai(
            self.m1_server_url,
            self.m1_server_model,
            q1,
            images1,
            max_tokens=256,
            temperature=0.0,
        )
        return res1, extract_sub_instruction(res1), latency_ms

    def _run_m2_inference(
        self,
        instruction: str,
        sub_inst: str,
        rgb_pil_history: List[Image.Image],
    ) -> Tuple[str, List[int], float]:
        images2, _, _ = get_images_for_modules_from_pil_history(rgb_pil_history, mode='module2')
        q2 = build_m2_prompt(instruction, sub_inst, len(images2))
        res2, latency_ms = self._predict_openai(
            self.m2_server_url,
            self.m2_server_model,
            q2,
            images2,
            max_tokens=64,
            temperature=0.0,
        )
        actions = parse_action_string(res2, self.action_mapping)[:self.action_horizon]
        if not actions:
            actions = [0]
        return res2, actions, latency_ms

    def _plan_actions_serial(self, session: SessionState) -> Tuple[List[int], Dict[str, float]]:
        timings = {
            'm1_latency_ms': 0.0,
            'm2_latency_ms': 0.0,
            'request_latency_ms': 0.0,
            'serial_equivalent_latency_ms': 0.0,
            'overlap_saved_ms': 0.0,
        }
        rgb_pil_history = list(session.rgb_pil_history)
        t0 = time.time()
        res1, sub_inst, timings['m1_latency_ms'] = self._run_m1_inference(session.instruction, rgb_pil_history)
        session.last_reasoning = res1
        session.last_sub_instruction = sub_inst
        session.last_m2_sub_instruction = sub_inst
        session.async_m2_sub_instruction = sub_inst

        if self._is_stop_sub_instruction(sub_inst):
            session.last_action_text = 'STOP'
            timings['request_latency_ms'] = (time.time() - t0) * 1000.0
            timings['serial_equivalent_latency_ms'] = timings['request_latency_ms']
            return [0], timings

        res2, actions, timings['m2_latency_ms'] = self._run_m2_inference(session.instruction, sub_inst, rgb_pil_history)
        session.last_action_text = res2
        timings['request_latency_ms'] = (time.time() - t0) * 1000.0
        timings['serial_equivalent_latency_ms'] = timings['m1_latency_ms'] + timings['m2_latency_ms']
        timings['overlap_saved_ms'] = max(0.0, timings['serial_equivalent_latency_ms'] - timings['request_latency_ms'])
        return actions, timings

    def _plan_actions_async_staggered(self, session: SessionState) -> Tuple[List[int], Dict[str, float]]:
        timings = {
            'm1_latency_ms': 0.0,
            'm2_latency_ms': 0.0,
            'request_latency_ms': 0.0,
            'serial_equivalent_latency_ms': 0.0,
            'overlap_saved_ms': 0.0,
        }
        rgb_pil_history = list(session.rgb_pil_history)
        m2_sub_inst = session.async_m2_sub_instruction or session.instruction
        t0 = time.time()

        future_m1 = self.executor.submit(self._run_m1_inference, session.instruction, rgb_pil_history)
        future_m2 = self.executor.submit(self._run_m2_inference, session.instruction, m2_sub_inst, rgb_pil_history)

        res1, next_sub_inst, timings['m1_latency_ms'] = future_m1.result()
        res2, actions, timings['m2_latency_ms'] = future_m2.result()
        timings['request_latency_ms'] = (time.time() - t0) * 1000.0
        timings['serial_equivalent_latency_ms'] = timings['m1_latency_ms'] + timings['m2_latency_ms']
        timings['overlap_saved_ms'] = max(0.0, timings['serial_equivalent_latency_ms'] - timings['request_latency_ms'])

        session.last_reasoning = res1
        session.last_sub_instruction = next_sub_inst
        session.last_m2_sub_instruction = m2_sub_inst
        session.last_action_text = res2
        session.async_m2_sub_instruction = next_sub_inst

        if self._is_stop_sub_instruction(next_sub_inst):
            session.last_action_text = 'STOP (M1 override)'
            return [0], timings
        return actions, timings

    def _plan_actions(self, session: SessionState) -> Tuple[List[int], Dict[str, float]]:
        if self.inference_mode == 'async_staggered':
            return self._plan_actions_async_staggered(session)
        return self._plan_actions_serial(session)

    def _annotate_image(
        self,
        session: SessionState,
        image: Image.Image,
        action_text: str,
        request_latency_ms: float,
        error_text: str = '',
    ) -> None:
        if not self.save_annotated:
            return

        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)

        padding = 8
        box_x = 10
        box_y = 10
        max_text_width = max(120, annotated.width - 2 * box_x - 2 * padding)

        def load_font(size: int):
            try:
                return ImageFont.truetype('DejaVuSansMono.ttf', size)
            except OSError:
                return ImageFont.load_default()

        def text_width(text: str, font) -> int:
            bbox = draw.textbbox((0, 0), text, font=font)
            return bbox[2] - bbox[0]

        def split_long_token(token: str, font, max_width: int) -> List[str]:
            pieces = []
            current = ''
            for char in token:
                candidate = current + char
                if current and text_width(candidate, font) > max_width:
                    pieces.append(current)
                    current = char
                else:
                    current = candidate
            if current:
                pieces.append(current)
            return pieces or ['']

        def wrap_labeled_line(label: str, value: str, font) -> List[str]:
            value = value or '-'
            prefix = f'{label:<13}: '
            continuation = ' ' * len(prefix)
            lines = []
            current = prefix
            current_limit = max_text_width

            for token in value.split():
                candidate = current + ('' if current.endswith(' ') else ' ') + token
                if text_width(candidate, font) <= current_limit:
                    current = candidate
                    continue

                if current.strip():
                    lines.append(current.rstrip())
                current = continuation

                token_candidate = current + token
                if text_width(token_candidate, font) <= current_limit:
                    current = token_candidate
                    continue

                token_limit = max(20, current_limit - text_width(continuation, font))
                pieces = split_long_token(token, font, token_limit)
                lines.extend((continuation + piece).rstrip() for piece in pieces[:-1])
                current = continuation + pieces[-1]

            if current.strip():
                lines.append(current.rstrip())
            return lines or [prefix.rstrip()]

        def build_lines(font) -> List[str]:
            lines = [
                f'Frame        : {session.request_count}',
                f'Runtime      : {time.time() - session.start_time:.2f} s',
            ]
            lines.extend(wrap_labeled_line('Sub-Inst', session.last_sub_instruction or '-', font))
            lines.extend(wrap_labeled_line('Actions', action_text, font))
            if error_text:
                lines.extend(wrap_labeled_line('Error', error_text[:300], font))
            return lines

        font = load_font(14)
        lines = build_lines(font)
        for font_size in range(13, 9, -1):
            sample_bbox = draw.textbbox((0, 0), 'Ag', font=font)
            line_height = max(font_size + 2, sample_bbox[3] - sample_bbox[1] + 4)
            box_height = len(lines) * line_height + 2 * padding
            if box_height <= annotated.height - 2 * box_y:
                break
            font = load_font(font_size)
            lines = build_lines(font)

        max_width = 0
        sample_bbox = draw.textbbox((0, 0), 'Ag', font=font)
        line_height = max(14, sample_bbox[3] - sample_bbox[1] + 4)
        for line in lines:
            max_width = max(max_width, text_width(line, font))

        box_width = max_width + 2 * padding
        box_height = len(lines) * line_height + 2 * padding
        draw.rectangle([box_x, box_y, box_x + box_width, box_y + box_height], fill='black')

        y = box_y + padding
        for line in lines:
            draw.text((box_x + padding, y), line, fill='white', font=font)
            y += line_height

        output_path = os.path.join(session.output_dir, f'rgb_{session.request_count:04d}_annotated.jpg')
        annotated.save(output_path, quality=90)

    def _append_jsonl(self, path: str, payload: Dict[str, object]) -> None:
        with open(path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(payload, ensure_ascii=False) + '\n')

    def _save_inference_log(
        self,
        session: SessionState,
        image: Image.Image,
        actions: List[int],
        action_text: str,
        timings: Dict[str, float],
        error_text: str = '',
    ) -> None:
        """Persist per-request inference details to inference.jsonl in the session output dir."""
        payload: Dict[str, object] = {
            'timestamp': time.time(),
            'request_count': session.request_count,
            'frame_count': session.frame_count,
            'instruction': session.instruction,
            'sub_instruction': session.last_sub_instruction,
            'raw_reasoning': session.last_reasoning,
            'raw_action_text': session.last_action_text,
            'm2_sub_instruction_used': session.last_m2_sub_instruction,
            'actions': actions,
            'action_text': action_text,
            'terminated': session.terminated,
            'inference_mode': self.inference_mode,
            'm1_latency_ms': round(timings.get('m1_latency_ms', 0.0), 1),
            'm2_latency_ms': round(timings.get('m2_latency_ms', 0.0), 1),
            'request_latency_ms': round(timings.get('request_latency_ms', 0.0), 1),
            'serial_equivalent_latency_ms': round(timings.get('serial_equivalent_latency_ms', 0.0), 1),
            'overlap_saved_ms': round(timings.get('overlap_saved_ms', 0.0), 1),
            'total_latency_ms': round(session.total_latency_ms, 1),
            'runtime_s': round(time.time() - session.start_time, 3),
        }
        if error_text:
            payload['error'] = error_text
        self._append_jsonl(os.path.join(session.output_dir, 'inference.jsonl'), payload)

    def _save_camera_frame(
        self,
        session: SessionState,
        image: Image.Image,
        raw_data: Dict[str, object],
    ) -> Dict[str, object]:
        session.frame_count += 1
        frames_dir = os.path.join(session.output_dir, 'frames')
        os.makedirs(frames_dir, exist_ok=True)

        output_name = f'rgb_{session.frame_count:06d}.jpg'
        output_path = os.path.join(frames_dir, output_name)
        image.save(output_path, quality=90)

        metadata = {
            'frame_index': session.frame_count,
            'client_frame_id': raw_data.get('frame_id'),
            'client_timestamp': raw_data.get('timestamp'),
            'server_timestamp': time.time(),
            'request_count': session.request_count,
            'path': os.path.relpath(output_path, session.output_dir),
        }
        self._append_jsonl(os.path.join(session.output_dir, 'frames.jsonl'), metadata)
        return metadata

    def _load_frame_history(self, session: SessionState) -> List[Image.Image]:
        frames_dir = os.path.join(session.output_dir, 'frames')
        if not os.path.isdir(frames_dir):
            return []

        frame_names = sorted(
            name for name in os.listdir(frames_dir)
            if name.lower().endswith(('.jpg', '.jpeg', '.png'))
        )
        frame_names = frame_names[-self.history_limit:]
        history = []
        for name in frame_names:
            path = os.path.join(frames_dir, name)
            history.append(Image.open(path).convert('RGB').resize(self.image_size))
        return history

    def _health_payload(self) -> Dict[str, object]:
        return {
            'status': 'ok',
            'host': self.host,
            'port': self.port,
            'm1_server_url': self.m1_server_url,
            'm1_server_model': self.m1_server_model,
            'm2_server_url': self.m2_server_url,
            'm2_server_model': self.m2_server_model,
            'inference_mode': self.inference_mode,
            'num_sessions': len(self.sessions),
        }

    def handle_eval_request(self, image_storage, raw_data: Dict[str, object]) -> Dict[str, object]:
        session_id = str(raw_data.get('session_id') or 'default')
        reset = bool(raw_data.get('reset', False))
        instruction = str(raw_data.get('instruction') or '').strip()
        run_id = str(raw_data.get('run_id') or '').strip()

        with self.lock:
            session = self._get_session(session_id, reset=reset, instruction=instruction, run_id=run_id)
            if not session.instruction:
                return {
                    'success': False,
                    'action': [0],
                    'error': 'instruction is required on the first request or after reset',
                    'session_id': session_id,
                    'terminated': True,
                }

            already_terminated = session.terminated
            request_count = session.request_count

            if already_terminated:
                action = [0]
                action_text = format_action_sequence(action)
                return {
                    'success': True,
                    'action': action,
                    'session_id': session_id,
                    'request_count': request_count,
                    'sub_instruction': session.last_sub_instruction,
                    'action_text': action_text,
                    'terminated': True,
                    'reset_applied': reset,
                    'image_accepted': False,
                }

            image = self._decode_request_image(image_storage)
            session.request_count += 1
            session.rgb_pil_history = self._load_frame_history(session)
            if not session.rgb_pil_history:
                session.rgb_pil_history.append(image)

            request_count = session.request_count

        try:
            actions, timings = self._plan_actions(session)
        except Exception as exc:
            error_text = str(exc)
            self._annotate_image(session, image, 'STOP', request_latency_ms=0.0, error_text=error_text)
            self._save_inference_log(session, image, [0], 'STOP', {}, error_text=error_text)
            return {
                'success': False,
                'action': [0],
                'error': error_text,
                'session_id': session_id,
                'request_count': request_count,
                'sub_instruction': session.last_sub_instruction,
                'terminated': True,
                'reset_applied': reset,
            }

        request_latency_ms = timings['request_latency_ms']
        session.total_latency_ms += request_latency_ms
        session.terminated = 0 in actions
        action_text = format_action_sequence(actions)
        self._annotate_image(session, image, action_text, request_latency_ms=request_latency_ms)
        self._save_inference_log(session, image, actions, action_text, timings)
        return {
            'success': True,
            'action': actions,
            'session_id': session_id,
            'request_count': request_count,
            'sub_instruction': session.last_sub_instruction,
            'raw_reasoning': session.last_reasoning,
            'raw_action_text': session.last_action_text,
            'action_text': action_text,
            'terminated': session.terminated,
            'reset_applied': reset,
            'inference_mode': self.inference_mode,
            'm2_sub_instruction_used': session.last_m2_sub_instruction,
            'm1_latency_ms': round(timings['m1_latency_ms'], 1),
            'm2_latency_ms': round(timings['m2_latency_ms'], 1),
            'serial_equivalent_latency_ms': round(timings['serial_equivalent_latency_ms'], 1),
            'overlap_saved_ms': round(timings['overlap_saved_ms'], 1),
            'total_latency_ms': round(request_latency_ms, 1),
        }

    def handle_frame_log_request(self, image_storage, raw_data: Dict[str, object]) -> Dict[str, object]:
        session_id = str(raw_data.get('session_id') or 'default')
        reset = bool(raw_data.get('reset', False))
        instruction = str(raw_data.get('instruction') or '').strip()
        run_id = str(raw_data.get('run_id') or '').strip()

        with self.lock:
            session = self._get_session(session_id, reset=reset, instruction=instruction, run_id=run_id)
            if session.terminated:
                return {
                    'success': True,
                    'session_id': session_id,
                    'frame_index': session.frame_count,
                    'client_frame_id': raw_data.get('frame_id'),
                    'output_dir': session.output_dir,
                    'terminated': True,
                    'image_accepted': False,
                }

            image = self._decode_request_image(image_storage)
            metadata = self._save_camera_frame(session, image, raw_data)
            return {
                'success': True,
                'session_id': session_id,
                'frame_index': metadata['frame_index'],
                'client_frame_id': metadata['client_frame_id'],
                'output_dir': session.output_dir,
                'terminated': False,
                'image_accepted': True,
            }


app = Flask(__name__)
SERVER = None


@app.get('/health')
def health():
    if SERVER is None:
        return jsonify({'status': 'not_ready'}), 503
    return jsonify(SERVER._health_payload())


@app.post('/eval_vln')
def eval_vln():
    if SERVER is None:
        return jsonify({'success': False, 'action': [0], 'error': 'server not initialized'}), 503

    image_file = request.files.get('image')
    if image_file is None:
        return jsonify({'success': False, 'action': [0], 'error': 'missing image file'}), 400

    raw_json = request.form.get('json', '{}')
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        return jsonify({'success': False, 'action': [0], 'error': f'invalid json payload: {exc}'}), 400

    result = SERVER.handle_eval_request(image_file, data)
    return jsonify(result)


@app.post('/log_frame')
def log_frame():
    if SERVER is None:
        return jsonify({'success': False, 'error': 'server not initialized'}), 503

    image_file = request.files.get('image')
    if image_file is None:
        return jsonify({'success': False, 'error': 'missing image file'}), 400

    raw_json = request.form.get('json', '{}')
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        return jsonify({'success': False, 'error': f'invalid json payload: {exc}'}), 400

    result = SERVER.handle_frame_log_request(image_file, data)
    return jsonify(result)


def parse_args():
    parser = argparse.ArgumentParser(description='Real-world VLN server backed by vLLM.')
    parser.add_argument('--host', type=str, required=True, help='Address/interface to bind the HTTP server.')
    parser.add_argument('--port', type=int, default=5801)
    parser.add_argument('--m1_server_url', type=str, default='http://localhost:8081')
    parser.add_argument('--m1_server_model', type=str, default='m1')
    parser.add_argument('--m2_server_url', type=str, default='http://localhost:8080')
    parser.add_argument('--m2_server_model', type=str, default='m2')
    parser.add_argument('--output_dir', type=str, default='realworld_runs')
    parser.add_argument('--image_width', type=int, default=640)
    parser.add_argument('--image_height', type=int, default=480)
    parser.add_argument('--action_horizon', type=int, default=3)
    parser.add_argument('--request_timeout', type=float, default=120.0)
    parser.add_argument('--history_limit', type=int, default=DEFAULT_HISTORY_LIMIT)
    parser.add_argument(
        '--inference_mode',
        type=str,
        default='serial',
        choices=['serial', 'async_staggered'],
        help='serial: current M1 -> current M2; async_staggered: current M1 || current M2(previous sub-instruction).',
    )
    parser.add_argument('--disable_annotated_output', action='store_true')
    return parser.parse_args()


def main():
    global SERVER
    args = parse_args()

    m2_server_url = args.m2_server_url or args.m1_server_url
    SERVER = RealWorldDualVLNServer(
        m1_server_url=args.m1_server_url,
        m1_server_model=args.m1_server_model,
        m2_server_url=m2_server_url,
        m2_server_model=args.m2_server_model,
        output_dir=args.output_dir,
        host=args.host,
        port=args.port,
        image_size=(args.image_width, args.image_height),
        action_horizon=args.action_horizon,
        request_timeout=args.request_timeout,
        history_limit=args.history_limit,
        save_annotated=not args.disable_annotated_output,
        inference_mode=args.inference_mode,
    )

    print('Real-world VLN server configuration:')
    print(f'  listen          : http://{args.host}:{args.port}')
    print(f'  M1 vLLM         : {SERVER.m1_server_url} (model={SERVER.m1_server_model})')
    print(f'  M2 vLLM         : {SERVER.m2_server_url} (model={SERVER.m2_server_model})')
    print(f'  inference_mode  : {SERVER.inference_mode}')
    print(f'  action_horizon  : {SERVER.action_horizon}')
    print(f'  image_size      : {SERVER.image_size}')
    print(f'  output_dir      : {SERVER.output_dir}')

    app.run(host=args.host, port=args.port)


if __name__ == '__main__':
    main()
