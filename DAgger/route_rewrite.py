# =============================================================================
# Fallback semantics (easily confused — read before modifying):
#
#   In the step loop, iteration t:
#     rgb_history.append(rgb)        → rgb_history[t] = ft  (observation at state_t)
#     predict act_t from ft
#     step_record {frame:t, action:act_t}
#     env.step(act_t)                → state_{t+1}
#     traj_points[t] = {frame_idx:t+1, pos:state_{t+1}, rgb:fallback_rgb}
#
#   "Fallback to frame t" really means: fallback to the WAYPOINT at state_{t+1},
#   reached after executing act_t from ft.  We keep:
#     frames  = [f0 .. ft]           # last frame is ft (observation before act_t)
#     actions = [a0 .. at]           # includes act_t (ft → waypoint)
#     records = [{0,a0} .. {t,at}]   # includes the record for act_t
#     final_state = state_{t+1}      # waypoint position for next segment to start
#
#   Non-final: next segment starts from waypoint state → first observation g0 is
#              new (state_{t+1}), no duplication.  reuse_last_frame = False.
#   Final:     append waypoint observation (best_p["rgb"]) + stop=0 to close the
#              trajectory at the goal.
# =============================================================================
import os
import json
import re
import sys
import argparse
import base64
import random
import time
from io import BytesIO
from collections import OrderedDict
import httpx
import numpy as np
import torch
import quaternion
import cv2
from PIL import Image
from tqdm import tqdm
import multiprocessing as mp

import habitat
from habitat.config.default import get_config
from omegaconf import OmegaConf
from habitat.utils.visualizations import maps
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.sims.habitat_simulator.actions import HabitatSimActions

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from agent_dual_qwen2_5_lm import SYSTEM_PROMPT, build_m2_prompt, get_images_for_modules_from_pil_history

# --- NumPy Compatibility Patch ---
if not hasattr(np, 'float'): np.float = float
if not hasattr(np, 'int'): np.int = int
if not hasattr(np, 'bool'): np.bool = bool

# Set image token limits for the released model interface.
os.environ['MAX_PIXELS'] = '1003520'
os.environ['IMAGE_MAX_TOKEN_NUM'] = '2048'


def _ordered_values(raw_value):
    if isinstance(raw_value, dict):
        return [raw_value[k] for k in sorted(raw_value.keys(), key=lambda x: int(x))]
    if isinstance(raw_value, list):
        return list(raw_value)
    return raw_value


def _normalize_seg_item(item, fallback_episode_id=None):
    item = dict(item)
    if "episode_id" not in item and fallback_episode_id is not None:
        item["episode_id"] = fallback_episode_id
    if "original_instruction" not in item and "instruction" in item:
        item["original_instruction"] = item["instruction"]
    if "sub_instructions" not in item and "split_instructions" in item:
        item["sub_instructions"] = item["split_instructions"]
    if "sub_instructions" in item:
        item["sub_instructions"] = _ordered_values(item["sub_instructions"])
    if "cut_points_details" in item:
        item["cut_points_details"] = _ordered_values(item["cut_points_details"])
    return item


def load_seg_data_list(seg_data_path):
    with open(seg_data_path, 'r') as f:
        raw_data = json.load(f)
    if isinstance(raw_data, dict):
        return [
            _normalize_seg_item(item, fallback_episode_id=episode_id)
            for episode_id, item in raw_data.items()
            if isinstance(item, dict)
        ]
    return [_normalize_seg_item(item) for item in raw_data]

try:
    from swift.infer_engine import TransformersEngine, InferRequest, RequestConfig
except ImportError:
    print("Error: ms-swift not found. Please install it.")
    raise

def dynamic_scene_worker_fn(worker_id, scene_queue, remaining, args):
    """Worker pulls one scene at a time.  Stays alive until ALL items are done."""
    gpu_list = args.gpus.split(',')
    physical_gpu = int(gpu_list[worker_id % len(gpu_list)])

    # Map the selected physical GPU to device 0 for EGL/OpenGL rendering.
    os.environ['CUDA_VISIBLE_DEVICES'] = str(physical_gpu)
    args.gpu_id = 0          # The selected GPU is device 0 inside the worker.
    args.worker_id = worker_id

    evaluator = ActionRepairEvaluator(args)
    while True:
        try:
            item = scene_queue.get(timeout=2.0)
        except:
            # Queue empty — check if all work is truly done.
            if remaining.value == 0:
                break
            continue
        scene_id, episode_ids = item
        print(
            f"[Worker {worker_id}] Claimed: {scene_id or 'unknown'} "
            f"({len(episode_ids)} episodes)",
            flush=True,
        )
        evaluator.run_custom(episode_ids)
        with remaining.get_lock():
            remaining.value -= 1


def _build_scene_queue(scene_groups, num_workers, max_eps_per_chunk=50):
    """Build queue items: large scenes split into chunks, small scenes kept whole.

    Large scenes are split so multiple workers can process them in parallel,
    keeping vLLM request concurrency high.  Small scenes stay as single items
    so a scene is loaded by at most one worker.
    """
    scene_items = sorted(scene_groups.items(), key=lambda x: len(x[1]), reverse=True)
    items = []
    for scene_id, eids in scene_items:
        n = len(eids)
        if n > max_eps_per_chunk:
            for start in range(0, n, max_eps_per_chunk):
                items.append((scene_id, list(eids[start:start + max_eps_per_chunk])))
        else:
            items.append((scene_id, list(eids)))
    return items


class ActionRepairEvaluator:
    def __init__(self, args):
        self.args = args
        self.output_root = args.output_root
        self.output_episode_suffix = str(getattr(args, "output_episode_suffix", "") or "")
        self.m2_server_url = args.m2_server_url.rstrip("/") if args.m2_server_url else None
        self.m2_server_model = args.m2_server_model
        self.image_size = (640, 480)
        self.max_history_frames = 100
        self._req_count = 0
        self._req_t0 = None
        
        # 1. Initialize the model, preferring the M2 vLLM server.
        if self.m2_server_url:
            print(f"[Worker {args.worker_id}] Using M2 server: {self.m2_server_url} (model={self.m2_server_model})")
            self.engine = None
            self.http_client = httpx.Client(
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=20),
                timeout=httpx.Timeout(120.0, connect=10.0),
            )
        else:
            print(f"[Worker {args.worker_id}] Loading Module 2 on GPU {args.gpu_id}")
            self.engine = TransformersEngine(
                model=args.model_path,
                model_type='qwen2_5_vl',
                torch_dtype='bfloat16',
                device_map={"": args.gpu_id}
            )
        self.request_config = RequestConfig(max_tokens=self.args.max_tokens, temperature=self.args.temperature)
        
        # 2. Initialize the Habitat environment.
        self.config = self._setup_config()
        self.env = habitat.Env(config=self.config)
        
        # 3. Load segmentation data.
        raw_data = load_seg_data_list(args.seg_data_path)
        self.seg_data = {str(item['episode_id']): item for item in raw_data}
        self.dense_trajectories = self._load_dense_trajectories(args.recover_release_coords_path)
        if getattr(self.args, "use_dense_segment_distance", False) and not self.dense_trajectories:
            print(
                f"[Worker {args.worker_id}] WARNING: --use_dense_segment_distance is enabled, "
                "but no dense trajectories were loaded. Falling back to segment-line distance.",
                flush=True,
            )
            
        self.result_log_path = f"./results/train/action_repair/result_worker_{args.worker_id}.json"
        os.makedirs(os.path.dirname(self.result_log_path), exist_ok=True)
        
        self.action_mapping = {"move forward 25 cm": 1, "turn left 15 degrees": 2, "turn right 15 degrees": 3, "stop": 0}
        self.habitat_to_id = {
            HabitatSimActions.stop: 0,
            HabitatSimActions.move_forward: 1,
            HabitatSimActions.turn_left: 2,
            HabitatSimActions.turn_right: 3,
        }

    def _load_dense_trajectories(self, coords_path):
        if not coords_path:
            return {}
        if not os.path.exists(coords_path):
            print(f"[Worker {self.args.worker_id}] Dense trajectory file not found: {coords_path}")
            return {}
        with open(coords_path, "r") as f:
            raw = json.load(f)
        items = raw.values() if isinstance(raw, dict) else raw
        trajectories = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            episode_id = item.get("episode_id", item.get("id", item.get("r2r_id")))
            coords = item.get("coordinates") or item.get("trajectory") or item.get("positions")
            if episode_id is None or not isinstance(coords, list):
                continue
            normalized = []
            for idx, point in enumerate(coords):
                if isinstance(point, dict):
                    position = point.get("position")
                    rotation = point.get("rotation")
                    step = point.get("step", idx + 1)
                else:
                    position = point
                    rotation = None
                    step = idx + 1
                if not isinstance(position, (list, tuple)) or len(position) < 3:
                    continue
                try:
                    normalized.append({
                        "step": int(step),
                        "position": [float(position[0]), float(position[1]), float(position[2])],
                        "rotation": rotation,
                    })
                except (TypeError, ValueError):
                    continue
            if normalized:
                normalized = sorted(normalized, key=lambda x: x["step"])
                keys = {str(episode_id)}
                try:
                    keys.add(str(int(str(episode_id))))
                except (TypeError, ValueError):
                    pass
                for key in keys:
                    trajectories[key] = normalized
        print(
            f"[Worker {self.args.worker_id}] Loaded dense release trajectories: "
            f"{len(trajectories)} episodes from {coords_path}",
            flush=True,
        )
        return trajectories

    def _normalize_scene_id(self, scene_id):
        if not scene_id:
            return ""
        # Support the mp3d/SCAN/SCAN.glb scene identifier format.
        if not scene_id.startswith("mp3d/"):
            return os.path.join("data/scene_datasets/mp3d", scene_id, f"{scene_id}.glb")
        return os.path.join("data/scene_datasets", scene_id)

    def _switch_scene_if_needed(self, scene_id):
        full_scene_id = self._normalize_scene_id(scene_id)
        if not full_scene_id:
            return
        if self.env.sim.config.sim_cfg.scene_id == full_scene_id:
            return
        print(
            f"[Worker {self.args.worker_id}] Switching scene from "
            f"{os.path.basename(self.env.sim.config.sim_cfg.scene_id)} to {scene_id}"
        )
        # Habitat 0.2+ reconfigure logic
        self.config.habitat.simulator.scene = full_scene_id
        self.env.sim.reconfigure(self.config.habitat.simulator)

    def _group_episode_ids_by_scene(self, episode_ids):
        scene_to_eids = OrderedDict()
        for eid in episode_ids:
            data = self.seg_data.get(str(eid))
            scene_id = data.get('scan_id', '') if data else ''
            if scene_id not in scene_to_eids:
                scene_to_eids[scene_id] = []
            scene_to_eids[scene_id].append(str(eid))
        return scene_to_eids

    def _setup_config(self):
        config = get_config(self.args.habitat_config)
        # Convert to plain DictConfig to avoid strict type validation in Habitat 0.2+
        config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
        
        # Habitat 0.2+ stores the simulator configuration under habitat.
        h_config = config.habitat
        h_config.dataset.type, h_config.dataset.split = self.args.dataset_type, self.args.split
        h_config.dataset.data_path = self.args.dataset_data_path.format(split=self.args.split)
        h_config.dataset.scenes_dir = "data/scene_datasets/"
        h_config.simulator.habitat_sim_v0.gpu_device_id = self.args.gpu_id
        h_config.simulator.forward_step_size = 0.25
        h_config.simulator.turn_angle = 15
        h_config.environment.max_episode_steps = 100000
        
        # Ensure measurements are present
        if "measurements" not in h_config.task:
            h_config.task.measurements = {}
        
        m = h_config.task.measurements
        if "distance_to_goal" not in m and "DISTANCE_TO_GOAL" not in m: 
            if isinstance(m, (list, tuple)):
                m.append("DISTANCE_TO_GOAL")
            else:
                m.distance_to_goal = {"type": "DistanceToGoal"}
        
        return config

    def _append_rgb_to_pil_history(self, rgb, pil_history):
        pil_history.append(Image.fromarray(rgb).convert('RGB').resize(self.image_size))
        if len(pil_history) > self.max_history_frames:
            pil_history.pop(0)

    def _get_images_for_model(self, rgb_pil_history):
        return get_images_for_modules_from_pil_history(rgb_pil_history, mode="module2")

    def _get_initial_subtask_pil_history(self, confirmed_rgb_pil_history):
        if self.args.hide_previous_subtask_memory:
            return []
        return list(confirmed_rgb_pil_history)

    def _predict(self, global_inst, sub_inst, rgb_pil_history):
        images, _, _ = self._get_images_for_model(rgb_pil_history)
        prompt = build_m2_prompt(global_inst, sub_inst, len(images))
        if self.m2_server_url:
            res_str = self._predict_openai(prompt, images).lower()
        else:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
            resp = self.engine.infer([InferRequest(messages=messages, images=images)], request_config=self.request_config)
            res_str = resp[0].choices[0].message.content.strip().lower()
        if "stop" in res_str and not re.search(r'move|turn', res_str): return [0]
        actions = []
        for p in [x.strip() for x in re.split(r'[,，\n]', res_str)]:
            for k, v in self.action_mapping.items():
                if k in p: actions.append(v); break
            if len(actions) >= 3: break
        return actions if actions else [1, 1, 1]

    def _predict_openai(self, prompt, images):
        content = []
        for img in images:
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode()
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        clean_text = re.sub(r'<image>', '', prompt).strip()
        content.append({"type": "text", "text": clean_text})

        payload = {
            "model": self.m2_server_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            "max_tokens": self.request_config.max_tokens,
            "temperature": self.request_config.temperature,
        }
        t0 = time.time()
        resp = self.http_client.post(f"{self.m2_server_url}/v1/chat/completions", json=payload)
        self._req_count += 1
        if self._req_t0 is None:
            self._req_t0 = t0
        elif self._req_count % 50 == 0:
            elapsed = t0 - self._req_t0
            rate = 50.0 / elapsed if elapsed > 0 else 0
            print(f"[Worker {self.args.worker_id}] req#{self._req_count}  rate={rate:.1f} req/s  "
                  f"vllm_latency={time.time() - t0:.2f}s", flush=True)
            self._req_t0 = t0
        if resp.status_code >= 400:
            body = resp.text
            if len(body) > 2000:
                body = body[:2000] + "...(truncated)"
            raise RuntimeError(
                f"M2 server request failed: status={resp.status_code}, url={self.m2_server_url}/v1/chat/completions, body={body}")
        return resp.json()["choices"][0]["message"]["content"].strip()

    def _quat_yaw(self, q):
        return np.arctan2(2 * (q.w * q.y + q.x * q.z), 1 - 2 * (q.y**2 + q.z**2))

    def _angle_diff_deg(self, current_rot, target_rot_q):
        diff = np.degrees(self._quat_yaw(current_rot) - self._quat_yaw(target_rot_q))
        while diff > 180:
            diff -= 360
        while diff < -180:
            diff += 360
        return diff

    def _target_metrics(self, current_pos, current_rot, target_info):
        t_pos = np.array(target_info['position'])
        dist = float(np.linalg.norm(current_pos - t_pos))
        target_rot = quaternion.quaternion(
            target_info['rotation']['w'],
            target_info['rotation']['x'],
            target_info['rotation']['y'],
            target_info['rotation']['z'],
        )
        angle_diff = abs(float(self._angle_diff_deg(current_rot, target_rot)))
        return dist, angle_diff

    def _check_success(self, current_pos, current_rot, target_info, is_final):
        dist, angle_diff = self._target_metrics(current_pos, current_rot, target_info)
        if is_final:
            return dist <= 2.0, dist, angle_diff
        return (dist <= 1.5 and angle_diff <= 45), dist, angle_diff

    def _rewrite_waypoint_score(self, dist, angle_diff, is_final):
        if is_final:
            return float(dist)
        return float(dist) / 1.5 + float(angle_diff) / 45.0

    def _precise_segment_success(self, st, target_info):
        dist, angle_abs = self._target_metrics(st.position, st.rotation, target_info)
        return (
            dist <= self.args.precise_success_distance and angle_abs <= self.args.precise_success_angle,
            dist,
            angle_abs,
        )

    def _point_to_segment_distance(self, p, a, b):
        p = np.array(p, dtype=np.float32)
        a = np.array(a, dtype=np.float32)
        b = np.array(b, dtype=np.float32)
        ab = b - a
        ab_norm_sq = float(np.dot(ab, ab))
        if ab_norm_sq < 1e-12:
            return float(np.linalg.norm(p - a))
        t = float(np.dot(p - a, ab) / ab_norm_sq)
        t = max(0.0, min(1.0, t))
        proj = a + t * ab
        return float(np.linalg.norm(p - proj))

    def _point_to_polyline_distance(self, p, points):
        if not points:
            return None
        if len(points) == 1:
            return self._point_distance(p, points[0])
        return min(
            self._point_to_segment_distance(p, points[i], points[i + 1])
            for i in range(len(points) - 1)
        )

    def _get_segment_reference_points(self, episode_id, all_points, seg_idx, gt_seg_start, gt_seg_end):
        if not getattr(self.args, "use_dense_segment_distance", False):
            return [gt_seg_start, gt_seg_end]

        dense_points = self._get_dense_segment_points(episode_id, all_points, seg_idx)
        reference_points = [
            p.get("position")
            for p in dense_points
            if isinstance(p, dict) and isinstance(p.get("position"), (list, tuple))
        ]
        if len(reference_points) < 2:
            return [gt_seg_start, gt_seg_end]
        return reference_points

    @staticmethod
    def _segment_distance_source(segment_reference_points):
        if segment_reference_points and len(segment_reference_points) > 2:
            return "dense_polyline"
        return "segment_line"

    def _gt_segment_distance(self, position, gt_seg_start, gt_seg_end, segment_reference_points=None):
        if segment_reference_points and len(segment_reference_points) > 2:
            dist = self._point_to_polyline_distance(position, segment_reference_points)
            if dist is not None:
                return dist
        return self._point_to_segment_distance(position, gt_seg_start, gt_seg_end)

    @staticmethod
    def _point_distance(p, q):
        return float(np.linalg.norm(np.array(p, dtype=np.float32) - np.array(q, dtype=np.float32)))

    def _get_dense_segment_points(self, episode_id, all_points, seg_idx):
        points = self.dense_trajectories.get(str(episode_id))
        if not points:
            return []
        if seg_idx + 1 >= len(all_points):
            return points
        start_frame = all_points[seg_idx].get("frame") if isinstance(all_points[seg_idx], dict) else None
        end_frame = all_points[seg_idx + 1].get("frame") if isinstance(all_points[seg_idx + 1], dict) else None
        if start_frame is None or end_frame is None:
            return points
        try:
            start_frame = int(start_frame)
            end_frame = int(end_frame)
        except (TypeError, ValueError):
            return points
        if end_frame < start_frame:
            start_frame, end_frame = end_frame, start_frame
        segment_points = [p for p in points if start_frame <= int(p.get("step", -1)) <= end_frame]
        return segment_points if segment_points else points

    def _rotation_yaw_deg(self, rotation):
        if rotation is None:
            return None
        if isinstance(rotation, dict):
            try:
                rotation = quaternion.quaternion(
                    float(rotation["w"]),
                    float(rotation["x"]),
                    float(rotation["y"]),
                    float(rotation["z"]),
                )
            except (KeyError, TypeError, ValueError):
                return None
        elif isinstance(rotation, (list, tuple)) and len(rotation) == 4:
            try:
                rotation = quaternion.quaternion(
                    float(rotation[3]),
                    float(rotation[0]),
                    float(rotation[1]),
                    float(rotation[2]),
                )
            except (TypeError, ValueError):
                return None
        return float(np.degrees(self._quat_yaw(rotation)))

    @staticmethod
    def _heading_error_from_yaws(current_yaw, target_yaw):
        diff = float(current_yaw) - float(target_yaw)
        while diff > 180:
            diff -= 360
        while diff < -180:
            diff += 360
        return abs(diff)

    def _dense_tangent_yaw_deg(self, segment_points, nearest_idx, min_motion=0.05):
        nearest_pos = np.array(segment_points[nearest_idx]["position"], dtype=np.float32)
        best = None
        max_radius = max(nearest_idx, len(segment_points) - nearest_idx - 1)
        for radius in range(1, max_radius + 1):
            prev_idx = max(0, nearest_idx - radius)
            next_idx = min(len(segment_points) - 1, nearest_idx + radius)
            if prev_idx == next_idx:
                continue
            prev_pos = np.array(segment_points[prev_idx]["position"], dtype=np.float32)
            next_pos = np.array(segment_points[next_idx]["position"], dtype=np.float32)
            delta = next_pos - prev_pos
            horizontal_norm = float(np.linalg.norm(delta[[0, 2]]))
            if horizontal_norm >= min_motion:
                best = delta
                break

            for candidate_idx in (next_idx, prev_idx):
                candidate_pos = np.array(segment_points[candidate_idx]["position"], dtype=np.float32)
                delta = candidate_pos - nearest_pos
                horizontal_norm = float(np.linalg.norm(delta[[0, 2]]))
                if horizontal_norm >= min_motion:
                    best = delta
                    break
            if best is not None:
                break

        if best is None:
            return None
        return float(np.degrees(np.arctan2(best[0], -best[2])))

    def _release_metrics(self, episode_id, all_points, seg_idx, position, rotation, gt_seg_start, gt_seg_end):
        segment_points = self._get_dense_segment_points(episode_id, all_points, seg_idx)
        current_yaw = float(np.degrees(self._quat_yaw(rotation)))
        if segment_points:
            nearest_idx, nearest_point = min(
                enumerate(segment_points),
                key=lambda item: self._point_distance(position, item[1]["position"]),
            )
            distance = self._point_distance(position, nearest_point["position"])
            tangent_yaw = self._dense_tangent_yaw_deg(segment_points, nearest_idx)
            heading_source = "dense_tangent"
            if tangent_yaw is None:
                tangent_yaw = self._rotation_yaw_deg(nearest_point.get("rotation"))
                heading_source = "dense_rotation"
            if tangent_yaw is not None:
                return distance, self._heading_error_from_yaws(current_yaw, tangent_yaw), heading_source
            return distance, self._segment_heading_error_deg(rotation, gt_seg_start, gt_seg_end), "segment_fallback"

        return (
            self._point_to_segment_distance(position, gt_seg_start, gt_seg_end),
            self._segment_heading_error_deg(rotation, gt_seg_start, gt_seg_end),
            "segment",
        )

    def _get_deviation_threshold_by_frames(self, seg_frame_len):
        if seg_frame_len is None:
            seg_frame_len = 0
        if seg_frame_len > self.args.long_segment_min_frames:
            return float(self.args.deviation_threshold_long)
        if seg_frame_len > self.args.medium_segment_min_frames:
            return float(self.args.deviation_threshold_medium)
        return float(self.args.deviation_threshold_short)

    def _get_drift_start_threshold_by_frames(self, seg_frame_len):
        if seg_frame_len is None:
            seg_frame_len = 0
        if seg_frame_len > self.args.long_segment_min_frames:
            return float(self.args.drift_start_threshold_long)
        if seg_frame_len > self.args.medium_segment_min_frames:
            return float(self.args.drift_start_threshold_medium)
        return float(self.args.drift_start_threshold_short)

    def _recover_release_heading_angle(self):
        if self.args.recover_release_heading_angle > 0:
            return float(self.args.recover_release_heading_angle)
        return float(self.args.precise_success_angle)

    def _recover_release_heading_ok(self, heading_error):
        if getattr(self.args, "recover_release_ignore_heading", False):
            return True
        # Conservative handoff heuristic.
        # Endpoint-heading alignment may delay release on long/curved segments, especially in RxR.
        return float(heading_error) <= self._recover_release_heading_angle()

    def _recover_release_metrics(
        self,
        position,
        rotation,
        target,
        gt_seg_start,
        gt_seg_end,
        segment_reference_points=None,
    ):
        dist_to_gt_segment = self._gt_segment_distance(
            position,
            gt_seg_start,
            gt_seg_end,
            segment_reference_points,
        )
        _, heading_error_to_target = self._target_metrics(position, rotation, target)
        return (
            dist_to_gt_segment,
            heading_error_to_target,
            f"{self._segment_distance_source(segment_reference_points)}_target_rotation",
        )

    @staticmethod
    def _segment_heading_yaw_deg(gt_seg_start, gt_seg_end):
        delta = np.array(gt_seg_end, dtype=np.float32) - np.array(gt_seg_start, dtype=np.float32)
        if float(np.linalg.norm(delta[[0, 2]])) < 1e-12:
            return None
        # Habitat yaw convention: 0° faces -Z, +90° faces +X.
        return float(np.degrees(np.arctan2(delta[0], -delta[2])))

    def _segment_heading_error_deg(self, current_rot, gt_seg_start, gt_seg_end):
        segment_yaw = self._segment_heading_yaw_deg(gt_seg_start, gt_seg_end)
        if segment_yaw is None:
            return 0.0
        current_yaw = float(np.degrees(self._quat_yaw(current_rot)))
        return self._heading_error_from_yaws(current_yaw, segment_yaw)

    @staticmethod
    def _find_continuous_drift_start(step_records, trigger_frame, drift_start_threshold):
        if trigger_frame is None:
            return None
        trigger_idx = None
        for idx, rec in enumerate(step_records):
            if int(rec.get("frame", -1)) == int(trigger_frame):
                trigger_idx = idx
                break
        if trigger_idx is None:
            return None
        start_idx = trigger_idx
        while start_idx > 0:
            prev = step_records[start_idx - 1]
            if float(prev.get("dist_to_gt_segment", 0.0)) < drift_start_threshold:
                break
            start_idx -= 1
        return int(step_records[start_idx]["frame"])

    @staticmethod
    def _apply_navigation_state_labels(step_records, drift_start_frame, trigger_frame, trigger_source,
                                        release_frame=None):
        for rec in step_records:
            rec["navigation_state"] = "normal"
        if drift_start_frame is None or trigger_frame is None:
            return
        if trigger_source != "deviation":
            return
        for rec in step_records:
            frame = int(rec.get("frame", -1))
            controller = str(rec.get("controller", ""))
            if int(drift_start_frame) <= frame < int(trigger_frame):
                rec["navigation_state"] = "drifting"
            elif frame >= int(trigger_frame):
                if controller.startswith("habitat_deviation"):
                    if release_frame is None or frame < int(release_frame):
                        rec["navigation_state"] = "recovering"

    @staticmethod
    def _navigation_state_summary(step_records):
        counts = {"normal": 0, "drifting": 0, "recovering": 0, "align": 0}
        ranges = {}
        for state in counts:
            frames = [int(rec["frame"]) for rec in step_records if rec.get("navigation_state") == state]
            counts[state] = len(frames)
            if frames:
                ranges[state] = [min(frames), max(frames)]
        return counts, ranges

    def _get_follower_action(self, follower, target_pos):
        next_action = follower.get_next_action(np.array(target_pos, dtype=np.float32))
        if next_action is None:
            return 0
        return self.habitat_to_id.get(next_action, 0)

    def _get_heading_alignment_action(self, current_rot, target_rot_q, active_turn_action=None):
        if active_turn_action in (2, 3):
            return active_turn_action
        yaw_diff_signed = self._angle_diff_deg(current_rot, target_rot_q)
        turn_amount = float(getattr(self.config.habitat.simulator, "turn_angle", 15.0))
        near_half_turn = abs(abs(yaw_diff_signed) - 180.0) <= (turn_amount / 2.0)
        if near_half_turn:
            return random.choice([2, 3])
        return 3 if yaw_diff_signed > 0 else 2

    def _controller_for_trigger(self, trigger_source):
        if trigger_source == "deviation":
            return "habitat_deviation"
        if trigger_source == "near_goal":
            return "habitat_near_goal"
        if trigger_source == "max_steps":
            return "habitat_max_steps"
        if trigger_source == "model_stop":
            return "habitat_model_stop"
        return "habitat"

    def _make_step_record(
        self,
        seg_idx,
        frame,
        action,
        state,
        target,
        gt_seg_start,
        gt_seg_end,
        controller,
        deviation_threshold,
        drift_start_threshold,
        segment_reference_points=None,
    ):
        dist, angle_abs = self._target_metrics(state.position, state.rotation, target)
        dev = self._gt_segment_distance(
            state.position,
            gt_seg_start,
            gt_seg_end,
            segment_reference_points,
        )
        rot = state.rotation
        if hasattr(rot, "w"):
            rotation = {
                "w": float(rot.w),
                "x": float(rot.x),
                "y": float(rot.y),
                "z": float(rot.z),
            }
        elif isinstance(rot, (list, tuple)) and len(rot) == 4:
            # Habitat set_agent_state lists in this repo use [x, y, z, w].
            rotation = {
                "w": float(rot[3]),
                "x": float(rot[0]),
                "y": float(rot[1]),
                "z": float(rot[2]),
            }
        else:
            rotation = None
        return {
            "segment": int(seg_idx),
            "frame": int(frame),
            "action": int(action),
            "agent_position": [float(x) for x in state.position],
            "agent_rotation": rotation,
            "dist_to_gt_segment": float(dev),
            "dist_to_target": float(dist),
            "heading_error_to_target": float(angle_abs),
            "controller": controller,
            "deviation_threshold": float(deviation_threshold),
            "drift_start_threshold": float(drift_start_threshold),
            "navigation_state": "normal",
        }

    def _get_segment_frame_length(self, all_points, seg_idx):
        if seg_idx + 1 >= len(all_points):
            return None
        start_frame = all_points[seg_idx].get("frame") if isinstance(all_points[seg_idx], dict) else None
        end_frame = all_points[seg_idx + 1].get("frame") if isinstance(all_points[seg_idx + 1], dict) else None
        if start_frame is None or end_frame is None:
            return None
        try:
            return max(0, int(end_frame) - int(start_frame))
        except (TypeError, ValueError):
            return None

    def _get_segment_step_limit(self, all_points, seg_idx):
        gt_frame_len = self._get_segment_frame_length(all_points, seg_idx)
        if gt_frame_len is not None and gt_frame_len > 0 and self.args.segment_timeout_gt_ratio > 0:
            return max(1, int(np.ceil(gt_frame_len * self.args.segment_timeout_gt_ratio))), gt_frame_len
        return max(1, int(self.args.max_segment_steps)), gt_frame_len

    def _make_attempt_trace(
        self,
        seg_idx,
        attempt_number,
        current_start_state,
        target,
        step_records,
        action_history,
        stop_called,
        target_reached,
        final_state,
        final_dist,
        final_angle,
    ):
        """Serialize one full model attempt for qualitative top-down plots.

        ``rewrite_S.json`` intentionally keeps only the selected attempt, which
        makes the original training corpus compact but hides the rejected
        branches.  This sidecar uses JSON primitives only and includes the
        terminal state after the last action, so a failed branch is drawable as
        a complete polyline.
        """
        positions = []
        for position in [current_start_state.get("pos")] + [rec.get("agent_position") for rec in step_records]:
            if not isinstance(position, (list, tuple)) or len(position) < 3:
                continue
            point = [float(position[0]), float(position[1]), float(position[2])]
            if not positions or point != positions[-1]:
                positions.append(point)
        terminal_position = [float(value) for value in final_state.position]
        if not positions or terminal_position != positions[-1]:
            positions.append(terminal_position)
        return {
            "segment": int(seg_idx),
            "attempt_number": int(attempt_number),
            "status": "success" if stop_called and target_reached else (
                "waypoint_reached_without_stop" if target_reached else "failed"
            ),
            "stop_called": bool(stop_called),
            "target_reached": bool(target_reached),
            "target_position": [float(value) for value in target["position"]],
            "positions": positions,
            "actions": [int(action) for action in action_history],
            "step_records": [dict(record) for record in step_records],
            "final_distance_to_target": float(final_dist),
            "final_heading_error_to_target": float(final_angle),
        }

    def _run_rewrite_attempts(
        self,
        data,
        sub_inst,
        seg_idx,
        target,
        all_points,
        current_start_state,
        confirmed_rgb_pil_history,
        segment_step_limit,
        segment_frame_len_gt,
        is_final,
    ):
        best_attempt = None
        attempt_traces = []
        gt_seg_start = all_points[seg_idx]["position"]
        gt_seg_end = all_points[seg_idx + 1]["position"]
        segment_reference_points = self._get_segment_reference_points(
            data.get("episode_id"),
            all_points,
            seg_idx,
            gt_seg_start,
            gt_seg_end,
        )
        deviation_threshold = self._get_deviation_threshold_by_frames(segment_frame_len_gt)
        drift_start_threshold = self._get_drift_start_threshold_by_frames(segment_frame_len_gt)

        for attempt_idx in range(self.args.max_attempts_per_segment):
            self.env.sim.set_agent_state(current_start_state["pos"], current_start_state["rot"])
            rgb_history = []
            rgb_pil_history = self._get_initial_subtask_pil_history(confirmed_rgb_pil_history)
            action_history, traj_points, step_count, stop_called = [], [], 0, False
            attempt_step_records = []
            pending_actions = []
            attempt_number = attempt_idx + 1
            attempt_accepted = False

            while step_count < segment_step_limit and not stop_called:
                obs = self.env.sim.get_sensor_observations()
                rgb = obs.get("rgb", obs.get("RGB"))
                self._append_rgb_to_pil_history(rgb, rgb_pil_history)

                if not pending_actions:
                    pending_actions = self._predict(data['original_instruction'], sub_inst, rgb_pil_history)

                act = pending_actions.pop(0)
                if act == 0:
                    stop_called = True
                    break

                rgb_history.append(rgb)
                st_before = self.env.sim.get_agent_state()
                local_frame = len(rgb_history) - 1

                attempt_step_records.append(
                    self._make_step_record(
                        seg_idx,
                        local_frame,
                        act,
                        st_before,
                        target,
                        gt_seg_start,
                        gt_seg_end,
                        "model",
                        deviation_threshold,
                        drift_start_threshold,
                        segment_reference_points,
                    )
                )
                action_history.append(act)
                self.env.sim.step(act)
                st = self.env.sim.get_agent_state()

                ok, d, angle = self._check_success(st.position, st.rotation, target, is_final)
                fallback_rgb = None
                if ok:
                    fallback_obs = self.env.sim.get_sensor_observations()
                    fallback_rgb = fallback_obs.get("rgb", fallback_obs.get("RGB"))
                rot_list = [st.rotation.x, st.rotation.y, st.rotation.z, st.rotation.w] if hasattr(st.rotation, 'x') else st.rotation
                traj_points.append({
                    "pos": st.position,
                    "rot": rot_list,
                    "dist": d,
                    "angle": angle,
                    "score": self._rewrite_waypoint_score(d, angle, is_final),
                    "ok": ok,
                    "frame_idx": len(rgb_history),
                    "rgb": fallback_rgb,
                })
                step_count += 1

            if stop_called:
                final_st = self.env.sim.get_agent_state()
                ok_end, dist_end, angle_end = self._check_success(final_st.position, final_st.rotation, target, is_final)
                if ok_end:
                    final_rot_list = (
                        [final_st.rotation.x, final_st.rotation.y, final_st.rotation.z, final_st.rotation.w]
                        if hasattr(final_st.rotation, 'x')
                        else final_st.rotation
                    )
                    # rgb_history does NOT include the stop observation (loop breaks
                    # before appending on act==0).  Always capture the waypoint
                    # frame — even if the model stopped on its first action (empty
                    # action_history), the terminal frame is needed so
                    # _build_frame_aligned_actions can append stop=0.
                    final_obs = self.env.sim.get_sensor_observations()
                    final_rgb = final_obs.get("rgb", final_obs.get("RGB"))
                    rgb_history.append(final_rgb)
                    best_attempt = {
                        "status": "Success",
                        "dist": dist_end,
                        "angle": angle_end,
                        "score": self._rewrite_waypoint_score(dist_end, angle_end, is_final),
                        "frames": rgb_history,
                        "actions": action_history,
                        "step_records": attempt_step_records,
                        "final_state": {"pos": final_st.position, "rot": final_rot_list},
                        "attempt_number": attempt_number,
                    }
                    attempt_accepted = True

            valid_pts = [p for p in traj_points if p["ok"]]
            if valid_pts:
                best_p = min(valid_pts, key=lambda x: x["score"])
                if best_attempt is None or best_p["score"] < best_attempt["score"]:
                    # idx = best_p["frame_idx"] = len(rgb_history) after appending ft = t+1.
                    # best_p["pos"]/rgb = state/observation at waypoint (after act_t).
                    # We keep [f0..ft] and [a0..at]: act_t is the action from ft to waypoint.
                    # Final: also append waypoint frame + stop=0.
                    idx = best_p["frame_idx"]
                    selected_frames = list(rgb_history[:idx])             # [f0..ft]
                    selected_actions = list(action_history[:idx]) if idx > 0 else []  # [a0..at]
                    if is_final:
                        if best_p.get("rgb") is not None:
                            selected_frames.append(best_p["rgb"])        # waypoint observation
                        selected_actions.append(0)                       # stop
                    best_attempt = {
                        "status": "Fallback",
                        "dist": best_p["dist"],
                        "angle": best_p["angle"],
                        "score": best_p["score"],
                        "frames": selected_frames,
                        "actions": selected_actions,
                        "step_records": attempt_step_records[:idx],      # [{0,a0}..{t,at}]
                        "final_state": {"pos": best_p["pos"], "rot": best_p["rot"]},
                        "attempt_number": attempt_number,
                    }
                # A rollout only counts as a direct success after its explicit
                # STOP.  But once this attempt has stopped or timed out, a
                # visited point inside the waypoint threshold is a valid
                # fallback: accept it immediately instead of collecting the
                # remaining retry trajectories.
                attempt_accepted = True

            if getattr(self.args, "save_attempt_traces", False):
                final_state = self.env.sim.get_agent_state()
                final_ok, final_dist, final_angle = self._check_success(
                    final_state.position, final_state.rotation, target, is_final,
                )
                attempt_traces.append(
                    self._make_attempt_trace(
                        seg_idx,
                        attempt_number,
                        current_start_state,
                        target,
                        attempt_step_records,
                        action_history,
                        stop_called,
                        bool(final_ok or valid_pts),
                        final_state,
                        final_dist,
                        final_angle,
                    )
                )
            if attempt_accepted:
                break

        return best_attempt, attempt_traces

    def _rollout_expert_to_subgoal(
        self,
        seg_idx,
        target,
        gt_seg_start,
        gt_seg_end,
        expert_follower,
        full_rgb_trajectory,
        rgb_history,
        rgb_pil_history,
        action_history,
        segment_step_records,
        trigger_source,
        deviation_threshold,
        drift_start_threshold,
        segment_reference_points=None,
    ):
        final_dist = None
        final_angle = None
        heading_align_action = None
        target_rot = quaternion.quaternion(
            target['rotation']['w'],
            target['rotation']['x'],
            target['rotation']['y'],
            target['rotation']['z'],
        )
        for _ in range(self.args.max_oracle_steps):
            obs = self.env.sim.get_sensor_observations()
            rgb = obs.get("rgb", obs.get("RGB"))
            rgb_history.append(rgb)
            self._append_rgb_to_pil_history(rgb, rgb_pil_history)
            global_frame = len(full_rgb_trajectory) + len(rgb_history) - 1

            st = self.env.sim.get_agent_state()
            seg_ok, dist, angle_abs = self._precise_segment_success(st, target)
            final_dist, final_angle = dist, angle_abs
            if seg_ok:
                return True, final_dist, final_angle

            controller = self._controller_for_trigger(trigger_source)
            act = self._get_follower_action(expert_follower, target['position'])
            if act == 0:
                if dist <= self.args.direct_intervene_distance:
                    heading_align_action = self._get_heading_alignment_action(
                        st.rotation,
                        target_rot,
                        heading_align_action,
                    )
                    act = heading_align_action
                else:
                    return False, final_dist, final_angle
            else:
                heading_align_action = None

            segment_step_records.append(
                self._make_step_record(
                    seg_idx,
                    global_frame,
                    act,
                    st,
                    target,
                    gt_seg_start,
                    gt_seg_end,
                    controller,
                    deviation_threshold,
                    drift_start_threshold,
                    segment_reference_points,
                )
            )
            action_history.append(act)
            self.env.sim.step(act)

        final_st = self.env.sim.get_agent_state()
        seg_ok, final_dist, final_angle = self._precise_segment_success(final_st, target)
        return bool(seg_ok), final_dist, final_angle

    def _run_subgoal_dagger_segment(
        self,
        data,
        sub_inst,
        seg_idx,
        target,
        all_points,
        current_start_state,
        full_rgb_trajectory,
        confirmed_rgb_pil_history,
        segment_frame_len_gt,
    ):
        self.env.sim.set_agent_state(current_start_state["pos"], current_start_state["rot"])
        gt_seg_start = all_points[seg_idx]["position"]
        gt_seg_end = all_points[seg_idx + 1]["position"]
        segment_reference_points = self._get_segment_reference_points(
            data.get("episode_id"),
            all_points,
            seg_idx,
            gt_seg_start,
            gt_seg_end,
        )
        drift_start_threshold = self._get_drift_start_threshold_by_frames(segment_frame_len_gt)
        base_deviation_threshold = self._get_deviation_threshold_by_frames(segment_frame_len_gt)
        target_rot = quaternion.quaternion(
            target['rotation']['w'],
            target['rotation']['x'],
            target['rotation']['y'],
            target['rotation']['z'],
        )
        expert_follower = ShortestPathFollower(
            self.env.sim,
            goal_radius=self.args.precise_success_distance,
            return_one_hot=False,
            stop_on_error=True,
        )

        rgb_history = []
        rgb_pil_history = self._get_initial_subtask_pil_history(confirmed_rgb_pil_history)
        action_history = []
        segment_step_records = []
        pending_actions = []
        expert_triggered = False
        expert_active = False
        deviation_triggered = False
        near_goal_triggered = False
        trigger_source = "none"
        expert_intervene_frame = None
        deviation_intervene_frame = None
        expert_release_frame = None
        expert_release_source = None
        drift_start_frame = None
        seg_success = False
        final_dist = None
        final_angle = None
        heading_align_action = None

        for _ in range(self.args.max_steps_per_segment):
            obs = self.env.sim.get_sensor_observations()
            rgb = obs.get("rgb", obs.get("RGB"))
            rgb_history.append(rgb)
            self._append_rgb_to_pil_history(rgb, rgb_pil_history)
            global_frame = len(full_rgb_trajectory) + len(rgb_history) - 1

            st = self.env.sim.get_agent_state()
            seg_ok, dist, angle_abs = self._precise_segment_success(st, target)
            final_dist, final_angle = dist, angle_abs
            dev = self._gt_segment_distance(
                st.position,
                gt_seg_start,
                gt_seg_end,
                segment_reference_points,
            )
            release_dev, release_heading_error, release_heading_source = self._recover_release_metrics(
                st.position,
                st.rotation,
                target,
                gt_seg_start,
                gt_seg_end,
                segment_reference_points,
            )
            if seg_ok:
                seg_success = True
                break

            if not expert_active and dist < self.args.direct_intervene_distance:
                expert_triggered = True
                expert_active = True
                near_goal_triggered = True
                trigger_source = "near_goal"
                if expert_intervene_frame is None:
                    expert_intervene_frame = global_frame

            if not expert_active and dev >= base_deviation_threshold:
                expert_triggered = True
                expert_active = True
                deviation_triggered = True
                trigger_source = "deviation"
                if expert_intervene_frame is None:
                    expert_intervene_frame = global_frame
                if deviation_intervene_frame is None:
                    deviation_intervene_frame = global_frame

            release_gt_distance = (
                float(self.args.recover_release_gt_distance)
                if self.args.recover_release_gt_distance > 0
                else float(drift_start_threshold)
            )
            if (
                expert_active
                and trigger_source == "deviation"
                and deviation_intervene_frame is not None
                and global_frame > deviation_intervene_frame
                and release_dev <= release_gt_distance
                and self._recover_release_heading_ok(release_heading_error)
            ):
                expert_active = False
                expert_release_frame = global_frame
                expert_release_source = release_heading_source
                pending_actions = []

            if not expert_active:
                if not pending_actions:
                    pending_actions = self._predict(data['original_instruction'], sub_inst, rgb_pil_history)
                act = pending_actions.pop(0)
                controller = "model"
                if act == 0:
                    expert_triggered = True
                    expert_active = True
                    trigger_source = "model_stop"
                    if expert_intervene_frame is None:
                        expert_intervene_frame = global_frame
                    controller = "habitat_model_stop"
                    act = self._get_follower_action(expert_follower, target['position'])
            else:
                controller = self._controller_for_trigger(trigger_source)
                act = self._get_follower_action(expert_follower, target['position'])

            if act == 0:
                if dist <= self.args.direct_intervene_distance:
                    heading_align_action = self._get_heading_alignment_action(
                        st.rotation,
                        target_rot,
                        heading_align_action,
                    )
                    act = heading_align_action
                else:
                    break
            else:
                heading_align_action = None

            segment_step_records.append(
                self._make_step_record(
                    seg_idx,
                    global_frame,
                    act,
                    st,
                    target,
                    gt_seg_start,
                    gt_seg_end,
                    controller,
                    base_deviation_threshold,
                    drift_start_threshold,
                    segment_reference_points,
                )
            )
            action_history.append(act)
            self.env.sim.step(act)

        if not seg_success:
            if not expert_active:
                expert_triggered = True
                expert_active = True
                trigger_source = "max_steps"
                if expert_intervene_frame is None:
                    expert_intervene_frame = len(full_rgb_trajectory) + len(rgb_history)
            seg_success, final_dist, final_angle = self._rollout_expert_to_subgoal(
                seg_idx,
                target,
                gt_seg_start,
                gt_seg_end,
                expert_follower,
                full_rgb_trajectory,
                rgb_history,
                rgb_pil_history,
                action_history,
                segment_step_records,
                trigger_source,
                base_deviation_threshold,
                drift_start_threshold,
                segment_reference_points=segment_reference_points,
            )

        if deviation_triggered:
            drift_start_frame = self._find_continuous_drift_start(
                segment_step_records,
                deviation_intervene_frame,
                drift_start_threshold,
            )
        label_trigger_source = "deviation" if deviation_triggered else trigger_source
        self._apply_navigation_state_labels(
            segment_step_records,
            drift_start_frame,
            deviation_intervene_frame,
            label_trigger_source,
        )
        navigation_state_counts, navigation_state_ranges = self._navigation_state_summary(segment_step_records)
        final_st = self.env.sim.get_agent_state()
        final_state = {
            "pos": final_st.position,
            "rot": [final_st.rotation.x, final_st.rotation.y, final_st.rotation.z, final_st.rotation.w],
        }
        return {
            "success": bool(seg_success),
            "dist": final_dist,
            "angle": final_angle,
            "frames": rgb_history,
            "actions": action_history,
            "step_records": segment_step_records,
            "final_state": final_state,
            "metadata": {
                "segment_source": "subgoal_dagger",
                "status": "Success" if seg_success else "Failed",
                "previous_subtask_memory_visible": not bool(self.args.hide_previous_subtask_memory),
                "deviation_threshold_base": float(base_deviation_threshold),
                "drift_start_threshold": float(drift_start_threshold),
                "expert_triggered": bool(expert_triggered),
                "deviation_triggered": bool(deviation_triggered),
                "near_goal_triggered": bool(near_goal_triggered),
                "drift_start_frame": drift_start_frame,
                "deviation_intervene_frame": deviation_intervene_frame,
                "expert_intervene_frame": expert_intervene_frame,
                "expert_release_frame": expert_release_frame,
                "expert_release_source": expert_release_source,
                "trigger_source": trigger_source,
                "recover_release_gt_distance": (
                    float(self.args.recover_release_gt_distance)
                    if self.args.recover_release_gt_distance > 0
                    else float(drift_start_threshold)
                ),
                "recover_release_heading_angle": float(self._recover_release_heading_angle()),
                "recover_release_ignore_heading": bool(getattr(self.args, "recover_release_ignore_heading", False)),
                "gt_segment_distance_source": self._segment_distance_source(segment_reference_points),
                "gt_segment_reference_points": int(len(segment_reference_points)),
                "navigation_state_counts": navigation_state_counts,
                "navigation_state_ranges": navigation_state_ranges,
                "step_count": len(segment_step_records),
            },
        }

    def run_custom(self, episode_ids):
        """Custom segmented data handling without Habitat split"""
        for scene_id, eids in self._group_episode_ids_by_scene(episode_ids).items():
            self._switch_scene_if_needed(scene_id)
            for eid in tqdm(eids, desc=f"Worker {self.args.worker_id} | scene {scene_id or 'unknown'}"):
                self._process_episode(eid)

    def _output_episode_id(self, eid):
        return f"{eid}{self.output_episode_suffix}"

    def _process_episode(self, eid):
        output_eid = self._output_episode_id(eid)
        save_path = os.path.join(self.output_root, output_eid, "rgb/rewrite")
        if os.path.exists(os.path.join(save_path, "rewrite_S.json")): return
        if eid not in self.seg_data: return
        
        data = self.seg_data[eid]
        # Support dictionary-form sub-instructions.
        if isinstance(data['sub_instructions'], dict):
            sub_inst_dict = data['sub_instructions']
            sub_instructions = [sub_inst_dict[k].lstrip('0123456789. ') for k in sorted(sub_inst_dict.keys(), key=int)]
        else:
            sub_instructions = [s.lstrip('0123456789. ') for s in data['sub_instructions']]
            
        # Support dictionary-form cut-point details.
        if isinstance(data['cut_points_details'], dict):
            cp_dict = data['cut_points_details']
            all_points = [cp_dict[k] for k in sorted(cp_dict.keys(), key=int)]
        else:
            all_points = data['cut_points_details']

        start_state = all_points[0]
        targets = all_points[1:] 
        
        self.env.sim.set_agent_state(start_state["position"], [start_state["rotation"]['x'], start_state["rotation"]['y'], start_state["rotation"]['z'], start_state["rotation"]['w']])
        current_start_state = {"pos": start_state["position"], "rot": [start_state["rotation"]['x'], start_state["rotation"]['y'], start_state["rotation"]['z'], start_state["rotation"]['w']]}
        
        full_rgb_trajectory, full_action_trajectory, full_step_records = [], [-1], []
        all_rewrite_attempt_traces = []
        segment_history_results, episode_failed = [], False
        episode_has_subgoal_dagger = False
        failure_reason = None
        confirmed_rgb_pil_history = []
        reuse_last_frame_for_next_segment = False

        for seg_idx, (sub_inst, target) in enumerate(zip(sub_instructions, targets)):
            is_final, best_attempt = (seg_idx == len(sub_instructions) - 1), None
            segment_step_limit, segment_frame_len_gt = self._get_segment_step_limit(all_points, seg_idx)
            gt_seg_start = all_points[seg_idx]["position"]
            gt_seg_end = all_points[seg_idx + 1]["position"]
            segment_reference_points = self._get_segment_reference_points(
                eid,
                all_points,
                seg_idx,
                gt_seg_start,
                gt_seg_end,
            )
            segment_distance_source = self._segment_distance_source(segment_reference_points)
            deviation_threshold = self._get_deviation_threshold_by_frames(segment_frame_len_gt)
            drift_start_threshold = self._get_drift_start_threshold_by_frames(segment_frame_len_gt)

            best_attempt, segment_attempt_traces = self._run_rewrite_attempts(
                data,
                sub_inst,
                seg_idx,
                target,
                all_points,
                current_start_state,
                confirmed_rgb_pil_history,
                segment_step_limit,
                segment_frame_len_gt,
                is_final,
            )
            all_rewrite_attempt_traces.extend(segment_attempt_traces)

            if best_attempt:
                if reuse_last_frame_for_next_segment and best_attempt["frames"]:
                    best_attempt["frames"] = best_attempt["frames"][1:]

                step_idx_start = len(full_step_records)
                frame_offset = len(full_rgb_trajectory)
                segment_step_records = []
                for rec in best_attempt.get("step_records", []):
                    rec = dict(rec)
                    rec["frame"] = int(frame_offset + rec["frame"])
                    segment_step_records.append(rec)
                full_rgb_trajectory.extend(best_attempt["frames"])
                full_action_trajectory.extend(best_attempt["actions"])
                full_step_records.extend(segment_step_records)
                for frame in best_attempt["frames"]:
                    self._append_rgb_to_pil_history(frame, confirmed_rgb_pil_history)
                cut_point = len(full_rgb_trajectory) - 1
                step_idx_end = len(full_step_records) - 1
                segment_source = best_attempt.get("segment_source", "rewrite")
                navigation_state_counts, navigation_state_ranges = self._navigation_state_summary(segment_step_records)
                if not best_attempt.get("metadata"):
                    navigation_state_counts = {
                        "normal": len(segment_step_records),
                        "drifting": 0,
                        "recovering": 0,
                        "align": 0,
                    }
                    navigation_state_ranges = (
                        {"normal": [segment_step_records[0]["frame"], segment_step_records[-1]["frame"]]}
                        if segment_step_records else {}
                    )
                segment_extra_meta = dict(best_attempt.get("metadata", {}))
                segment_extra_meta.pop("navigation_state_counts", None)
                segment_extra_meta.pop("navigation_state_ranges", None)
                seg_meta = {
                    "segment": seg_idx,
                    "segment_source": segment_source,
                    "status": best_attempt["status"],
                    "dist": best_attempt["dist"],
                    "heading_error_to_target": best_attempt.get("angle"),
                    "waypoint_score": best_attempt.get("score"),
                    "previous_subtask_memory_visible": not bool(self.args.hide_previous_subtask_memory),
                    "cut_point_frame": cut_point,
                    "segment_frame_len_gt": segment_frame_len_gt,
                    "segment_step_limit": segment_step_limit,
                    "deviation_threshold": float(deviation_threshold),
                    "drift_start_threshold": float(drift_start_threshold),
                    "gt_segment_distance_source": segment_distance_source,
                    "gt_segment_reference_points": int(len(segment_reference_points)),
                    "navigation_state_counts": navigation_state_counts,
                    "navigation_state_ranges": navigation_state_ranges,
                    "step_count": len(segment_step_records),
                    "step_record_start_idx": step_idx_start,
                    "step_record_end_idx": step_idx_end,
                    "attempt_number": best_attempt.get("attempt_number"),
                }
                seg_meta.update(segment_extra_meta)
                segment_history_results.append(seg_meta)
                
                current_start_state = best_attempt["final_state"]
                self.env.sim.set_agent_state(current_start_state["pos"], current_start_state["rot"])
                reuse_last_frame_for_next_segment = False
            else:
                episode_has_subgoal_dagger = True
                dagger_result = self._run_subgoal_dagger_segment(
                    data,
                    sub_inst,
                    seg_idx,
                    target,
                    all_points,
                    current_start_state,
                    full_rgb_trajectory,
                    confirmed_rgb_pil_history,
                    segment_frame_len_gt,
                )
                if reuse_last_frame_for_next_segment and dagger_result["frames"]:
                    dagger_result["frames"] = dagger_result["frames"][1:]
                    for rec in dagger_result["step_records"]:
                        rec["frame"] = int(rec["frame"]) - 1
                reuse_last_frame_for_next_segment = False

                step_idx_start = len(full_step_records)
                full_rgb_trajectory.extend(dagger_result["frames"])
                full_action_trajectory.extend(dagger_result["actions"])
                full_step_records.extend(dagger_result["step_records"])
                for frame in dagger_result["frames"]:
                    self._append_rgb_to_pil_history(frame, confirmed_rgb_pil_history)
                cut_point = len(full_rgb_trajectory) - 1
                step_idx_end = len(full_step_records) - 1

                seg_meta = {
                    "segment": seg_idx,
                    "dist": float(dagger_result["dist"]) if dagger_result["dist"] is not None else None,
                    "heading_error_to_target": (
                        float(dagger_result["angle"]) if dagger_result["angle"] is not None else None
                    ),
                    "cut_point_frame": cut_point,
                    "segment_frame_len_gt": segment_frame_len_gt,
                    "segment_step_limit": segment_step_limit,
                    "rewrite_failed_attempts": int(self.args.max_attempts_per_segment),
                    "gt_segment_distance_source": segment_distance_source,
                    "gt_segment_reference_points": int(len(segment_reference_points)),
                    "step_record_start_idx": step_idx_start,
                    "step_record_end_idx": step_idx_end,
                }
                seg_meta.update(dagger_result["metadata"])
                segment_history_results.append(seg_meta)

                if dagger_result["success"]:
                    current_start_state = dagger_result["final_state"]
                    self.env.sim.set_agent_state(current_start_state["pos"], current_start_state["rot"])
                else:
                    failure_reason = {
                        "failed_segment": seg_idx,
                        "target_is_final": is_final,
                        "saved_until_segment": len(segment_history_results) - 1,
                        "failed_stage": "subgoal_dagger",
                        "max_attempts_per_segment": self.args.max_attempts_per_segment,
                        "max_steps_per_segment": self.args.max_steps_per_segment,
                        "max_oracle_steps": self.args.max_oracle_steps,
                        "segment_timeout_gt_ratio": self.args.segment_timeout_gt_ratio,
                        "segment_frame_len_gt": segment_frame_len_gt,
                        "segment_step_limit": segment_step_limit,
                    }
                    episode_failed = True
                    break

        self._save_results(
            eid,
            not episode_failed,
            episode_has_subgoal_dagger,
            data['original_instruction'],
            segment_history_results,
            full_rgb_trajectory,
            full_action_trajectory,
            full_step_records,
            save_path,
            failure_reason=failure_reason,
            attempt_traces=all_rewrite_attempt_traces,
        )

    @staticmethod
    def _build_frame_aligned_actions(num_frames, step_records, success):
        if num_frames <= 0:
            return [-1]
        frame_to_action = {
            int(rec["frame"]): int(rec["action"])
            for rec in step_records
            if "frame" in rec and "action" in rec
        }
        frame_actions = []
        for frame_idx in range(num_frames):
            if frame_idx in frame_to_action:
                # Exact match: action recorded FROM this frame.
                frame_actions.append(frame_to_action[frame_idx])
            elif success and frame_idx == num_frames - 1:
                # Implicit stop: last frame has no step_record (waypoint / stop obs).
                frame_actions.append(0)
            elif frame_idx + 1 in frame_to_action:
                # Gap (e.g. cutpoint fallback): no record for frame_idx, use the
                # next segment's first action which belongs to this frame.
                frame_actions.append(frame_to_action[frame_idx + 1])
            else:
                next_frames = [f for f in frame_to_action if f > frame_idx]
                frame_actions.append(frame_to_action[min(next_frames)] if next_frames else 0)
        return [-1] + frame_actions

    def _save_results(
        self,
        eid,
        success,
        has_subgoal_dagger,
        instruction,
        results,
        rgbs,
        actions,
        step_records,
        save_path,
        failure_reason=None,
        attempt_traces=None,
    ):
        should_save = success or len(rgbs) > 0
        if should_save:
            os.makedirs(save_path, exist_ok=True)
            try:
                if not bool(getattr(self.args, "save_json_only", False)):
                    for i, frame in enumerate(rgbs):
                        Image.fromarray(frame).convert('RGB').save(os.path.join(save_path, f"{i:04d}.jpg"))

                terminal_stop_source = None
                if success:
                    last_segment = results[-1] if results else {}
                    last_status = str(last_segment.get("status", "")).lower()
                    last_segment_source = str(last_segment.get("segment_source", "")).lower()
                    if has_subgoal_dagger or last_status == "fallback" or "fallback" in last_segment_source:
                        terminal_stop_source = "fallback_stop"
                    else:
                        terminal_stop_source = "real_stop"
                    actions = self._build_frame_aligned_actions(len(rgbs), step_records, bool(success))

                rewrite_meta = {
                    "episode_id": eid,
                    "instruction": instruction,
                    "segments": results,
                    "success": bool(success),
                    "episode_success": bool(success),
                    "partial": not bool(success),
                    "has_subgoal_dagger": bool(has_subgoal_dagger),
                    "terminal_stop_source": terminal_stop_source,
                    "failure_reason": failure_reason,
                    "total_frames": len(rgbs),
                    "actions": actions,
                    "step_records": step_records,
                }
                if getattr(self.args, "save_attempt_traces", False):
                    rewrite_meta["attempt_traces_file"] = "attempt_traces.json"
                    rewrite_meta["attempt_trace_count"] = len(attempt_traces or [])
                with open(os.path.join(save_path, "rewrite_S.json"), "w") as f:
                    json.dump(rewrite_meta, f, indent=4)
                if getattr(self.args, "save_attempt_traces", False):
                    attempt_trace_meta = {
                        "schema_version": 1,
                        "episode_id": eid,
                        "instruction": instruction,
                        "success": bool(success),
                        "attempts": attempt_traces or [],
                    }
                    with open(os.path.join(save_path, "attempt_traces.json"), "w") as f:
                        json.dump(attempt_trace_meta, f, indent=2)
                
                with open(self.result_log_path, 'a') as f:
                    f.write(json.dumps({
                        "episode_id": eid,
                        "success": bool(success),
                        "saved": True,
                        "partial": not bool(success),
                        "has_subgoal_dagger": bool(has_subgoal_dagger),
                        "segments": results,
                        "failure_reason": failure_reason,
                    }) + "\n")
                status = "Successfully" if success else "Partially"
                print(f"Episode {eid} Saved {status}.")
                
            except Exception as e:
                print(f"Error saving results for episode {eid}: {e}")
        else:
            with open(self.result_log_path, 'a') as f:
                f.write(json.dumps({
                    "episode_id": eid,
                    "success": False,
                    "saved": False,
                    "partial": False,
                    "segments": results,
                    "failure_reason": failure_reason,
                }) + "\n")
            print(f"Episode {eid} Abandoned (no successful waypoint to save).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="path/to/module2_checkpoint")
    parser.add_argument("--m2_server_url", type=str, default="http://127.0.0.1:8082", help="M2 vLLM server URL, e.g. http://127.0.0.1:8080")
    parser.add_argument("--m2_server_model", type=str, default="m2")
    parser.add_argument("--seg_data_path", type=str, default="path/to/r2r_segmentation.json")
    parser.add_argument("--recover_release_coords_path", type=str, default="",
                        help="Optional dense GT coordinates file for recover release distance and tangent heading.")
    parser.add_argument(
        "--use_dense_segment_distance",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use dense GT polyline distance for deviation/recover labels instead of the straight "
            "line from each segment start to end. Requires --recover_release_coords_path; falls back "
            "to straight-line distance for segments without dense coordinates."
        ),
    )
    parser.add_argument("--habitat_config", type=str, default="configs/vln_r2r_dual.yaml")
    parser.add_argument("--dataset_type", type=str, default="R2RVLN-v1")
    parser.add_argument("--dataset_data_path", type=str, default="data/datasets/R2R_VLNCE_v1-3/{split}/{split}.json.gz")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument(
        "--scene_rank_start",
        type=int,
        default=0,
        help=(
            "1-based inclusive start rank after sorting train scenes by scene/scan id. "
            "Use 0 to disable scene slicing."
        ),
    )
    parser.add_argument(
        "--scene_rank_end",
        type=int,
        default=0,
        help=(
            "1-based inclusive end rank after sorting train scenes by scene/scan id. "
            "Use <=0 to select through the last scene."
        ),
    )
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--max_episodes_per_chunk",
        type=int,
        default=50,
        help="Split scenes larger than this into chunks so multiple workers can share the load.",
    )
    parser.add_argument("--output_root", type=str, default="path/to/rewrite_output")
    parser.add_argument(
        "--episode_ids",
        nargs="+",
        default=None,
        help="Replay only these episode ids (for example: --episode_ids 7107 267 29395).",
    )
    parser.add_argument(
        "--save_json_only",
        action="store_true",
        help="Only save rewrite_S.json metadata and skip writing per-frame jpg images.",
    )
    parser.add_argument(
        "--save_attempt_traces",
        action="store_true",
        help=(
            "Write attempt_traces.json beside rewrite_S.json. It preserves every normal-model attempt, "
            "including rejected failures, as world-coordinate polylines for qualitative figures."
        ),
    )
    parser.add_argument(
        "--output_episode_suffix",
        type=str,
        default="",
        help="Append this suffix to per-episode output directories while keeping metadata episode_id unchanged.",
    )
    parser.add_argument("--temperature", type=float, default=0.5, help="LLM decoding temperature")
    parser.add_argument("--max_tokens", type=int, default=32, help="M2 decoding max_tokens")
    parser.add_argument(
        "--max_attempts_per_segment",
        type=int,
        default=2,
        help="Maximum rewrite attempts for each sub-instruction segment before subgoal DAgger expert intervention",
    )
    parser.add_argument("--max_segment_steps", type=int, default=100,
                        help="Fallback frame/action timeout per segment attempt when GT frame metadata is missing")
    parser.add_argument("--segment_timeout_gt_ratio", type=float, default=3.0,
                        help="Use ratio * GT segment frames as timeout when cut_points_details has frame metadata; <=0 disables")
    parser.add_argument("--max_steps_per_segment", type=int, default=160,
                        help="M2 rollout step budget for inline subgoal DAgger after rewrite failure")
    parser.add_argument("--max_oracle_steps", type=int, default=160,
                        help="Oracle fallback step budget for inline subgoal DAgger")
    parser.add_argument(
        "--hide_previous_subtask_memory",
        action="store_true",
        help=(
            "Do not seed each sub-instruction rollout with RGB history from previous completed subtasks. "
            "When set, M2 only sees frames accumulated inside the current subtask."
        ),
    )
    parser.add_argument("--long_segment_min_frames", type=int, default=30,
                        help="segment frame length > this uses long deviation threshold")
    parser.add_argument("--medium_segment_min_frames", type=int, default=10,
                        help="segment frame length > this uses medium deviation threshold")
    parser.add_argument("--deviation_threshold_long", type=float, default=2.0)
    parser.add_argument("--deviation_threshold_medium", type=float, default=1.0)
    parser.add_argument("--deviation_threshold_short", type=float, default=0.5)
    parser.add_argument("--drift_start_threshold_long", type=float, default=0.8)
    parser.add_argument("--drift_start_threshold_medium", type=float, default=0.5)
    parser.add_argument("--drift_start_threshold_short", type=float, default=0.3)
    parser.add_argument("--direct_intervene_distance", type=float, default=0.5)
    parser.add_argument("--precise_success_distance", type=float, default=0.5)
    parser.add_argument("--precise_success_angle", type=float, default=15.0)
    parser.add_argument(
        "--recover_release_gt_distance",
        type=float,
        default=0.0,
        help="Release deviation expert when dist_to_gt_segment is <= this; <=0 uses the segment drift_start_threshold.",
    )
    parser.add_argument(
        "--recover_release_heading_angle",
        type=float,
        default=0.0,
        help="Release deviation expert when heading_error_to_target is <= this; <=0 uses precise_success_angle from the mechanism.",
    )
    parser.add_argument(
        "--recover_release_ignore_heading",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Release recover/deviation expert by GT path distance only, ignoring heading_error_to_target.",
    )
    parser.add_argument(
        "--scene_chunk_round_robin",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Interleave scene chunks in round-robin order so early progress covers more scenes. "
            "Use --no-scene_chunk_round_robin to restore the old per-scene contiguous queue order."
        ),
    )
    parser.add_argument("--ignore_global_finished", action="store_true",
                        help="Ignore the global data/route_rewrite resume scan and only use the current output_root for skipping")
    args = parser.parse_args()
    
    seg_data_list = load_seg_data_list(args.seg_data_path)
    all_seg_ids = [str(item['episode_id']) for item in seg_data_list]

    # Resume from completed episodes recorded below the output root.
    finished_ids = set()
    rewrite_root = "data/route_rewrite"
    if not args.ignore_global_finished and os.path.isdir(rewrite_root):
        for root, _, files in os.walk(rewrite_root):
            if "rewrite_S.json" not in files:
                continue
            meta_path = os.path.join(root, "rewrite_S.json")
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                if "episode_id" in meta:
                    finished_ids.add(str(meta["episode_id"]))
                else:
                    finished_ids.add(os.path.basename(os.path.dirname(os.path.dirname(root))))
            except Exception:
                # Fall back to the directory name if a result file is invalid.
                finished_ids.add(os.path.basename(os.path.dirname(os.path.dirname(root))))

    pending_ids = all_seg_ids
    if args.episode_ids:
        requested_ids = {str(episode_id) for episode_id in args.episode_ids}
        available_ids = set(all_seg_ids)
        unknown_ids = sorted(requested_ids - available_ids)
        if unknown_ids:
            print(f"WARNING: {len(unknown_ids)} requested episode ids are absent from seg_data: {unknown_ids[:10]}")
        pending_ids = [episode_id for episode_id in pending_ids if episode_id in requested_ids]
        print(f"Episode-id filter: requested={len(requested_ids)}, matched={len(pending_ids)}")

    # Optional scene slicing is applied before resume filtering so rank ranges
    # stay stable across independently launched single-GPU services.
    seg_map = {str(item['episode_id']): item for item in seg_data_list}
    if args.scene_rank_start > 0:
        scene_ids = sorted({
            str(seg_map[eid].get("scan_id", ""))
            for eid in pending_ids
            if eid in seg_map
        })
        if not scene_ids:
            raise RuntimeError("No scenes found for scene slicing.")
        if args.scene_rank_end > 0 and args.scene_rank_end < args.scene_rank_start:
            raise ValueError(
                f"--scene_rank_end ({args.scene_rank_end}) must be >= --scene_rank_start ({args.scene_rank_start})"
            )
        start_idx = max(0, args.scene_rank_start - 1)
        end_idx = args.scene_rank_end if args.scene_rank_end > 0 else len(scene_ids)
        end_idx = min(end_idx, len(scene_ids))
        selected_scene_ids = set(scene_ids[start_idx:end_idx])
        if not selected_scene_ids:
            raise RuntimeError(
                f"Scene slice is empty: start={args.scene_rank_start}, end={args.scene_rank_end}, "
                f"total_scenes={len(scene_ids)}"
            )
        before_scene_filter = len(pending_ids)
        pending_ids = [
            eid for eid in pending_ids
            if eid in seg_map and str(seg_map[eid].get("scan_id", "")) in selected_scene_ids
        ]
        selected_min = args.scene_rank_start
        selected_max = end_idx
        print(
            f"Scene slice: ranks {selected_min}-{selected_max} of {len(scene_ids)} sorted train scenes, "
            f"scenes={len(selected_scene_ids)}, episodes={len(pending_ids)}/{before_scene_filter}"
        )
        print(f"Scene slice first/last: {scene_ids[start_idx]} / {scene_ids[min(end_idx, len(scene_ids)) - 1]}")

    output_episode_suffix = str(getattr(args, "output_episode_suffix", "") or "")

    # Filter completed episodes before assigning work to workers.
    pending_ids = [eid for eid in pending_ids if eid not in finished_ids]
    print(f"Scanning {args.output_root} for completed episodes ...", flush=True)
    output_finished = set()
    if os.path.isdir(args.output_root):
        for name in os.listdir(args.output_root):
            if os.path.exists(os.path.join(args.output_root, name, "rgb/rewrite/rewrite_S.json")):
                output_finished.add(name.replace(output_episode_suffix, "") if output_episode_suffix else name)
    pending_ids = [eid for eid in pending_ids if eid not in output_finished]
    print(f"Total Segments: {len(all_seg_ids)}, Finished(Global): {len(finished_ids)}, Pending: {len(pending_ids)}")
    if not pending_ids: exit()

    # Assign scene groups to workers to avoid repeated scene loading.
    pending_set = set(pending_ids)
    scene_groups = OrderedDict()
    for item in seg_data_list:
        eid = str(item['episode_id'])
        if eid not in pending_set:
            continue
        scene_id = item.get("scan_id", "")
        if scene_id not in scene_groups:
            scene_groups[scene_id] = []
        scene_groups[scene_id].append(eid)

    mp.set_start_method('spawn', force=True)
    scene_queue = mp.Queue()
    max_eps = int(getattr(args, "max_episodes_per_chunk", 50))
    queue_items = _build_scene_queue(scene_groups, args.num_workers, max_eps_per_chunk=max_eps)
    for item in queue_items:
        scene_queue.put(item)
    remaining = mp.Value('i', len(queue_items))
    num_active_workers = min(args.num_workers, len(queue_items))
    print(
        f"Hybrid scene queue: scenes={len(scene_groups)}, queue_items={len(queue_items)}, "
        f"workers={num_active_workers}, max_eps_per_chunk={max_eps}",
        flush=True,
    )
    processes = []
    for i in range(num_active_workers):
        p = mp.Process(target=dynamic_scene_worker_fn, args=(i, scene_queue, remaining, args))
        p.start()
        processes.append(p)
    for p in processes: p.join()
