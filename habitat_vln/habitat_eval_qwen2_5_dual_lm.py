import os
import sys
import torch
import json
import gzip
import math
import copy
import argparse
import multiprocessing as mp
from omegaconf import OmegaConf, open_dict, read_write

# Add project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import habitat
from habitat.config.default import get_config
from habitat.datasets import make_dataset
from agent_dual_qwen2_5_lm import DualReason_Agent
from habitat_vln.route2step_vln_dataset import Route2StepVLNDatasetV1  # noqa: F401
from vln_path_metrics import compute_ndtw, compute_sdtw


def _parse_language_filter(raw_langs):
    if not raw_langs:
        return []
    return [x.strip().lower() for x in raw_langs.split(",") if x.strip()]


def _lang_match(lang, targets):
    if not targets:
        return True
    lang = (lang or "").strip().lower()
    for t in targets:
        if t == "en":
            if lang == "en" or lang.startswith("en-"):
                return True
        elif lang == t:
            return True
    return False


def _get_success_distance(config):
    try:
        success_cfg = config.habitat.task.measurements.success
        if isinstance(success_cfg, dict):
            return float(success_cfg.get("success_distance", 3.0))
        return float(getattr(success_cfg, "success_distance", 3.0))
    except Exception:
        return 3.0


def _load_episode_ids_by_language(dataset_path, raw_langs):
    targets = _parse_language_filter(raw_langs)
    if not targets:
        return None, 0, 0

    with gzip.open(dataset_path, "rt", encoding="utf-8") as f:
        payload = json.load(f)
    episodes = payload.get("episodes", [])

    matched_ids = set()
    for ep in episodes:
        ep_id = ep.get("episode_id", None)
        if ep_id is None:
            continue
        inst = ep.get("instruction", {})
        lang = inst.get("language", "") if isinstance(inst, dict) else ""
        if _lang_match(lang, targets):
            matched_ids.add(str(ep_id))

    return matched_ids, len(episodes), len(matched_ids)


def _load_episode_ids_file(path):
    if not path:
        return None

    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".json"):
            payload = json.load(f)
        else:
            return {line.strip() for line in f if line.strip() and not line.lstrip().startswith("#")}

    if isinstance(payload, dict):
        for key in ("episode_ids", "episodes", "ids"):
            if key in payload:
                payload = payload[key]
                break
        else:
            raise ValueError(f"Episode id file {path} must contain episode_ids, episodes, or ids.")

    if not isinstance(payload, list):
        raise ValueError(f"Episode id file {path} must be a JSON list or a text file.")

    episode_ids = set()
    for item in payload:
        if isinstance(item, dict):
            item = item.get("episode_id", item.get("id"))
        if item is None:
            continue
        episode_ids.add(str(item))
    return episode_ids


def _infer_rxr_follower_paths(dataset_path):
    if not dataset_path.endswith("_guide.json.gz"):
        return "", ""
    follower_path = dataset_path.replace("_guide.json.gz", "_follower.json.gz")
    follower_gt_path = dataset_path.replace("_guide.json.gz", "_follower_gt.json.gz")
    return follower_path, follower_gt_path


def _load_rxr_dynamic_max_steps(dataset_path, raw_langs, multiplier):
    if multiplier <= 0:
        return {}, {}

    follower_path, follower_gt_path = _infer_rxr_follower_paths(dataset_path)
    if not follower_path or not follower_gt_path:
        return {}, {}
    if not os.path.exists(follower_path) or not os.path.exists(follower_gt_path):
        return {}, {}

    targets = _parse_language_filter(raw_langs)

    with gzip.open(dataset_path, "rt", encoding="utf-8") as f:
        guide_payload = json.load(f)
    with gzip.open(follower_path, "rt", encoding="utf-8") as f:
        follower_payload = json.load(f)
    with gzip.open(follower_gt_path, "rt", encoding="utf-8") as f:
        follower_gt_payload = json.load(f)

    guide_episodes = guide_payload.get("episodes", [])
    follower_episodes = follower_payload.get("episodes", [])

    instruction_to_follower_ep = {}
    follower_path_fallback = {}
    for ep in follower_episodes:
        ep_id = str(ep.get("episode_id", ""))
        inst = ep.get("instruction", {})
        inst_id = str(inst.get("instruction_id", "")) if isinstance(inst, dict) else ""
        if not ep_id or not inst_id:
            continue
        instruction_to_follower_ep[inst_id] = ep_id
        reference_path = ep.get("reference_path", [])
        if isinstance(reference_path, list) and reference_path:
            follower_path_fallback[ep_id] = len(reference_path)

    dynamic_max_steps = {}
    stats = {
        "guide_episodes": len(guide_episodes),
        "follower_episodes": len(follower_episodes),
        "mapped_episodes": 0,
        "used_gt_actions": 0,
        "used_gt_forward_steps": 0,
        "used_gt_locations": 0,
        "used_follower_reference_path": 0,
    }
    for ep in guide_episodes:
        inst = ep.get("instruction", {})
        lang = inst.get("language", "") if isinstance(inst, dict) else ""
        if not _lang_match(lang, targets):
            continue

        guide_ep_id = str(ep.get("episode_id", ""))
        inst_id = str(inst.get("instruction_id", "")) if isinstance(inst, dict) else ""
        if not guide_ep_id or not inst_id:
            continue

        follower_ep_id = instruction_to_follower_ep.get(inst_id, "")
        if not follower_ep_id:
            continue

        gt_item = follower_gt_payload.get(follower_ep_id, {})
        base_steps = 0
        if isinstance(gt_item, dict):
            actions = gt_item.get("actions", [])
            if isinstance(actions, list) and actions:
                base_steps = len(actions)
                stats["used_gt_actions"] += 1
            else:
                forward_steps = gt_item.get("forward_steps", 0)
                if isinstance(forward_steps, int) and forward_steps > 0:
                    base_steps = forward_steps
                    stats["used_gt_forward_steps"] += 1
                else:
                    locations = gt_item.get("locations", [])
                    if isinstance(locations, list) and len(locations) > 1:
                        base_steps = len(locations) - 1
                        stats["used_gt_locations"] += 1

        if base_steps <= 0:
            base_steps = follower_path_fallback.get(follower_ep_id, 0)
            if base_steps > 0:
                stats["used_follower_reference_path"] += 1

        if base_steps <= 0:
            continue

        dynamic_max_steps[guide_ep_id] = max(200, int(math.ceil(base_steps * multiplier)))
        stats["mapped_episodes"] += 1

    return dynamic_max_steps, stats


def _get_visible_cuda_devices():
    raw_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if raw_visible:
        return [token.strip() for token in raw_visible.split(",") if token.strip()]

    if torch.cuda.is_available():
        return [str(i) for i in range(torch.cuda.device_count())]

    return []


def _run_episode_batch(
    worker_id,
    scene_id,
    episode_ids,
    episode_lookup,
    dataset_template,
    config,
    agent,
    split_dir,
    need_step_metrics,
    max_episode_steps,
    dynamic_max_steps_by_episode_id,
    stuck_timeout_steps,
    stuck_distance_threshold,
):
    dataset = copy.copy(dataset_template)
    dataset.episodes = [episode_lookup[eid] for eid in episode_ids if eid in episode_lookup]
    dataset.episodes = sorted(dataset.episodes, key=lambda ep: (ep.scene_id, int(ep.episode_id)))

    if len(dataset.episodes) == 0:
        return

    print(
        f"Worker {worker_id} processing scene chunk {os.path.basename(scene_id)} "
        f"({len(dataset.episodes)} episodes)"
    )

    env = habitat.Env(config=config, dataset=dataset)
    try:
        for _ in range(len(env.episodes)):
            observations = env.reset()
            episode = env.current_episode
            episode_id = str(episode.episode_id)

            episode_dir = os.path.join(split_dir, episode_id)
            os.makedirs(episode_dir, exist_ok=True)
            agent.result_path = episode_dir
            agent.reset()
            agent.episode_id = episode_id

            episode_log = {
                "episode_id": episode_id,
                "scene_id": episode.scene_id,
                "instruction": observations["instruction"]["text"],
                "steps": [],
                "metrics": {},
                "timeout": False,
                "timeout_reason": "",
            }

            episode_max_steps = dynamic_max_steps_by_episode_id.get(episode_id, max_episode_steps)
            episode_log["max_episode_steps"] = int(episode_max_steps)
            trajectory_positions = [env.sim.get_agent_state().position.tolist()]

            step_count = 0
            terminated = False
            stuck_steps = 0
            last_position = None
            while not terminated and step_count < episode_max_steps:
                info = env.get_metrics() if need_step_metrics else {}
                info["episode_id"] = episode_id
                action_data = agent.act(observations, info)
                action = action_data["action"]

                curr_state = env.sim.get_agent_state()
                curr_position = curr_state.position.tolist()
                if last_position is not None:
                    move_dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(curr_position, last_position)))
                    if move_dist <= stuck_distance_threshold:
                        stuck_steps += 1
                    else:
                        stuck_steps = 0
                last_position = curr_position

                step_info = {
                    "step": step_count,
                    "action": action,
                    "position": curr_position,
                    "rotation": [float(x) for x in curr_state.rotation.components],
                    "stuck_steps": stuck_steps,
                }
                if getattr(agent, "just_predicted", False):
                    step_info["reasoning"] = getattr(agent, "last_reasoning", "")
                    step_info["m2_sub_instruction"] = getattr(agent, "current_m2_sub_instruction", "")
                    step_info["action_prediction"] = getattr(agent, "last_action_str", "")

                episode_log["steps"].append(step_info)
                observations = env.step(action)
                trajectory_positions.append(env.sim.get_agent_state().position.tolist())
                step_count += 1
                if action == 0:
                    terminated = True
                elif stuck_timeout_steps > 0 and stuck_steps >= stuck_timeout_steps:
                    terminated = True
                    episode_log["timeout"] = True
                    episode_log["timeout_reason"] = "stuck"

            if not terminated and step_count >= episode_max_steps:
                episode_log["timeout"] = True
                episode_log["timeout_reason"] = "max_steps"

            episode_log["trajectory_positions"] = trajectory_positions
            agent.reset()

            metrics = env.get_metrics()
            success = float(metrics.get("success", 0.0))
            ndtw = compute_ndtw(
                env.sim,
                trajectory_positions,
                getattr(episode, "reference_path", []),
                success_distance=_get_success_distance(config),
                episode=episode,
            )
            episode_log["metrics"] = {
                "success": success,
                "spl": float(metrics.get("spl", 0.0)),
                "distance_to_goal": float(metrics.get("distance_to_goal", 0.0)),
                "ndtw": ndtw,
                "sdtw": compute_sdtw(success, ndtw),
            }

            with open(os.path.join(episode_dir, f"{episode_id}.json"), "w") as f:
                json.dump(episode_log, f, indent=4, ensure_ascii=False)

            print(f"Worker {worker_id} | Episode {episode_id} Done. SR: {episode_log['metrics']['success']}")
    finally:
        env.close()


def eval_worker(worker_id, task_queue, args, logical_gpu_id, physical_gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_id)

    raw_config = get_config(args.config_path)
    config = OmegaConf.create(OmegaConf.to_container(raw_config, resolve=True))

    with open_dict(config), read_write(config):
        config.habitat.dataset.split = args.eval_split
        config.habitat.environment.max_episode_steps = getattr(
            args, "dynamic_env_max_episode_steps", args.max_episode_steps
        )
        config.habitat.simulator.habitat_sim_v0.gpu_device_id = 0
        if "measurements" not in config.habitat.task:
            config.habitat.task.measurements = {}

        m = config.habitat.task.measurements
        if "distance_to_goal" not in m:
            m["distance_to_goal"] = {"type": "DistanceToGoal", "distance_to": "POINT"}
        if "success" not in m:
            m["success"] = {"type": "Success", "success_distance": 3.0}
        if "spl" not in m:
            m["spl"] = {"type": "SPL"}
        if args.save_video:
            if "top_down_map" not in m:
                m["top_down_map"] = {
                    "type": "TopDownMap",
                    "max_episode_steps": getattr(args, "dynamic_env_max_episode_steps", args.max_episode_steps),
                    "map_padding": 3,
                    "map_resolution": 1024,
                    "draw_source": True,
                    "draw_border": True,
                    "draw_shortest_path": True,
                    "draw_view_points": True,
                    "draw_goal_positions": True,
                    "draw_goal_aabbs": True,
                    "fog_of_war": {"draw": True, "visibility_dist": 5.0, "fov": 79},
                }
        else:
            m.pop("top_down_map", None)
            m.pop("top_down_map_vlnce", None)

    split_dir = os.path.join(args.result_dir, args.exp_name, args.eval_split)
    dataset_template = make_dataset(id_dataset=config.habitat.dataset.type, config=config.habitat.dataset)
    episode_lookup = {str(ep.episode_id): ep for ep in dataset_template.episodes}

    agent = DualReason_Agent(
        model1_path=args.model1_path,
        model2_path=args.model2_path,
        result_path=split_dir,
        require_map=args.save_video,
        model_type=args.model_type,
        action_horizon=args.action_horizon,
        m1_max_tokens=args.m1_max_tokens,
        m2_max_tokens=args.m2_max_tokens,
        m1_history_frames=args.m1_history_frames,
        m1_current_frames=args.m1_current_frames,
        m1_history_window=args.m1_history_window,
        m1_turn_aware_sampling=args.enable_m1_turn_aware_sampling,
        m1_turn_frame_budget=args.m1_turn_frame_budget,
        enable_m1_recursive_split=args.enable_m1_recursive_split,
        m2_recent_window=args.m2_recent_window,
        m2_num_frames=args.m2_num_frames,
        jpeg_quality=args.jpeg_quality,
        http_timeout_s=args.http_timeout_s,
        http_max_connections=args.http_max_connections,
        http_max_keepalive_connections=args.http_max_keepalive_connections,
        http_retry_count=args.http_retry_count,
        http_retry_backoff_s=args.http_retry_backoff_s,
        http_retry_max_backoff_s=args.http_retry_max_backoff_s,
        http_force_connection_close=args.http_force_connection_close,
        enable_image_data_url_cache=not args.disable_image_data_url_cache,
        image_data_url_cache_size=args.image_data_url_cache_size,
        stop_on_m1_stop_token=not args.disable_m1_stop_token_terminate,
        enable_align_turn_rule=args.enable_align_turn_rule,
        filter_align_memory_for_m2=args.filter_align_memory_for_m2,
        strip_m1_recovering_prefix_for_m2=args.strip_m1_recovering_prefix_for_m2,
        m2_recover_max_consecutive=args.m2_recover_max_consecutive,
        m2_recover_cooldown=args.m2_recover_cooldown,
        use_single_model=args.use_single_model,
        m1_server_url=getattr(args, "m1_server_url", None),
        m1_server_model=getattr(args, "m1_server_model", "m1"),
        m2_server_url=getattr(args, "m2_server_url", None),
        m2_server_model=getattr(args, "m2_server_model", "m2"),
    )

    print(
        f"Worker {worker_id} started on logical GPU {logical_gpu_id} "
        f"(physical GPU {physical_gpu_id}, worker CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']})"
    )

    while True:
        task = task_queue.get()
        if task is None:
            print(f"Worker {worker_id} received stop signal.")
            break
        scene_id, episode_ids = task
        print(f"Worker {worker_id} picked scene {os.path.basename(scene_id)} ({len(episode_ids)} episodes)")
        _run_episode_batch(
            worker_id,
            scene_id,
            episode_ids,
            episode_lookup,
            dataset_template,
            config,
            agent,
            split_dir,
            need_step_metrics=args.save_video,
            max_episode_steps=args.max_episode_steps,
            dynamic_max_steps_by_episode_id=getattr(args, "dynamic_max_steps_by_episode_id", {}),
            stuck_timeout_steps=args.stuck_timeout_steps,
            stuck_distance_threshold=args.stuck_distance_threshold,
        )

    close_fn = getattr(agent, "close", None)
    if callable(close_fn):
        close_fn()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model1_path", type=str, required=True)
    parser.add_argument("--model2_path", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="qwen2_5_vl")
    parser.add_argument("--eval_split", type=str, default="val_unseen")
    parser.add_argument("--config_path", type=str, default="configs/vln_r2r_dual.yaml")
    parser.add_argument("--result_dir", type=str, default="eval_results/qwen2_5")
    parser.add_argument("--exp_name", type=str, default="qwen2_5_dual")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--action_horizon", type=int, default=3)
    parser.add_argument("--m1_max_tokens", type=int, default=256)
    parser.add_argument("--m2_max_tokens", type=int, default=32)
    parser.add_argument("--m1_history_frames", type=int, default=13)
    parser.add_argument("--m1_current_frames", type=int, default=3)
    parser.add_argument("--m1_history_window", type=int, default=0, help="If > 0, sample M1 history only from the most recent N frames; 0 keeps global history.")
    parser.add_argument("--enable_m1_turn_aware_sampling", action="store_true")
    parser.add_argument("--m1_turn_frame_budget", type=int, default=3)
    parser.add_argument(
        "--enable_m1_recursive_split",
        action="store_true",
        help="If the first non-recovery M1 answer exceeds 30 characters, re-segment it once before calling M2.",
    )
    parser.add_argument("--m2_recent_window", type=int, default=40)
    parser.add_argument("--m2_num_frames", type=int, default=8)
    parser.add_argument("--jpeg_quality", type=int, default=95)
    parser.add_argument(
        "--http_timeout_s",
        type=float,
        default=120.0,
        help="Per-request vLLM HTTP timeout in seconds; <= 0 disables the timeout.",
    )
    parser.add_argument("--http_max_connections", type=int, default=32)
    parser.add_argument("--http_max_keepalive_connections", type=int, default=8)
    parser.add_argument(
        "--http_retry_count",
        type=int,
        default=3,
        help="Retries after transient vLLM failures; -1 retries indefinitely.",
    )
    parser.add_argument("--http_retry_backoff_s", type=float, default=2.0)
    parser.add_argument("--http_retry_max_backoff_s", type=float, default=8.0)
    parser.add_argument("--http_force_connection_close", action="store_true")
    parser.add_argument("--disable_image_data_url_cache", action="store_true")
    parser.add_argument("--image_data_url_cache_size", type=int, default=160)
    parser.add_argument("--disable_m1_stop_token_terminate", action="store_true")
    parser.add_argument("--enable_align_turn_rule", action="store_true")
    parser.add_argument("--filter_align_memory_for_m2", action="store_true")
    parser.add_argument(
        "--strip_m1_recovering_prefix_for_m2",
        action="store_true",
        help="If set, strip a leading 'Recovering:' prefix from the M1 answer before passing it to M2.",
    )
    parser.add_argument(
        "--m2_recover_max_consecutive",
        type=int,
        default=0,
        help="If > 0, allow at most this many consecutive M2 Recovering decisions before cooldown.",
    )
    parser.add_argument(
        "--m2_recover_cooldown",
        type=int,
        default=0,
        help="If > 0, number of M1/M2 decisions to strip Recovering after the consecutive limit is reached.",
    )
    parser.add_argument("--use_single_model", action="store_true")
    parser.add_argument("--max_episode_steps", type=int, default=500)
    parser.add_argument(
        "--rxr_dynamic_max_steps_multiplier",
        type=float,
        default=0.0,
        help="If > 0, use multiplier * RxR follower_gt action count as a per-episode max step budget.",
    )
    parser.add_argument("--stuck_timeout_steps", type=int, default=0)
    parser.add_argument("--stuck_distance_threshold", type=float, default=1e-3)
    parser.add_argument(
        "--rxr_language",
        type=str,
        default="",
        help="Comma-separated language filter for RxR json (e.g. en, en-US, hi-IN). 'en' matches en-*.",
    )
    parser.add_argument(
        "--episode_ids_file",
        type=str,
        default="",
        help="Optional JSON/text file of episode ids to evaluate, used for quick subsets.",
    )
    parser.add_argument(
        "--task_multiplier",
        type=int,
        default=4,
        help="Target number of scene-local tasks per worker for dynamic backfill scheduling.",
    )
    parser.add_argument("--m1_server_url", type=str, default=None)
    parser.add_argument("--m1_server_model", type=str, default="m1")
    parser.add_argument("--m2_server_url", type=str, default=None)
    parser.add_argument("--m2_server_model", type=str, default="m2")
    args = parser.parse_args()
    if (args.m2_recover_max_consecutive > 0) != (args.m2_recover_cooldown > 0):
        parser.error(
            "--m2_recover_max_consecutive and --m2_recover_cooldown must both be > 0 to enable recover limiting, "
            "or both be 0 to disable it."
        )

    print(
        "Agent eval config: "
        f"action_horizon={args.action_horizon}, "
        f"m1_max_tokens={args.m1_max_tokens}, "
        f"m2_max_tokens={args.m2_max_tokens}, "
        f"m1_recursive_split={args.enable_m1_recursive_split}, "
        f"http_timeout_s={args.http_timeout_s}, "
        f"http_retry_count={args.http_retry_count}, "
        f"http_keepalive={args.http_max_keepalive_connections}, "
        f"force_connection_close={args.http_force_connection_close}, "
        f"save_video={args.save_video}, "
        f"m1_server={bool(args.m1_server_url)}, "
        f"m2_server={bool(args.m2_server_url)}, "
        f"strip_m1_recovering_prefix_for_m2={args.strip_m1_recovering_prefix_for_m2}, "
        f"m2_recover_max_consecutive={args.m2_recover_max_consecutive}, "
        f"m2_recover_cooldown={args.m2_recover_cooldown}"
    )

    mp.set_start_method("spawn", force=True)

    split_dir = os.path.join(args.result_dir, args.exp_name, args.eval_split)
    os.makedirs(split_dir, exist_ok=True)

    tmp_config = get_config(args.config_path)
    config = OmegaConf.create(OmegaConf.to_container(tmp_config, resolve=True))
    with open_dict(config), read_write(config):
        config.habitat.dataset.split = args.eval_split

    dataset_path = config.habitat.dataset.data_path.format(split=config.habitat.dataset.split)
    args.dynamic_max_steps_by_episode_id = {}
    args.dynamic_env_max_episode_steps = args.max_episode_steps
    if "/rxr/" in dataset_path or dataset_path.endswith("_guide.json.gz"):
        if args.rxr_dynamic_max_steps_multiplier > 0:
            dynamic_max_steps_by_episode_id, dynamic_stats = _load_rxr_dynamic_max_steps(
                dataset_path,
                args.rxr_language,
                args.rxr_dynamic_max_steps_multiplier,
            )
            args.dynamic_max_steps_by_episode_id = dynamic_max_steps_by_episode_id
            if dynamic_max_steps_by_episode_id:
                dynamic_values = list(dynamic_max_steps_by_episode_id.values())
                args.dynamic_env_max_episode_steps = max(args.max_episode_steps, max(dynamic_values))
                print(
                    "Loaded RxR dynamic max steps: "
                    f"mapped={dynamic_stats['mapped_episodes']}, "
                    f"min={min(dynamic_values)}, "
                    f"max={max(dynamic_values)}, "
                    f"avg={sum(dynamic_values) / len(dynamic_values):.1f}, "
                    f"env_max={args.dynamic_env_max_episode_steps}, "
                    f"source_actions={dynamic_stats['used_gt_actions']}, "
                    f"source_forward_steps={dynamic_stats['used_gt_forward_steps']}, "
                    f"source_locations={dynamic_stats['used_gt_locations']}, "
                    f"source_reference_path={dynamic_stats['used_follower_reference_path']}"
                )
            else:
                print(
                    "RxR dynamic max steps requested but no follower/follower_gt mapping was found. "
                    f"Falling back to fixed max_episode_steps={args.max_episode_steps}."
                )

    tmp_config = config

    dataset = make_dataset(id_dataset=tmp_config.habitat.dataset.type, config=tmp_config.habitat.dataset)
    all_ep_ids = [str(ep.episode_id) for ep in dataset.episodes]

    if args.rxr_language:
        dataset_path = tmp_config.habitat.dataset.data_path.format(split=tmp_config.habitat.dataset.split)
        lang_ep_ids, total_eps_in_json, matched_eps = _load_episode_ids_by_language(
            dataset_path, args.rxr_language
        )
        before_filter = len(all_ep_ids)
        all_ep_ids = [ep_id for ep_id in all_ep_ids if ep_id in lang_ep_ids]
        print(
            f"Language filter [{args.rxr_language}] on {dataset_path}: "
            f"json episodes={total_eps_in_json}, matched={matched_eps}, "
            f"habitat_loaded={before_filter}, after_filter={len(all_ep_ids)}"
        )

    requested_ep_ids = _load_episode_ids_file(args.episode_ids_file)
    if requested_ep_ids is not None:
        before_filter = len(all_ep_ids)
        available_ep_ids = set(all_ep_ids)
        missing_ep_ids = sorted(
            requested_ep_ids - available_ep_ids,
            key=lambda value: int(value) if value.isdigit() else value,
        )
        all_ep_ids = [ep_id for ep_id in all_ep_ids if ep_id in requested_ep_ids]
        print(
            f"Episode id filter [{args.episode_ids_file}]: "
            f"requested={len(requested_ep_ids)}, matched={len(all_ep_ids)}, "
            f"missing={len(missing_ep_ids)}, before_filter={before_filter}"
        )
        if missing_ep_ids:
            preview = ",".join(missing_ep_ids[:10])
            suffix = "..." if len(missing_ep_ids) > 10 else ""
            print(f"Missing requested episode ids: {preview}{suffix}")

    pending_ids = []
    for ep_id in all_ep_ids:
        json_path = os.path.join(split_dir, ep_id, f"{ep_id}.json")
        if not os.path.exists(json_path):
            pending_ids.append(ep_id)

    print(f"Total: {len(all_ep_ids)}, Pending: {len(pending_ids)}, Workers: {args.num_workers}")
    if not pending_ids:
        print("All done.")
        return

    pending_set = set(pending_ids)
    scene_to_ids = {}
    for ep in dataset.episodes:
        ep_id = str(ep.episode_id)
        if ep_id in pending_set:
            scene = ep.scene_id
            if scene not in scene_to_ids:
                scene_to_ids[scene] = []
            scene_to_ids[scene].append(ep_id)

    for ep_ids in scene_to_ids.values():
        ep_ids.sort(key=int)

    def split_scene_into_chunks(scene_id, ep_ids, target_chunk_size):
        if target_chunk_size <= 0 or len(ep_ids) <= target_chunk_size:
            return [(scene_id, list(ep_ids))]
        chunks_local = []
        for start in range(0, len(ep_ids), target_chunk_size):
            chunks_local.append((scene_id, list(ep_ids[start:start + target_chunk_size])))
        return chunks_local

    def build_scene_tasks(scene_to_ids_map, num_workers, task_multiplier):
        scene_items = sorted(scene_to_ids_map.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        if not scene_items or num_workers <= 0:
            return []

        target_chunk_size = max(1, math.ceil(len(pending_ids) / max(num_workers * max(task_multiplier, 1), 1)))
        tasks_local = []
        for scene_id, ep_ids in scene_items:
            tasks_local.extend(split_scene_into_chunks(scene_id, ep_ids, target_chunk_size))

        tasks_local.sort(key=lambda item: (-len(item[1]), item[0], int(item[1][0])))
        return tasks_local

    tasks = build_scene_tasks(scene_to_ids, args.num_workers, args.task_multiplier)
    print(
        f"Built {len(tasks)} scene-aware tasks from {len(scene_to_ids)} scenes "
        f"(avg {len(pending_ids) / max(len(tasks), 1):.1f} eps/task)"
    )

    visible_cuda_devices = _get_visible_cuda_devices()
    num_visible_gpus = len(visible_cuda_devices)
    if num_visible_gpus <= 0:
        print("Warning: No CUDA device visible. Falling back to logical GPU 0 mapping.")
        visible_cuda_devices = ["0"]
        num_visible_gpus = 1
    else:
        print(f"Visible CUDA devices for workers: {visible_cuda_devices}")

    task_queue = mp.Queue()
    for task in tasks:
        task_queue.put(task)
    for _ in range(args.num_workers):
        task_queue.put(None)

    processes = []
    for i in range(args.num_workers):
        logical_gpu_id = i % num_visible_gpus
        physical_gpu_id = visible_cuda_devices[logical_gpu_id]
        print(f"Starting Worker {i} on logical GPU {logical_gpu_id} (physical GPU {physical_gpu_id})...")
        p = mp.Process(target=eval_worker, args=(i, task_queue, args, logical_gpu_id, physical_gpu_id))
        p.start()
        print(f"Worker {i} started (PID: {p.pid})")
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
