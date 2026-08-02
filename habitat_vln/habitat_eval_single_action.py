"""Evaluate the action-only single-model VLN agent on a Habitat split.

This evaluator intentionally uses one worker: a complete global observation
history is held by the agent for each active episode, matching the training
dataset's global-uniform eight-frame sampling.
"""

import argparse
import copy
import json
import os
import sys
import time

import habitat
from omegaconf import OmegaConf, open_dict, read_write

from habitat.config.default import get_config
from habitat.datasets import make_dataset

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent_single_action import SingleActionAgent
from vln_path_metrics import compute_ndtw, compute_sdtw


def _success_distance(config) -> float:
    success = config.habitat.task.measurements.success
    return float(success.get("success_distance", 3.0) if isinstance(success, dict) else success.success_distance)


def _load_episode_ids(path: str):
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as file:
        if path.endswith(".json"):
            payload = json.load(file)
            if isinstance(payload, dict):
                payload = payload.get("episode_ids", payload.get("episodes", payload.get("ids", [])))
            return {str(item.get("episode_id", item.get("id"))) if isinstance(item, dict) else str(item) for item in payload}
        return {line.strip() for line in file if line.strip() and not line.lstrip().startswith("#")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="")
    parser.add_argument("--model_type", default="qwen2_5_vl")
    parser.add_argument("--server_url", default=None)
    parser.add_argument("--server_model", default="single_action")
    parser.add_argument("--config_path", default="configs/vln_r2r_dual.yaml")
    parser.add_argument("--eval_split", default="val_unseen")
    parser.add_argument("--result_dir", default="eval_results/single_action")
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--max_episode_steps", type=int, default=500)
    parser.add_argument("--action_horizon", type=int, default=3)
    parser.add_argument("--max_tokens", type=int, default=32)
    parser.add_argument("--jpeg_quality", type=int, default=85)
    parser.add_argument("--episode_ids_file", default="")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.model_path and not args.server_url:
        parser.error("Specify --model_path or --server_url.")

    raw_config = get_config(args.config_path)
    config = OmegaConf.create(OmegaConf.to_container(raw_config, resolve=True))
    with open_dict(config), read_write(config):
        config.habitat.dataset.split = args.eval_split
        config.habitat.environment.max_episode_steps = args.max_episode_steps
        if "measurements" not in config.habitat.task:
            config.habitat.task.measurements = {}
        measurements = config.habitat.task.measurements
        measurements.setdefault("distance_to_goal", {"type": "DistanceToGoal", "distance_to": "POINT"})
        measurements.setdefault("success", {"type": "Success", "success_distance": 3.0})
        measurements.setdefault("spl", {"type": "SPL"})

    dataset_template = make_dataset(id_dataset=config.habitat.dataset.type, config=config.habitat.dataset)
    requested_ids = _load_episode_ids(args.episode_ids_file)
    episodes = [episode for episode in dataset_template.episodes if requested_ids is None or str(episode.episode_id) in requested_ids]
    episodes.sort(key=lambda episode: (episode.scene_id, str(episode.episode_id)))
    output_root = os.path.join(args.result_dir, args.exp_name, args.eval_split)
    os.makedirs(output_root, exist_ok=True)

    agent = SingleActionAgent(
        model_path=args.model_path,
        model_type=args.model_type,
        action_horizon=args.action_horizon,
        max_tokens=args.max_tokens,
        jpeg_quality=args.jpeg_quality,
        server_url=args.server_url,
        server_model=args.server_model,
    )
    print(f"Evaluating {len(episodes)} episodes with action-only global-history sampling.")
    try:
        for index, episode in enumerate(episodes, start=1):
            episode_id = str(episode.episode_id)
            output_dir = os.path.join(output_root, episode_id)
            output_path = os.path.join(output_dir, f"{episode_id}.json")
            if os.path.exists(output_path) and not args.overwrite:
                continue
            os.makedirs(output_dir, exist_ok=True)

            dataset = copy.copy(dataset_template)
            dataset.episodes = [episode]
            env = habitat.Env(config=config, dataset=dataset)
            try:
                trajectory_start = time.perf_counter()
                observations = env.reset()
                agent.reset()
                trajectory_positions = [env.sim.get_agent_state().position.tolist()]
                steps = []
                model_request_seconds = []
                agent_act_seconds = 0.0
                env_step_seconds = 0.0
                for step_index in range(args.max_episode_steps):
                    act_start = time.perf_counter()
                    action_data = agent.act(observations, {"episode_id": episode_id})
                    act_elapsed = time.perf_counter() - act_start
                    agent_act_seconds += act_elapsed
                    action = int(action_data["action"])
                    state = env.sim.get_agent_state()
                    step_record = {
                        "step": step_index,
                        "action": action,
                        "position": state.position.tolist(),
                        "agent_act_seconds": act_elapsed,
                    }
                    if agent.just_predicted:
                        step_record["action_prediction"] = agent.last_action_str
                        step_record["prompt_frame_indices"] = agent.last_frame_indices
                        step_record["model_request_seconds"] = act_elapsed
                        model_request_seconds.append(act_elapsed)
                    steps.append(step_record)
                    env_step_start = time.perf_counter()
                    observations = env.step(action)
                    env_step_seconds += time.perf_counter() - env_step_start
                    trajectory_positions.append(env.sim.get_agent_state().position.tolist())
                    if action == 0:
                        break

                metrics = env.get_metrics()
                success = float(metrics.get("success", 0.0))
                ndtw = compute_ndtw(
                    env.sim,
                    trajectory_positions,
                    getattr(episode, "reference_path", []),
                    success_distance=_success_distance(config),
                    episode=episode,
                )
                result = {
                    "episode_id": episode_id,
                    "scene_id": episode.scene_id,
                    "instruction": observations["instruction"]["text"],
                    "steps": steps,
                    "trajectory_positions": trajectory_positions,
                    "metrics": {
                        "success": success,
                        "spl": float(metrics.get("spl", 0.0)),
                        "distance_to_goal": float(metrics.get("distance_to_goal", 0.0)),
                        "ndtw": ndtw,
                        "sdtw": compute_sdtw(success, ndtw),
                    },
                    "timing": {
                        "trajectory_wall_seconds": time.perf_counter() - trajectory_start,
                        "model_request_count": len(model_request_seconds),
                        "model_request_total_seconds": sum(model_request_seconds),
                        "model_request_mean_seconds": (
                            sum(model_request_seconds) / len(model_request_seconds) if model_request_seconds else 0.0
                        ),
                        "model_request_max_seconds": max(model_request_seconds, default=0.0),
                        "agent_act_total_seconds": agent_act_seconds,
                        "env_step_total_seconds": env_step_seconds,
                    },
                }
                with open(output_path, "w", encoding="utf-8") as file:
                    json.dump(result, file, ensure_ascii=False, indent=2)
                print(f"[{index}/{len(episodes)}] {episode_id}: SR={success:.0f}")
            finally:
                env.close()
    finally:
        agent.close()


if __name__ == "__main__":
    main()
