#!/usr/bin/env python3
"""
Add expert_chunk labels to existing rewrite trajectory data.

For each step_record in rewrite_S.json, sets the Habitat agent to the recorded
position/rotation, then queries expert_chunk (H oracle actions toward the
segment's waypoint).  Writes expert_chunk back into step_records in-place.

Usage:
  PYTHONPATH=. python DAgger/add_expert_chunk.py \
      --input_root path/to/rewrite_data \
      --seg_data_path path/to/r2r_segmentation.json \
      --habitat_config configs/vln_r2r_dual.yaml \
      --action_horizon 3 --gpu 0
"""

import os
import sys
import json
import argparse
import multiprocessing as mp

import numpy as np
import quaternion
from tqdm import tqdm

import habitat
from habitat.config.default import get_config
from omegaconf import OmegaConf
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.sims.habitat_simulator.actions import HabitatSimActions

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

ACTION_STOP = 0
ACTION_FORWARD = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3

HABITAT_TO_ID = {
    HabitatSimActions.stop: ACTION_STOP,
    HabitatSimActions.move_forward: ACTION_FORWARD,
    HabitatSimActions.turn_left: ACTION_LEFT,
    HabitatSimActions.turn_right: ACTION_RIGHT,
}

if not hasattr(np, 'float'): np.float = float
if not hasattr(np, 'int'): np.int = int
os.environ.setdefault('MAX_PIXELS', '1003520')
os.environ.setdefault('IMAGE_MAX_TOKEN_NUM', '2048')


# ──────────────────────────────────────────────────────────────────────
# Seg data loading
# ──────────────────────────────────────────────────────────────────────

def load_seg_lookup(seg_data_path):
    with open(seg_data_path, 'r') as f:
        raw = json.load(f)
    items = raw.values() if isinstance(raw, dict) else raw
    lookup = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        eid = str(item.get('episode_id', ''))
        if not eid:
            continue
        subs = item.get('sub_instructions') or item.get('split_instructions') or []
        if isinstance(subs, dict):
            subs = [subs[k].lstrip('0123456789. ') for k in sorted(subs.keys(), key=int)]
        elif isinstance(subs, list):
            subs = [s.lstrip('0123456789. ') for s in subs]
        cps = item.get('cut_points_details') or item.get('cut_points') or []
        if isinstance(cps, dict):
            cps = [cps[k] for k in sorted(cps.keys(), key=int)]
        lookup[eid] = {
            'sub_instructions': subs,
            'waypoints': cps,
            'scene_id': str(item.get('scan_id', item.get('scene_id', ''))),
        }
    return lookup


# ──────────────────────────────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────────────────────────────

def _quat_yaw(q):
    return np.arctan2(2 * (q.w * q.y + q.x * q.z), 1 - 2 * (q.y ** 2 + q.z ** 2))


def _angle_diff_deg(current_rot, target_rot_q):
    diff = np.degrees(_quat_yaw(current_rot) - _quat_yaw(target_rot_q))
    while diff > 180: diff -= 360
    while diff < -180: diff += 360
    return diff


def _check_waypoint(pos, rot, target_info, is_final, sd, sh):
    t_pos = np.array(target_info['position'])
    dist = float(np.linalg.norm(pos - t_pos))
    t_rot = quaternion.quaternion(
        target_info['rotation']['w'], target_info['rotation']['x'],
        target_info['rotation']['y'], target_info['rotation']['z'],
    )
    angle = abs(float(_angle_diff_deg(rot, t_rot)))
    if is_final:
        return dist <= sd, dist, angle
    return (dist <= sd and angle <= sh), dist, angle


# ──────────────────────────────────────────────────────────────────────
# Expert chunk query (same semantics as TraditionalDaggerCollector)
# ──────────────────────────────────────────────────────────────────────

def _snap_to_navmesh(sim, pos):
    """Snap a 3D position to the closest point on the navmesh.  Returns
    the snapped position, or None if snapping fails."""
    try:
        snapped = sim.pathfinder.snap_point(np.array(pos, dtype=np.float32))
        if snapped is not None and np.all(np.isfinite(snapped)):
            return snapped.tolist()
    except Exception:
        pass
    return None


def query_expert_chunk(sim, agent_pos, agent_rot_list, target_info, horizon,
                       is_final, sd, sh):
    """Set agent to (pos, rot), then save→step H oracle→restore."""
    # Snap to navmesh — rewrite positions may drift off-mesh
    snapped = _snap_to_navmesh(sim, agent_pos)
    if snapped is None:
        return None
    sim.set_agent_state(snapped, agent_rot_list)

    saved_state = sim.get_agent_state()
    saved_pos = np.array(saved_state.position, dtype=np.float64).copy()
    saved_rot = saved_state.rotation

    target_rot_q = quaternion.quaternion(
        target_info['rotation']['w'], target_info['rotation']['x'],
        target_info['rotation']['y'], target_info['rotation']['z'],
    )
    follower = ShortestPathFollower(sim, goal_radius=sd,
                                    return_one_hot=False, stop_on_error=True)

    chunk = []
    for _ in range(horizon):
        st = sim.get_agent_state()
        ok, dist, angle = _check_waypoint(st.position, st.rotation,
                                          target_info, is_final, sd, sh)
        if ok:
            chunk.append(ACTION_STOP)
            continue
        if not is_final and dist <= sd:
            signed_diff = _angle_diff_deg(st.rotation, target_rot_q)
            act = ACTION_RIGHT if signed_diff > 0 else ACTION_LEFT
            sim_act = (HabitatSimActions.turn_right if signed_diff > 0
                       else HabitatSimActions.turn_left)
            chunk.append(act)
            sim.step(sim_act)
            continue
        na = follower.get_next_action(np.array(target_info['position'], dtype=np.float32))
        if na is None:
            chunk.append(ACTION_STOP)
        else:
            act = HABITAT_TO_ID.get(na, ACTION_STOP)
            chunk.append(act)
            if act == ACTION_STOP:
                continue
            sim.step(na)

    while len(chunk) < horizon:
        chunk.append(ACTION_STOP)
    sim.set_agent_state(saved_pos, saved_rot)
    return chunk


# ──────────────────────────────────────────────────────────────────────
# Per-episode processing
# ──────────────────────────────────────────────────────────────────────

def process_episode(sim, ep_dir, seg_lookup, horizon, sd, sh):
    """Read rewrite_S.json, add expert_chunk, write dagger_meta.json."""
    rewrite_path = os.path.join(ep_dir, 'rgb', 'rewrite', 'rewrite_S.json')
    dagger_path = os.path.join(ep_dir, 'rgb', 'dagger_meta.json')
    if not os.path.exists(rewrite_path):
        return False
    if os.path.exists(dagger_path):
        return False  # already done

    with open(rewrite_path, 'r') as f:
        meta = json.load(f)

    eid = str(meta['episode_id'])
    seg_info = seg_lookup.get(eid)
    if not seg_info:
        return False
    waypoints = seg_info['waypoints']
    sub_insts = seg_info.get('sub_instructions', [])
    if len(waypoints) < 2:
        return False

    step_records = meta.get('step_records', [])
    if not step_records:
        return False

    old_segments = meta.get('segments', [])

    # ── Build new step_records with expert_chunk ──
    new_records = []
    for rec in step_records:
        seg_idx = rec.get('segment', 0)
        target_idx = min(seg_idx + 1, len(waypoints) - 1)
        target_info = waypoints[target_idx]
        is_final = (target_idx == len(waypoints) - 1)

        pos = rec['agent_position']
        rot = rec['agent_rotation']
        rot_list = [rot['x'], rot['y'], rot['z'], rot['w']]

        expert_chunk = query_expert_chunk(
            sim, pos, rot_list, target_info, horizon, is_final, sd, sh,
        )
        if expert_chunk is None:
            expert_chunk = []

        new_records.append({
            'segment': rec.get('segment', 0),
            'frame': rec.get('frame', 0),
            'executed_action': rec.get('action', 0),
            'executed_by': rec.get('controller', 'unknown'),
            'model_chunk': [],
            'expert_chunk': expert_chunk,
            'agent_position': rec.get('agent_position'),
            'agent_rotation': rec.get('agent_rotation'),
            'dist_to_target': rec.get('dist_to_target'),
            'heading_error_to_target': rec.get('heading_error_to_target'),
        })

    # ── Build new segments (dagger_meta format) ──
    new_segments = []
    for seg in old_segments:
        seg_idx = seg.get('segment', 0)
        new_segments.append({
            'segment': seg_idx,
            'segment_source': 'rewrite_expert_query',
            'status': seg.get('status', 'Success'),
            'dist': seg.get('dist'),
            'heading_error_to_target': seg.get('heading_error_to_target'),
            'cut_point_frame': seg.get('cut_point_frame', 0),
            'step_record_start_idx': seg.get('step_record_start_idx', 0),
            'step_record_end_idx': seg.get('step_record_end_idx', 0),
            'segment_idx': seg_idx,
            'sub_instruction': (sub_insts[seg_idx] if seg_idx < len(sub_insts) else ''),
            'global_instruction': meta.get('instruction', ''),
            'sub_instruction_source': 'seg_data',
            'waypoint_source': 'seg_data',
            'expert_target_source': 'segment_waypoint',
            'segment_completed_by': 'rewrite',
            'dist_to_waypoint_at_completion': seg.get('dist'),
            'waypoint_success_distance': sd,
            'waypoint_success_heading_deg': sh,
            'skipped_to_next_segment': False,
            'beta': 0.0,
            'action_horizon': horizon,
            'expert_steps_executed': 0,
            'model_steps_executed': seg.get('step_count', 0),
            'model_stop_override_count': 0,
            'oracle_stop_override_count': 0,
            'override_ratio': 0.0,
            'expert_query_restore_error': {'count': 0, 'max_position': 0.0,
                                           'max_rotation_deg': 0.0, 'max': 0.0},
            'previous_subtask_memory_visible': True,
            'step_count': seg.get('step_count', 0),
        })

    # ── Build actions array ──
    num_frames = meta.get('total_frames', len(new_records))
    actions = [-1] + [r['executed_action'] for r in new_records]
    # Pad to frame count
    while len(actions) < num_frames + 1:
        actions.append(0)

    dagger_meta = {
        'episode_id': eid,
        'scene_id': seg_info.get('scene_id', ''),
        'instruction': meta.get('instruction', ''),
        'segments': new_segments,
        'success': meta.get('success', False),
        'episode_success': meta.get('success', False),
        'partial': meta.get('partial', False),
        'failure_reason': meta.get('failure_reason'),
        'total_frames': num_frames,
        'actions': actions,
        'step_records': new_records,
        'collection_method': 'rewrite_expert_query',
        'beta': 0.0,
        'action_horizon': horizon,
    }

    with open(dagger_path, 'w') as f:
        json.dump(dagger_meta, f, indent=2)
    return True


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def _normalize_scene(scene_id):
    if not scene_id:
        return ''
    if not scene_id.startswith('mp3d/'):
        return os.path.join('data/scene_datasets/mp3d', scene_id, f'{scene_id}.glb')
    return os.path.join('data/scene_datasets', scene_id)


def _switch_scene(env, config, scene_id):
    full = _normalize_scene(scene_id)
    if not full or env.sim.config.sim_cfg.scene_id == full:
        return
    config.habitat.simulator.scene = full
    env.sim.reconfigure(config.habitat.simulator)


def _worker(worker_id, ep_items, seg_lookup, habitat_config_path,
            action_horizon, sd, sh, gpu_id):
    """Process a batch of episodes in one worker process."""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    config = get_config(habitat_config_path)
    config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    config.habitat.simulator.habitat_sim_v0.gpu_device_id = 0  # mapped to gpu_id via CUDA_VISIBLE_DEVICES
    env = habitat.Env(config=config)
    sim = env.sim

    done = skipped = 0
    current_scene = None
    for ep_dir, scene_id in ep_items:
        if scene_id != current_scene:
            _switch_scene(env, config, scene_id)
            current_scene = scene_id
        try:
            if process_episode(sim, ep_dir, seg_lookup, action_horizon, sd, sh):
                done += 1
            else:
                skipped += 1
        except Exception as e:
            print(f'[W{worker_id}] Error on {os.path.basename(ep_dir)}: {e}')
            skipped += 1

    env.close()
    return done, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_root', type=str, required=True)
    parser.add_argument('--seg_data_path', type=str, required=True)
    parser.add_argument('--habitat_config', type=str, default='configs/vln_r2r_dual.yaml')
    parser.add_argument('--action_horizon', type=int, default=3)
    parser.add_argument('--waypoint_success_distance', type=float, default=0.5)
    parser.add_argument('--waypoint_success_heading_deg', type=float, default=30.0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    seg_lookup = load_seg_lookup(args.seg_data_path)
    print(f'Loaded seg lookup: {len(seg_lookup)} episodes')

    ep_dirs = sorted(
        os.path.join(args.input_root, d) for d in os.listdir(args.input_root)
        if os.path.isdir(os.path.join(args.input_root, d))
    )

    # Group by scene
    scene_eps = {}
    for ep_dir in ep_dirs:
        eid = os.path.basename(ep_dir)
        sid = (seg_lookup.get(eid) or {}).get('scene_id', '')
        scene_eps.setdefault(sid, []).append(ep_dir)

    # Build per-scene item lists, distribute whole scenes to workers
    scene_items = []
    for sid, eps in sorted(scene_eps.items()):
        scene_items.append((sid, [(ep_dir, sid) for ep_dir in eps]))

    workers = min(args.num_workers, len(scene_items))
    batches = [[] for _ in range(workers)]
    for i, (sid, items) in enumerate(scene_items):
        batches[i % workers].extend(items)

    total_items = sum(len(b) for b in batches)
    print(f'Episodes: {total_items}, scenes: {len(scene_eps)}, workers: {workers}')
    sd, sh, hz = args.waypoint_success_distance, args.waypoint_success_heading_deg, args.action_horizon

    mp.set_start_method('spawn', force=True)
    with mp.Pool(workers) as pool:
        async_results = []
        for w_id, batch in enumerate(batches):
            if not batch: continue
            r = pool.apply_async(_worker, (w_id, batch, seg_lookup,
                                           args.habitat_config, hz, sd, sh, args.gpu))
            async_results.append(r)

        # Wait for all, showing periodic progress
        import time
        total_done = total_skip = 0
        while async_results:
            time.sleep(30)
            still_running = []
            for r in async_results:
                if r.ready():
                    d, s = r.get()
                    total_done += d
                    total_skip += s
                else:
                    still_running.append(r)
            async_results = still_running
            done_now = sum(1 for ep in os.listdir(args.input_root)
                          if os.path.exists(os.path.join(args.input_root, ep, 'rgb', 'dagger_meta.json')))
            print(f'Progress: {done_now}/{total_items} ({done_now/total_items*100:.1f}%)')

    print(f'Done: {total_done} updated, {total_skip} skipped')


if __name__ == '__main__':
    main()
