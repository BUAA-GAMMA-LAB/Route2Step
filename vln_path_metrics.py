import math
from typing import Iterable, List, Optional, Sequence

import numpy as np


def _to_position_array(position) -> np.ndarray:
    return np.asarray(position, dtype=np.float32)


def normalize_positions(positions: Optional[Iterable[Sequence[float]]]) -> List[List[float]]:
    normalized: List[List[float]] = []
    if not positions:
        return normalized
    for position in positions:
        array = _to_position_array(position)
        if array.shape[0] < 3:
            continue
        normalized.append([float(array[0]), float(array[1]), float(array[2])])
    return normalized


def compress_consecutive_positions(
    positions: Optional[Iterable[Sequence[float]]], atol: float = 1e-4
) -> List[List[float]]:
    compressed: List[List[float]] = []
    for position in normalize_positions(positions):
        if not compressed:
            compressed.append(position)
            continue
        if np.allclose(position, compressed[-1], atol=atol):
            continue
        compressed.append(position)
    return compressed


def euclidean_path_length(positions: Optional[Iterable[Sequence[float]]]) -> float:
    path = compress_consecutive_positions(positions)
    if len(path) < 2:
        return 0.0
    return float(
        sum(
            np.linalg.norm(_to_position_array(curr) - _to_position_array(prev))
            for prev, curr in zip(path, path[1:])
        )
    )


def resample_path_by_spacing(
    positions: Optional[Iterable[Sequence[float]]], spacing: float = 0.25
) -> List[List[float]]:
    path = compress_consecutive_positions(positions)
    if len(path) < 2 or spacing <= 0:
        return path

    segment_lengths = [
        float(np.linalg.norm(_to_position_array(curr) - _to_position_array(prev)))
        for prev, curr in zip(path, path[1:])
    ]
    total_length = float(sum(segment_lengths))
    if total_length <= 1e-8:
        return [path[0]]

    num_samples = max(len(path), int(math.ceil(total_length / spacing)) + 1)
    sample_distances = [
        total_length * idx / float(num_samples - 1) for idx in range(num_samples)
    ]

    resampled: List[List[float]] = []
    segment_idx = 0
    traversed = 0.0
    for target_distance in sample_distances:
        while (
            segment_idx < len(segment_lengths) - 1
            and traversed + segment_lengths[segment_idx] < target_distance
        ):
            traversed += segment_lengths[segment_idx]
            segment_idx += 1

        segment_length = segment_lengths[segment_idx]
        if segment_length <= 1e-8:
            resampled.append(path[segment_idx])
            continue

        ratio = (target_distance - traversed) / segment_length
        start = _to_position_array(path[segment_idx])
        end = _to_position_array(path[segment_idx + 1])
        point = start * (1.0 - ratio) + end * ratio
        resampled.append([float(point[0]), float(point[1]), float(point[2])])

    return resampled


def extract_result_trajectory_positions(result_payload) -> List[List[float]]:
    if not isinstance(result_payload, dict):
        return []

    trajectory_positions = result_payload.get('trajectory_positions', [])
    if trajectory_positions:
        return compress_consecutive_positions(trajectory_positions)

    steps = result_payload.get('steps', [])
    positions = []
    if isinstance(steps, list):
        for step in steps:
            if isinstance(step, dict) and 'position' in step:
                positions.append(step['position'])
    return compress_consecutive_positions(positions)


def geodesic_distance(sim, position, target_positions, episode=None) -> float:
    targets = [_to_position_array(target) for target in target_positions or []]
    if not targets:
        return float('inf')

    source = _to_position_array(position)
    if sim is not None:
        try:
            if episode is None:
                distance = sim.geodesic_distance(source, targets)
            else:
                distance = sim.geodesic_distance(source, targets, episode)
            if np.isfinite(distance):
                return float(distance)
        except Exception:
            pass

    return float(min(np.linalg.norm(source - target) for target in targets))


def compute_ndtw(
    sim,
    predicted_positions: Optional[Iterable[Sequence[float]]],
    reference_path: Optional[Iterable[Sequence[float]]],
    success_distance: float = 3.0,
    episode=None,
    reference_spacing: float = 0.25,
    use_geodesic: bool = False,
) -> float:
    predicted = compress_consecutive_positions(predicted_positions)
    reference = resample_path_by_spacing(reference_path, spacing=reference_spacing)

    if not predicted or not reference or success_distance <= 0:
        return 0.0

    n_pred = len(predicted)
    n_ref = len(reference)
    dtw = np.full((n_pred + 1, n_ref + 1), np.inf, dtype=np.float64)
    dtw[0, 0] = 0.0

    for i in range(1, n_pred + 1):
        predicted_position = predicted[i - 1]
        for j in range(1, n_ref + 1):
            if use_geodesic:
                local_distance = geodesic_distance(
                    sim,
                    predicted_position,
                    [reference[j - 1]],
                    episode=episode,
                )
            else:
                local_distance = float(
                    np.linalg.norm(
                        _to_position_array(predicted_position)
                        - _to_position_array(reference[j - 1])
                    )
                )
            dtw[i, j] = local_distance + min(
                dtw[i - 1, j],
                dtw[i, j - 1],
                dtw[i - 1, j - 1],
            )

    normalization = success_distance * float(n_ref)
    if normalization <= 0:
        return 0.0

    return float(math.exp(-float(dtw[n_pred, n_ref]) / normalization))


def compute_sdtw(success: float, ndtw: float) -> float:
    return float(ndtw if float(success) > 0.0 else 0.0)
