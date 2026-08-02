# Route2Step supplementary core code

This package releases the core implementation used in the paper. It is
intended to make the main method and its preprocessing logic inspectable.
Running the preprocessing and evaluation pipelines still requires datasets,
Matterport3D images, trajectory files, model checkpoints, and API credentials
downloaded or supplied separately by the user.

## Core components

| Paper component | Main entry points |
|---|---|
| R2R-train E-SPA alignment | `seg/r2r_train/seg_r2r_train.py` |
| Action-intent labels | `seg/r2r_train/gen_sub_instruction_actions_r2r_train.py` |
| Stair filtering/alignment | `seg/r2r_train/stair_align_api.py`, `stair_alignment_utils.py` |
| Corrective DAgger rollout | `DAgger/route_rewrite.py`, `agent_dual_qwen2_5_lm.py` |
| Deviation trajectory generation | `scripts/build_r2r_deviation_trajectories.py`, `scripts/build_rxr_deviation_trajectories.py` |
| M1 static evaluation | `eval_m1_static_qa.py`, `scripts/eval_m1_static_qa.sh` |
| Navigation evaluation | `habitat_vln/`, `scripts/eval_qwen2_5_dual_lm.sh` |

The R2R E-SPA implementation contains semantic CLIP cost, action-intent
motion cost, duration regularization, monotonic dynamic programming, and the
stair-aware alignment path. The DAgger entry point uses the released dual LM
agent implementation.

## External inputs

The following are intentionally not duplicated in the code package:

- R2R/RxR annotations and continuous trajectory files;
- extracted trajectory images and Matterport3D simulator assets;
- CLIP/Qwen checkpoints;
- API keys and API-generated caches.

The complete segmentation annotations, continuous trajectories,
deviation-evaluation JSON, trajectory images, and simulator assets remain
external inputs. Compact or partial JSON artifacts used for schema and
pipeline inspection are included under `data/` in this same supplementary
package.

The code package is not self-contained. In particular, before running the R2R
E-SPA entry point, download the R2R/VLN-CE train annotations, continuous
trajectory coordinates, RGB trajectory frames, and the required CLIP
checkpoints. The included JSON artifacts do not replace these source inputs.

## Included data artifacts

The `data/` directory is included alongside the code in this supplementary
package. It contains schema-valid compact or partial JSON artifacts for
inspecting the segmentation pipeline; it is not a replacement for the
complete public datasets or the full experiment data.

The included R2R segmentation artifact
`data/seg/r2r_train/seg_r2r_train_with_coords.json`
contains 10,818 episodes and 48,447 sub-instruction segments. The other files
under `data/seg/` are auxiliary conversion artifacts and are
not required for the core R2R-train E-SPA reproduction. The included data are
partial inspection artifacts rather than complete training or evaluation
datasets.

## Basic commands

Run from this directory after installing the required external environment and
downloading the external R2R/VLN-CE train inputs and continuous coordinates:

```bash
python seg/r2r_train/seg_r2r_train.py \
  --input_file <R2R_SEGMENTATION_INPUT> \
  --annotation_file <R2R_ANNOTATIONS> \
  --coord_file <R2R_COORDINATES> \
  --images_base_dir <R2R_IMAGE_ROOT> \
  --action_map_file <ACTION_MAP> \
  --output_file <OUTPUT_JSON>
```

For the final stair-aware E-SPA result, first generate the external stair
alignment cache, then pass that cache to the E-SPA entry point. The cache is
not generated implicitly by `seg_r2r_train.py`.

```bash
python seg/r2r_train/stair_align_api.py \
  --dataset r2r \
  --input_file <R2R_SEGMENTATION_INPUT> \
  --coord_file <R2R_COORDINATES> \
  --stair_filter_cache <STAIR_FILTER_CACHE> \
  --output_file <STAIR_ALIGNMENT_CACHE> \
  --api_keys_file <API_KEYS_FILE>

python seg/r2r_train/seg_r2r_train.py \
  --input_file <R2R_SEGMENTATION_INPUT> \
  --annotation_file <R2R_ANNOTATIONS> \
  --coord_file <R2R_COORDINATES> \
  --images_base_dir <R2R_IMAGE_ROOT> \
  --action_map_file <ACTION_MAP> \
  --stair_alignment_file <STAIR_ALIGNMENT_CACHE> \
  --output_file <OUTPUT_JSON>
```

The stair filter and stair alignment API use the endpoint, model, and key file
supplied by command-line arguments. If no valid alignment entry is available
for an episode, the E-SPA entry point uses the ordinary monotonic DP path.

The discrete-to-continuous converter enables turn-boundary refinement by
default. Use `--no-turn-boundary-api` only for a projection-only run without
external API requests.

For corrective rollout, use `DAgger/route_rewrite.py` with the local Habitat
configuration and LM checkpoint/server supplied by the user.

## Environment

`environment_route2step_py310.yml` is a direct export of the tested
`route2step_py310` environment. It is an environment snapshot rather than a
minimal lockfile: it includes the CUDA, Habitat, training, and inference
dependencies present on the reference machine. GPU/CPU hardware, simulator
assets, checkpoints, and service credentials are machine-specific and are not
bundled here.

The release intentionally targets the checklist's core-code scope. It does
not claim to include every training-data builder, every historical DAgger
branch, or all per-run experiment metadata.
