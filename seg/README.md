# Segmentation core code

## R2R-train E-SPA

The main entry point is:

```text
r2r_train/seg_r2r_train.py
```

Supporting files provide the action-intent mapping, stair filtering and
alignment, dataset normalization, and height-span utilities:

```text
r2r_train/gen_sub_instruction_actions_r2r_train.py
r2r_train/stair_align_api.py
r2r_train/stair_alignment_utils.py
r2r_train/stair_dataset_utils.py
```

The implementation uses CLIP semantic cost, 3-D action/motion cost, duration
regularization, monotonic dynamic programming, and an optional stair-aware
alignment path. R2R annotations, trajectory coordinates, images, checkpoints,
and API credentials are external inputs.

## Discrete-to-continuous utilities

`discrete_to_continuous/` contains the shared projection and turn-boundary
utilities used by the released preprocessing path. These scripts are
auxiliary to the core R2R E-SPA entry point and require the corresponding
external dataset files.
