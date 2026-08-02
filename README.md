# Route2Step

This repository contains the implementation of **Route2Step**, a
vision-and-language navigation framework that separates route-level progress
tracking from local action execution.

**Project page:** <https://sisyphus-hxy.github.io/Route2Step/>

## Environment

The released evaluation code targets Python 3.10. Create a new environment or
reuse the same environment for future training dependencies:

```bash
conda create -n route2step_py310 python=3.10 -y
conda activate route2step_py310
pip install -r requirements_eval.txt
```

`requirements_eval.txt` contains the minimum packages for static M1 scoring
and Habitat navigation evaluation. PyTorch should match the CUDA version and
GPU driver of the target machine. The evaluation scripts call an external
OpenAI-compatible vLLM server; `vllm` itself is not installed by this file.

## External data and models

Datasets, Matterport3D scene assets, trajectory images, model checkpoints, and
the local sentence-embedding checkpoint are not bundled with the code. Put
them under the paths expected by the configuration files, or edit the paths in
the evaluation scripts and YAML configs before running them.

The Route2Step model weights are available in the single Hugging Face
repository [`XiangyunHuang/Route2Step`](https://huggingface.co/XiangyunHuang/Route2Step),
under the `MIA/` and `MAG/` subdirectories. Download the repository and pass
the corresponding subdirectory as the M1 or M2 model path.

## Evaluation

Run commands from this directory.

### Static M1 evaluation

Start a vLLM server at `http://127.0.0.1:8086`, then run:

```bash
TASK_TYPE=single_m1 \
DATASET=rxr_deviation \
RESULT_DIR=eval_results/m1/rxr_deviation \
bash scripts/eval_m1_static_qa.sh
```

Supported `TASK_TYPE` values are `single_m1` and `m1_subinstruction`. Supported
`DATASET` values are `r2r`, `r2r_deviation`, `rxr`, and `rxr_deviation`.

### R2R navigation evaluation

Start the M1 and M2 vLLM servers at ports `8081` and `8080`, respectively.
Set the model, data, and result paths in
`scripts/eval_qwen2_5_dual_lm.sh`, then run:

```bash
bash scripts/eval_qwen2_5_dual_lm.sh
```

### RxR navigation evaluation

Configure the RxR data paths and model servers in
`scripts/eval_qwen2_5_dual_rxr.sh`, then run:

```bash
bash scripts/eval_qwen2_5_dual_rxr.sh
```

The alignment and data-construction utilities are under `seg/`, `DAgger/`,
and `scripts/`. Use `python <script> --help` to inspect their input paths and
options. Training-only dependencies can be added to this same Conda
environment later.
