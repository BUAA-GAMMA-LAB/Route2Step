<br>
<p align="center">
<h1 align="center"><strong>From Routes to Steps: Separating Semantic Progress from Local Execution in Vision-and-Language Navigation</strong></h1>
  <p align="center">
    <strong>
    Xiangyun Huang<sup>1</sup>&emsp;
    Xiangchen Wang<sup>2</sup>&emsp;
    Runfeng Lin<sup>1,3</sup>&emsp;
    Yihao Xu<sup>1</sup>
    <br>
    Kangyu Huang<sup>4</sup>&emsp;
    Jiang Hengchen<sup>1,5</sup>&emsp;
    Xiwang Dong<sup>1</sup>&emsp;
    Lin Jiarong<sup>1,*</sup>
    </strong>
    <br>
    <sup>1</sup>Beihang University&emsp;
    <sup>2</sup>Southern University of Science and Technology&emsp;
    <sup>3</sup>Central South University&emsp;
    <br>
    <sup>4</sup>Harbin Institute of Technology, Shenzhen&emsp;
    <sup>5</sup>Dalian University of Technology
    <br>
    <code>23231088@buaa.edu.cn, zivlin@buaa.edu.cn</code>
  </p>
</p>

<!-- TODO: add the arXiv and Bilibili links when available. -->
<p id="top" align="center">
  <img src="https://img.shields.io/badge/arXiv-coming_soon-red?logo=arxiv" alt="arXiv">
  <a href="https://sisyphus-hxy.github.io/Route2Step/"><img src="https://img.shields.io/badge/Project_Page-0065D3?logo=rocket&amp;logoColor=white" alt="Project Page"></a>
  <a href="https://huggingface.co/XiangyunHuang/Route2Step"><img src="https://img.shields.io/badge/Hugging_Face-FF9D00?logo=huggingface&amp;logoColor=white" alt="Hugging Face"></a>
  <a href="https://www.youtube.com/watch?v=vBUAny2WqM0"><img src="https://img.shields.io/badge/YouTube-D33846?logo=youtube&amp;logoColor=white" alt="YouTube"></a>
  <img src="https://img.shields.io/badge/Bilibili-coming_soon-00A1D6?logo=bilibili&amp;logoColor=white" alt="Bilibili">
</p>

## 🏠 About

This repository contains the implementation of **Route2Step**, a
vision-and-language navigation framework that separates route-level progress
tracking from local action execution.

## Environment

The released evaluation code targets Python 3.10. `requirements_eval.txt`
installs the regular Python dependencies, but it intentionally does not
install Habitat-Sim or Habitat-Lab. Both Habitat packages must use v0.2.4 and
are installed separately from their original source repositories.

### 1. Base environment

Create the environment and install the evaluation packages:

```bash
conda create -n route2step_py310 python=3.10 -y
conda activate route2step_py310
pip install -r requirements_eval.txt
```

PyTorch should match the CUDA version and GPU driver of the target machine.
The evaluation scripts call an external OpenAI-compatible vLLM server;
`vllm` itself is not installed by this file.

### 2. Build Habitat-Sim v0.2.4

The Habitat source repositories can be stored anywhere. The following sibling
layout is recommended for convenience:

```text
workspace/
├── Route2Step/
├── habitat-sim/
└── habitat-lab/
```

From `workspace/`, clone Habitat-Sim with all submodules and compile the
headless CUDA build:

```bash
git clone --branch v0.2.4 --recursive https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim
python setup.py install --headless --with-cuda --bullet
cd ..
```

### 3. Install Habitat-Lab v0.2.4

Install the original Habitat-Lab v0.2.4 source without modifying it:

```bash
git clone --branch v0.2.4 https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
pip install -e habitat-lab
cd ../Route2Step
```

`habitat-baselines` is not required by the released evaluation entry points.
Because Habitat-Lab is installed in editable mode, do not move or delete its
source directory afterward. If the source repositories use a different
layout, Route2Step does not need to be changed because it does not hard-code
either Habitat source directory. R2R/RxR schema compatibility is handled by
the dataset loader included in this repository.

Verify that both packages come from the active environment and that
Habitat-Sim was compiled with CUDA:

```bash
python -c "import habitat, habitat_sim; from habitat_sim.bindings import cuda_enabled; print(habitat.__file__); print(habitat_sim.__file__); print('cuda_enabled:', cuda_enabled)"
```

### 4. Runtime data paths

Run evaluation commands from the Route2Step repository root. Unlike the
Habitat source location, dataset paths are relative to this working directory
in the released YAML files:

```text
Route2Step/
└── data/
    ├── scene_datasets/
    │   └── mp3d/
    │       └── <scan_id>/
    │           └── <scan_id>.glb
    └── datasets/
        ├── R2R_VLNCE_v1-3/
        │   └── val_unseen/
        │       └── val_unseen.json.gz
        └── rxr/
            └── val_unseen/
                └── val_unseen_guide.json.gz
```

You may store the datasets elsewhere, but then update `scenes_dir` and
`data_path` in `configs/vln_r2r_dual.yaml` and
`configs/vln_rxr_dual.yaml`. Trajectory-image roots are configured separately
near the top of the evaluation shell scripts.

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
