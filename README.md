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

<!-- TODO: add the Bilibili link when available. -->
<p id="top" align="center">
  <a href="https://arxiv.org/abs/2608.03143"><img src="https://img.shields.io/badge/arXiv-2608.03143-red?logo=arxiv" alt="arXiv"></a>
  <a href="https://sisyphus-hxy.github.io/Route2Step/"><img src="https://img.shields.io/badge/Project_Page-0065D3?logo=rocket&amp;logoColor=white" alt="Project Page"></a>
  <a href="https://huggingface.co/XiangyunHuang/Route2Step"><img src="https://img.shields.io/badge/Hugging_Face-FF9D00?logo=huggingface&amp;logoColor=white" alt="Hugging Face"></a>
  <a href="https://www.youtube.com/watch?v=vBUAny2WqM0"><img src="https://img.shields.io/badge/YouTube-D33846?logo=youtube&amp;logoColor=white" alt="YouTube"></a>
  <img src="https://img.shields.io/badge/Bilibili-coming_soon-00A1D6?logo=bilibili&amp;logoColor=white" alt="Bilibili">
</p>

## 🏠 About

This repository contains the implementation of **Route2Step**, a
vision-and-language navigation framework that separates route-level progress
tracking from local action execution.

## 🛠 Getting Started

We test under Python 3.10 with Habitat-Sim and Habitat-Lab v0.2.4.

1. **Create the environment**

```bash
conda create -n route2step_py310 python=3.10 -y
conda activate route2step_py310
pip install -r requirements_eval.txt
```

2. **Install Habitat v0.2.4**

Run from the Route2Step repository root. Both repositories will be placed
under `Route2Step/`.

```bash
git clone --branch v0.2.4 --recursive https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim
python setup.py install --headless --with-cuda --bullet
cd ..
git clone --branch v0.2.4 https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
pip install -e habitat-lab
cd ..
```

## External data and models

Datasets and Matterport3D assets are not bundled. Place MP3D scenes under
`data/scene_datasets/mp3d/` and R2R/RxR episodes under `data/datasets/`, or
update the paths in `configs/`.

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

## Citation

```bibtex
@misc{huang2026routesstepsseparatingsemantic,
      title={From Routes to Steps: Separating Semantic Progress from Local Execution in Vision-and-Language Navigation},
      author={Xiangyun Huang and Xiangchen Wang and Runfeng Lin and Yihao Xu and Kangyu Huang and Jiang Hengchen and Xiwang Dong and Lin Jiarong},
      year={2026},
      eprint={2608.03143},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2608.03143},
}
```
