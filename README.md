# FastKV: KV Cache Compression for Fast Long-Context Processing with Token-Selective Propagation

<p align="middle">
    <a href="https://arxiv.org/abs/2502.01068"><img src="https://img.shields.io/badge/arXiv-2502.01068-b31b1b.svg" alt="arXiv"/></a>
</p>

<div align=center>
<img width=80% src="./images/overview.PNG"/>
</div>

This is the official repository of **"FastKV: KV Cache Compression for Fast Long-Context Processing with Token-Selective Propagation"**.

* FastKV introduces a novel Token-Selective Propagation (TSP) approach, selectively propagating only critical tokens to layer layers while retaining full-context information in early layers.
* This method significantly reduces KV cache size while maintaining accuracy, leading to improved latency and efficiency in long-context processing of LLMs.
* FastKV integrates GQA-aware KV cache compression, further optimizing memory and computation while leveraging grouped-query attention.
* Experimental results demonstrate that FastKV achieves up to 1.97× speedup in Time-To-First-Token (TTFT) and 5.07× higher throughput compared to full-context inference with 128k input tokens, all while preserving long-context accuracy.

## Usage
### 1. Installation

```
conda create -n fastkv python=3.9
conda activate fastkv
cd FastKV
pip install -r requirements.txt
pip install flash-attn==2.6.3

# For AdaKV and HeadKV
cd baseline/adakv
make i
```

### 2. Quick Start

```
# Run LongBench Evaluation
./scripts/run_longbench.sh

# Run Needle-in-a-Haystack Evaluation
./scripts/run_needle.sh

# Run TTFT Benchmark
./scripts/run_ttft.sh

# Run Throughput Benchmark
./scripts/run_throughput.sh
```

## Model Support

|         | FastKV | GemFilter | SnapKV | AdaKV | HeadKV |
|:-------:|:------:|:---------:|:------:|:-----:|:------:|
|  LLaMA  |    O   |     O     |    O   |   O   |    O   |
| Mistral |    O   |     O     |    O   |   O   |    O   |

## Citation
If you use the FastKV approach in your research,  please consider citing:

```
@article{fastkv,
  title={FastKV: KV Cache Compression for Fast Long-Context Processing with Token-Selective Propagation},
  author={Dongwon Jo, Jiwon Song, Yulhwa Kim, Jae-Joon Kim},
  journal={arXiv preprint arXiv:2502.01068},
  year={2024}
  }
```
