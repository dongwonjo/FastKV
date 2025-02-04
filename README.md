# Mixture of Scales: Memory-Efficient Token-Adaptive Binarization for Large Language Models

<p align="middle">
    <a href="https://arxiv.org/abs/2502.01068"><img src="https://img.shields.io/badge/arXiv-2502.01068-b31b1b.svg" alt="arXiv"/></a>
</p>

<div align=center>
<img width=100% src="./imgs/FastKV.PNG"/>
</div>


This is the official repository of **"FastKV: KV Cache Compression for Fast Long-Context Processing with Token-Selective Propagation"**

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

## Support Models

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
