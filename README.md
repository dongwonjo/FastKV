### Installation

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

### Quick Start

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

### Support Models

|         | FastKV | GemFilter | SnapKV | AdaKV | HeadKV |
|:-------:|:------:|:---------:|:------:|:-----:|:------:|
|  LLaMA  |    O   |     O     |    O   |   O   |    O   |
| Mistral |    O   |     O     |    O   |   O   |    O   |