# Flexformer

Flexible Linear Transformer with Learnable Kernel.

This repository includes the Flexformer and Transformer models for ListOps,
as well as a causal Flexformer language-model decoder with fused Triton
forward/backward kernels and incremental state caching.

See [README_CAUSAL.md](README_CAUSAL.md) for dependencies, model usage,
backend selection, tensor layouts, and implementation details.

The Triton implementation uses a recurrent scan over time with rectangular
feature/value states and state recomputation during backward. GPU numerical
validation and performance benchmarks have not yet been run.

## Layout

- `models/`: models, causal attention dispatch, and Triton kernels.
- `listops/`, `listops_train.py`: ListOps data and training code.
- `scripts/listops/`: training script examples.
- `tests/`: optional correctness tests for causal attention and gradients.

## Citation

If you find this repo helpful, please cite our paper.

```bibtex
@inproceedings{
zhang2026flexformer,
title={Flexformer: Flexible Linear Transformer with Learnable Attention Kernel},
author={Zhang, Haoran and Dong, Zisu and Zhou, Feng},
booktitle={The Fortieth Annual Conference on Neural Information Processing Systems},
year={2026},
url={https://openreview.net/forum?id=D8frDvzgAf}
}
```
