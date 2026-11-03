# Flexformer causal Triton implementation

The causal path follows the same recurrent algorithm as the Linear update in
`deltanet-linear-triton-update`: one Triton program per batch/head, a forward
time loop updating S/z, a reverse time loop for gradients, and state
recomputation during backward. It is not a chunk-parallel attention algorithm.

The Flexformer feature map, frequency parameters, key norm factor, and
temperature remain in PyTorch. Autograd propagates gradients returned by the
Triton scan through those operations. The necessary dimensional adaptation is
that the feature dimension M = 2*num_freq can differ from value dimension Dv:

```text
Q/K features: [B, H, T, M]
V/output:    [B, H, T, Dv]
S state:     [B, H, M, Dv]
z state:     [B, H, M]

S[t] = S[t-1] + phi(K[t]) outer V[t]
z[t] = z[t-1] + phi(K[t])
O[t] = (phi(Q[t]) @ S[t]) / (phi(Q[t]) @ z[t] + 1e-6)
```

Padding keys do not update either state. The scan includes the current input
token; shift targets by one token for next-token prediction.

## Files

- `models/flexformer.py`: causal flags, decoder model, and incremental cache.
- `models/causal_linear_attention.py`: validation, dispatch, and torch reference.
- `models/linear_triton.py`: fused forward/backward kernels with rectangular states.
- `tests/test_causal_attention.py`: optional output/gradient and causality checks.

The existing `FlexformerEncoder` defaults to bidirectional attention. For causal
hidden states, set `causal=True, classifier=False`. A prepended CLS classifier
is rejected in causal mode because it cannot read subsequent tokens.

## Dependencies and backend selection

Use a PyTorch installation appropriate for your CUDA environment, and install
NumPy. Triton is optional; install it on a supported Linux/CUDA environment:

```bash
pip install numpy
pip install triton
```

`attention_backend` accepts:

- `"auto"`: use Triton on supported CUDA tensors, otherwise use torch.
- `"triton"`: require the Triton path and fail explicitly if unsupported.
- `"torch"`: use the differentiable recurrent reference.

The Triton path supports float32, float16, and bfloat16, with FP32 state
accumulation. Current limits: M <= 256, Dv <= 128, and the product of their
power-of-two padded dimensions <= 32768. The dimensions need not be equal or
powers of two. Initial/final state gradients are supported as well.

## Language-model training

Run from the code directory. The decoder returns unnormalized [B,T,vocab]
logits; it does not apply log-softmax.

```python
import torch
import torch.nn.functional as F
from models.flexformer import FlexformerDecoder

model = FlexformerDecoder(
    vocab_size=32000,
    emb_dim=512,
    qkv_dim=512,
    num_heads=8,
    num_layers=6,
    mlp_dim=2048,
    num_freq=32,
    max_len=512,
    attention_backend="triton",
).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

# tokens: CUDA LongTensor [B,T], with padding token ID 0.
logits = model(tokens[:, :-1])
loss = F.cross_entropy(
    logits.reshape(-1, logits.size(-1)),
    tokens[:, 1:].reshape(-1),
    ignore_index=0,
)
loss.backward()
optimizer.step()
```

MLP and position parameters are created before the first forward, so they are
included when constructing the optimizer. The duplicated attention helper's
method binding and padding after the nonlinear feature map are also corrected.
Flexformer MLP dropout now respects `model.eval()`.

## Prefill and incremental decoding

Pass only new tokens on each call. One cache belongs to one batch of sequences;
create a fresh dictionary when beginning a different batch. Beam reordering and
packed-sequence segment resets are not implemented. Total positions, including
cached tokens, must fit max_len.

```python
model.eval()
cache = {}
with torch.no_grad():
    prompt_logits = model(prompt_tokens, incremental_state=cache)
    next_token = prompt_logits[:, -1].argmax(-1, keepdim=True)
    next_logits = model(next_token, incremental_state=cache)
```

The cache contains a position offset and one (S,z) pair per layer. Padding does
not update S/z, while positional offsets count supplied tokens, including pads.

## Memory and verification scope

The recurrence's active state and decoding cache have size O(B*H*M*Dv).
Forward retains Q/K/V and initial states for autograd, but does not retain full
state histories across model layers. Backward rebuilds the current layer's
histories with O(B*H*T*M*Dv) temporary storage. This is recomputation, not a
constant-memory training claim.

The implementation was statically reviewed and Python syntax was checked.
Runtime/GPU verification has not yet been performed.
The optional tests can be run later with PyTorch/pytest installed:

```bash
python -m pytest -q tests/test_causal_attention.py
```

CUDA tests explicitly skip when CUDA/Triton is unavailable. No GPU speedup or
numerical-test result is claimed for this delivery.

The kernel-fusion approach is inspired by
[Flash Linear Attention (FLA)](https://github.com/fla-org/flash-linear-attention).
This code provides its own recurrent kernels and does not require the FLA package.
