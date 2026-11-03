"""Compare recurrence with dense causal attention, including learned features.

CUDA tests exercise the actual Triton forward/backward; they are explicitly
skipped when CUDA/Triton is unavailable, not reported as GPU validation.
"""

import copy

import pytest
import torch

from models.causal_linear_attention import causal_linear_attention
from models.flexformer import FlexformerDecoder, FlexformerEncoder, lnk_kernel, lnk_linear_attention
from models.linear_triton import triton_supported


def dense_reference(q, k, v, mask=None, state=None, eps=1e-6):
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    q, k, v = (x.to(dtype) for x in (q, k, v))
    if mask is not None:
        k = k.masked_fill(~mask[:, None, :, None], 0)
        v = v.masked_fill(~mask[:, None, :, None], 0)
    scores = q @ k.transpose(-1, -2)
    scores = scores.masked_fill(
        ~torch.ones(q.shape[2], k.shape[2], device=q.device, dtype=torch.bool).tril(), 0
    )
    numerator = scores @ v
    denominator = scores.sum(dim=-1, keepdim=True)
    final_s = k.transpose(-1, -2) @ v
    final_z = k.sum(dim=2)
    if state is not None:
        s0, z0 = state
        numerator = numerator + q @ s0
        denominator = denominator + (q * z0.unsqueeze(2)).sum(dim=-1, keepdim=True)
        final_s = final_s + s0
        final_z = final_z + z0
    return numerator / (denominator + eps), (final_s, final_z)


@pytest.mark.parametrize('with_mask', [False, True])
@pytest.mark.parametrize('with_state', [False, True])
def test_rectangular_outputs_and_gradients(with_mask, with_state):
    torch.manual_seed(31)
    q = (torch.rand(2, 2, 7, 10, dtype=torch.double) + 0.1).requires_grad_()
    k = (torch.rand_like(q) + 0.1).requires_grad_()
    v = torch.randn(2, 2, 7, 5, dtype=torch.double, requires_grad=True)
    mask = torch.tensor([[0, 1, 1, 0, 1, 1, 1], [1, 1, 0, 1, 1, 0, 0]], dtype=torch.bool) if with_mask else None
    state = None
    parameters = [q, k, v]
    if with_state:
        state = (
            torch.randn(2, 2, 10, 5, dtype=torch.double, requires_grad=True),
            torch.ones(2, 2, 10, dtype=torch.double, requires_grad=True),
        )
        parameters.extend(state)
    out, final = causal_linear_attention(q, k, v, mask, state, True, backend='torch')
    expected, expected_final = dense_reference(q, k, v, mask, state)
    torch.testing.assert_close(out, expected)
    for actual, wanted in zip(final, expected_final):
        torch.testing.assert_close(actual, wanted)
    weight = torch.randn_like(out)
    loss = (out * weight).sum() + final[0].square().mean() + final[1].square().mean()
    expected_loss = (expected * weight).sum() + expected_final[0].square().mean() + expected_final[1].square().mean()
    actual_grads = torch.autograd.grad(loss, parameters, retain_graph=True)
    expected_grads = torch.autograd.grad(expected_loss, parameters)
    for actual, wanted in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual, wanted, atol=1e-9, rtol=1e-8)


def test_learned_frequencies_and_temperature_gradients():
    torch.manual_seed(8)
    kernel = lnk_kernel(4, 5, 2).double()
    with torch.no_grad():
        kernel.omega2_bias.fill_(-0.8)
    q, k = (torch.randn(2, 2, 6, 4, dtype=torch.double) * 0.3 for _ in range(2))
    v = torch.randn(2, 2, 6, 3, dtype=torch.double)
    out = lnk_linear_attention(q, k, v, kernel, causal=True, backend='torch')
    tau = kernel.log_tau.exp()[None, :, None, None]
    phi_q = kernel(q)
    phi_k = kernel(k) * (k.norm(dim=-1, keepdim=True) / tau).exp()
    expected, _ = dense_reference(phi_q, phi_k, v)
    torch.testing.assert_close(out, expected)
    weight = torch.randn_like(out)
    grads = torch.autograd.grad((out * weight).sum(), tuple(kernel.parameters()), retain_graph=True)
    reference = torch.autograd.grad((expected * weight).sum(), tuple(kernel.parameters()))
    for actual, wanted in zip(grads, reference):
        torch.testing.assert_close(actual, wanted, atol=1e-9, rtol=1e-8)
        assert torch.isfinite(actual).all()
        assert actual.abs().sum() > 0


def small_decoder(**kwargs):
    return FlexformerDecoder(
        vocab_size=29, emb_dim=16, qkv_dim=16, num_heads=2, num_layers=2,
        mlp_dim=24, num_freq=5, max_len=20, dropout_rate=0.1,
        attention_dropout_rate=0.1, **kwargs,
    ).eval()


def test_decoder_causality_and_cached_prefill():
    torch.manual_seed(4)
    model = small_decoder(attention_backend='torch')
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6], [7, 8, 0, 9, 10, 0]])
    with torch.no_grad():
        full = model(tokens)
        changed = tokens.clone()
        changed[:, 4:] = 17
        torch.testing.assert_close(full[:, :4], model(changed)[:, :4])
        cache = {}
        pieces = [model(tokens[:, :2], incremental_state=cache)]
        pieces.append(model(tokens[:, 2:5], incremental_state=cache))
        pieces.append(model(tokens[:, 5:], incremental_state=cache))
        torch.testing.assert_close(torch.cat(pieces, dim=1), full, atol=2e-6, rtol=2e-5)
    assert cache['position'] == tokens.shape[1]
    assert len(cache['layers']) == 2
    assert cache['layers'][0][0].shape == (2, 2, 10, 8)


def test_parameters_are_registered_before_first_forward_and_trainable():
    torch.manual_seed(1)
    model = small_decoder(attention_backend='torch', learn_pos_emb=True, tied_weights=True)
    assert model.output_proj.weight is model.embedding.embedding
    before = {name for name, _ in model.named_parameters()}
    assert 'layers.0.mlp.dense1.weight' in before
    assert 'pos_emb.pos_embedding' in before
    model.train()
    tokens = torch.tensor([[1, 2, 3, 4, 5]])
    logits = model(tokens[:, :-1])
    torch.nn.functional.cross_entropy(logits.reshape(-1, 29), tokens[:, 1:].reshape(-1)).backward()
    assert before == {name for name, _ in model.named_parameters()}
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_bidirectional_default_and_post_feature_padding_mask():
    torch.manual_seed(12)
    model = FlexformerEncoder(
        vocab_size=30, emb_dim=16, qkv_dim=16, num_heads=2, num_layers=1,
        mlp_dim=24, num_freq=5, max_len=10,
    ).eval()
    tokens = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        full = model(tokens)
        padded = model(torch.cat([tokens, torch.zeros(1, 3, dtype=torch.long)], dim=1))
    torch.testing.assert_close(full, padded[:, :4], atol=2e-6, rtol=2e-5)
    assert not model.causal


def test_all_padding_zero_and_backend_validation():
    q, k = (torch.ones(2, 1, 3, 6) for _ in range(2))
    v = torch.randn(2, 1, 3, 4)
    out = causal_linear_attention(q, k, v, torch.zeros(2, 3, dtype=torch.bool))
    torch.testing.assert_close(out, torch.zeros_like(v))
    with pytest.raises(ValueError, match='backend'):
        causal_linear_attention(q, k, v, backend='unknown')
    with pytest.raises(RuntimeError, match='Triton requires'):
        causal_linear_attention(q, k, v, backend='triton')
    with pytest.raises(ValueError, match='valid_mask'):
        causal_linear_attention(q, k, v, torch.ones(2, 3))
    with pytest.raises(ValueError, match='CLS'):
        FlexformerEncoder(vocab_size=30, causal=True, classifier=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required for actual Triton execution')
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('length,m,dv', [(1, 6, 3), (17, 14, 5), (65, 32, 16)])
def test_triton_forward_backward_and_states(dtype, length, m, dv):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip('Device does not support bfloat16')
    torch.manual_seed(10)
    q = (torch.rand(2, 2, length, m, device='cuda', dtype=dtype) + 0.2).requires_grad_()
    k = (torch.rand_like(q) + 0.2).requires_grad_()
    v = torch.randn(2, 2, length, dv, device='cuda', dtype=dtype, requires_grad=True)
    if not triton_supported(q, v):
        pytest.skip('Triton is not installed or this configuration is unsupported')
    s0 = torch.randn(2, 2, m, dv, device='cuda', requires_grad=True)
    z0 = torch.ones(2, 2, m, device='cuda', requires_grad=True)
    mask = torch.rand(2, length, device='cuda') > 0.3
    expected, expected_state = causal_linear_attention(
        q, k, v, mask, (s0, z0), True, backend='torch'
    )
    out, state = causal_linear_attention(q, k, v, mask, (s0, z0), True, backend='triton')
    atol = 3e-2 if dtype == torch.bfloat16 else (3e-3 if dtype == torch.float16 else 2e-5)
    torch.testing.assert_close(out, expected, atol=atol, rtol=atol)
    for actual, reference in zip(state, expected_state):
        torch.testing.assert_close(actual, reference, atol=2e-5, rtol=2e-5)
    weight = torch.randn_like(out)
    loss = (out * weight).sum() + state[0].square().mean() + state[1].square().mean()
    expected_loss = (expected * weight).sum() + expected_state[0].square().mean() + expected_state[1].square().mean()
    actual_grads = torch.autograd.grad(loss, (q, k, v, s0, z0), retain_graph=True)
    expected_grads = torch.autograd.grad(expected_loss, (q, k, v, s0, z0))
    for actual, reference in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual, reference, atol=atol, rtol=atol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required for actual Triton execution')
def test_triton_model_gradients_and_incremental_outputs():
    torch.manual_seed(20)
    actual = small_decoder(attention_backend='triton').cuda()
    if not triton_supported(torch.empty(1, 2, 1, 10, device='cuda'), torch.empty(1, 2, 1, 8, device='cuda')):
        pytest.skip('Triton is not available')
    reference = copy.deepcopy(actual)
    for layer in reference.layers:
        layer.attn.attention_backend = 'torch'
    tokens = torch.tensor([[1, 2, 0, 3, 4], [6, 7, 8, 9, 0]], device='cuda')
    expected, out = reference(tokens), actual(tokens)
    torch.testing.assert_close(out, expected, atol=2e-5, rtol=2e-4)
    weight = torch.randn_like(out)
    (out * weight).sum().backward()
    (expected * weight).sum().backward()
    for (name, p), (_, r) in zip(actual.named_parameters(), reference.named_parameters()):
        torch.testing.assert_close(p.grad, r.grad, atol=3e-4, rtol=3e-3, msg=name)
    with torch.no_grad():
        cache = {}
        prefix = actual(tokens[:, :3], incremental_state=cache)
        suffix = actual(tokens[:, 3:], incremental_state=cache)
        torch.testing.assert_close(torch.cat([prefix, suffix], 1), out, atol=2e-5, rtol=2e-4)
