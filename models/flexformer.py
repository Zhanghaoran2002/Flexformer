import torch
import torch.nn as nn
import torch.nn.functional as F
from models import common_layers
from models.causal_linear_attention import causal_linear_attention
import math

class lnk_kernel(nn.Module):
    def __init__(self, head_dim, num_freqs, num_heads=1):
        super().__init__()
        self.head_dim = head_dim
        self.num_freqs = num_freqs
        self.num_heads = num_heads
        
        self.omega1_weight = nn.Parameter(torch.empty(self.num_heads, self.head_dim, self.num_freqs))
        self.omega2_weight = nn.Parameter(torch.empty(self.num_heads, self.head_dim, self.num_freqs))
        self.omega1_bias = nn.Parameter(torch.empty(self.num_heads, self.num_freqs))
        self.omega2_bias = nn.Parameter(torch.empty(self.num_heads, self.num_freqs))
        self.log_tau = nn.Parameter(torch.empty(self.num_heads))
        

        nn.init.xavier_uniform_(self.omega1_weight)
        nn.init.xavier_uniform_(self.omega2_weight)
        nn.init.constant_(self.omega1_bias, 0.0)
        nn.init.constant_(self.omega2_bias, 0.0)
        nn.init.constant_(self.log_tau, math.log(2*self.head_dim**0.5))

    def forward(self, x):
        B, H, L, D = x.shape
        assert H == self.num_heads, f"Expected {self.num_heads} heads, got {H}"
        

        prod1 = torch.einsum('bhld,hde->bhle', x, self.omega1_weight)
        prod2 = torch.einsum('bhld,hde->bhle', x, self.omega2_weight)

        prod1 = prod1 + self.omega1_bias.unsqueeze(1) # (H, D) -> (H, 1, D) -> broadcast to (B, H, L, D)
        prod2 = prod2 + self.omega2_bias.unsqueeze(1)
        
        prod1 = torch.sigmoid(prod1) * (torch.pi / 2)
        prod2 = torch.sigmoid(prod2) * torch.pi
        
        phi1 = torch.cat([torch.cos(prod1), torch.sin(prod1)], dim=-1)  # (B, H, L, 2*D)
        phi2 = torch.cat([torch.cos(prod2), torch.cos(prod2)], dim=-1)  # (B, H, L, 2*D)
        
        phi_x = torch.mul(phi1,phi2)
        return  F.relu(phi_x)
    
    
def lnk_linear_attention(q, k, v, lnk_kernel, eps=1e-6, *, causal=False,
                         key_padding_mask=None, initial_state=None,
                         return_state=False, backend='auto'):
    """LNK attention; causal=True enables the rectangular recurrent scan."""
    k_norm = torch.norm(k, p=2, dim=-1, keepdim=True)
    tau = torch.exp(lnk_kernel.log_tau).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # (1, H, 1, 1)
    phi_q = lnk_kernel(q)
    phi_k = (lnk_kernel(k) * torch.exp(k_norm / tau)).to(phi_q.dtype)
    # Fourier biases can promote features to FP32 under autocast. Match V to
    # that dtype while keeping this cast differentiable back to v_proj.
    v = v.to(phi_q.dtype)
    valid_mask = None
    if key_padding_mask is not None:
        if key_padding_mask.shape != (k.shape[0], k.shape[2]) or key_padding_mask.dtype != torch.bool:
            raise ValueError('key_padding_mask must be boolean [B,T], True for padding.')
        valid_mask = ~key_padding_mask
    if causal:
        return causal_linear_attention(
            phi_q, phi_k, v, valid_mask=valid_mask,
            initial_state=initial_state, return_state=return_state,
            eps=eps, backend=backend,
        )
    if initial_state is not None or return_state:
        raise ValueError('State caching requires causal=True.')
    # Mask AFTER the nonlinear map: phi(0) is not zero.
    if valid_mask is not None:
        phi_k = phi_k.masked_fill(~valid_mask[:, None, :, None], 0.0)
    kv_sum = torch.einsum('bhnd,bhne->bhde', phi_k, v)
    
    Z = phi_k.sum(dim=-2)  # (B, H, 2*R)
    
    numerator = torch.einsum('bhnd,bhde->bhne', phi_q, kv_sum)
    denominator = torch.einsum('bhnd,bhd->bhn', phi_q, Z)
    denominator = denominator.unsqueeze(-1) + eps  # (B, H, T, 1)
    out = numerator / denominator  # (B, H, T, D_v)
    return out

class LinearAttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads, num_freq, dropout_rate=0.1, attention_dropout_rate=0.1,
                 causal=False, attention_backend='auto'):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.causal = causal
        self.attention_backend = attention_backend
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout_rate)
        self.attn_dropout = nn.Dropout(attention_dropout_rate)
        self.lnk_kernel = lnk_kernel(self.head_dim, num_freq, num_heads)

    def forward(self, x, key_padding_mask=None, initial_state=None, return_state=False):
        B, T, _ = x.shape
        H, D = self.num_heads, self.head_dim

        # Q, K, V: (B, T, d_model)
        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        attn_output = lnk_linear_attention(
            q, k, v, self.lnk_kernel, causal=self.causal,
            key_padding_mask=key_padding_mask, initial_state=initial_state,
            return_state=return_state, backend=self.attention_backend,
        )
        if return_state:
            attn_output, final_state = attn_output
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, -1)  # (B, T, d_model)
        attn_output = self.out_proj(attn_output)
        attn_output = self.attn_dropout(attn_output)

        return (attn_output, final_state) if return_state else attn_output

    # Preserve the previous public helper without accidental method binding.
    lnk_linear_attention = staticmethod(lnk_linear_attention)


class LinearTransformerBlock(nn.Module):

    def __init__(self, qkv_dim, mlp_dim, num_heads, num_freq, dropout_rate=0.1, attention_dropout_rate=0.1,
                 causal=False, attention_backend='auto'):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(qkv_dim)
        self.attn = LinearAttentionLayer(qkv_dim, num_heads, num_freq, dropout_rate,
                                         attention_dropout_rate, causal, attention_backend)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.layernorm2 = nn.LayerNorm(qkv_dim)
        self.mlp = common_layers.MlpBlock(mlp_dim, out_dim=qkv_dim, dropout_rate=dropout_rate)
        # Register MLP parameters before an optimizer or checkpoint is created.
        self.mlp.build(qkv_dim)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x, src_key_padding_mask=None, initial_state=None, return_state=False):
        # Attention block
        residual = x
        x = self.layernorm1(x)
        attn_output = self.attn(x, key_padding_mask=src_key_padding_mask,
                                initial_state=initial_state, return_state=return_state)
        if return_state:
            attn_output, final_state = attn_output
        x = self.dropout1(attn_output)
        x = x + residual

        # MLP block
        residual2 = x
        x = self.layernorm2(x)
        x = self.mlp(x, deterministic=not self.training)
        x = self.dropout2(x)
        x = x + residual2
        return (x, final_state) if return_state else x


class FlexformerEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=512, num_heads=8, num_layers=6, qkv_dim=512, mlp_dim=2048,
                 max_len=512, num_freq=32, dropout_rate=0.1, attention_dropout_rate=0.1, learn_pos_emb=False,
                 classifier=False, classifier_pool='CLS', num_classes=10, tied_weights=False,
                 causal=False, attention_backend='auto'):
        super().__init__()
        if causal and classifier and classifier_pool == 'CLS':
            raise ValueError('A prepended CLS token cannot summarize future tokens in causal mode.')
        if causal and emb_dim != qkv_dim:
            raise ValueError('Causal Flexformer requires emb_dim == qkv_dim.')
        self.emb_dim = emb_dim
        self.causal = causal
        self.classifier = classifier
        self.classifier_pool = classifier_pool
        self.num_classes = num_classes
        self.tied_weights = tied_weights
        self.embedding = common_layers.Embed(vocab_size, emb_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim)) if classifier and classifier_pool == 'CLS' else None
        model_max_len = max_len + 1 if classifier and classifier_pool == 'CLS' else max_len
        self.pos_emb = common_layers.AddPositionEmbs(max_len=model_max_len, posemb_init=None if not learn_pos_emb else nn.init.normal_)
        self.pos_emb.build(emb_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.layers = nn.ModuleList([
            LinearTransformerBlock(qkv_dim, mlp_dim, num_heads, num_freq, dropout_rate,
                                   attention_dropout_rate, causal, attention_backend)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(qkv_dim)
        if self.classifier:
            self.classifier_layer = common_layers.ClassifierHead(num_classes, qkv_dim, mlp_dim, self.classifier_pool)

    def forward(self, inputs, inputs_positions=None, inputs_segmentation=None, incremental_state=None):
        if incremental_state is not None and (not self.causal or self.classifier):
            raise ValueError('Incremental states require a causal model without a classifier.')
        if self.causal and inputs_segmentation is not None:
            raise NotImplementedError('Packed-sequence segment resets are not supported in causal mode.')
        if inputs.is_cuda:
            src_padding_mask = torch.where(inputs == 0, 
                                          torch.tensor(True, device=inputs.device),
                                          torch.tensor(False, device=inputs.device))
        else:
            src_padding_mask = (inputs == 0)

        x = self.embedding(inputs)

        if self.cls_token is not None:
            batch_size = x.shape[0]
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
            src_padding_mask = torch.cat([
                torch.zeros((batch_size, 1), dtype=torch.bool, device=x.device),
                src_padding_mask
            ], dim=1)
        offset = incremental_state.get('position', 0) if incremental_state is not None else 0
        if self.causal:
            if inputs_positions is None:
                if offset + x.shape[1] > self.pos_emb.max_len:
                    raise ValueError('Sequence position exceeds max_len, including cached tokens.')
                positions = torch.arange(offset, offset + x.shape[1], device=x.device)
            else:
                positions = inputs_positions
                if positions.shape not in ((x.shape[1],), (x.shape[0], x.shape[1])):
                    raise ValueError('inputs_positions must have shape [T] or [B,T].')
            x = x + self.pos_emb.pos_embedding[0][positions]
        else:
            x = self.pos_emb(x, inputs_positions)
        x = self.dropout(x)
        previous = incremental_state.get('layers') if incremental_state is not None else None
        if previous is not None and len(previous) != len(self.layers):
            raise ValueError('Cached state has the wrong number of layers.')
        next_states = []
        for index, layer in enumerate(self.layers):
            if incremental_state is None:
                x = layer(x, src_key_padding_mask=src_padding_mask)
            else:
                x, final_state = layer(
                    x, src_key_padding_mask=src_padding_mask,
                    initial_state=None if previous is None else previous[index], return_state=True,
                )
                next_states.append(final_state)
        x = self.norm(x)
        if self.classifier:
            x = self.classifier_layer(x)

        if incremental_state is not None:
            incremental_state.update(position=offset + inputs.shape[1], layers=next_states)
        return x


class FlexformerDecoder(FlexformerEncoder):
    """Causal language model returning [B,T,vocab_size] unnormalized logits.

    Pass a mutable incremental_state={} for prefill/decoding; subsequent calls
    should contain only NEW tokens. The cache holds per-layer (S,z) and the
    positional offset. Use eval() and torch.no_grad() for generation.
    """

    def __init__(self, vocab_size, emb_dim=512, num_heads=8, num_layers=6,
                 qkv_dim=512, mlp_dim=2048, max_len=512, num_freq=32,
                 dropout_rate=0.1, attention_dropout_rate=0.1,
                 learn_pos_emb=False, tied_weights=False, attention_backend='auto'):
        super().__init__(
            vocab_size=vocab_size, emb_dim=emb_dim, num_heads=num_heads,
            num_layers=num_layers, qkv_dim=qkv_dim, mlp_dim=mlp_dim,
            max_len=max_len, num_freq=num_freq, dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            learn_pos_emb=learn_pos_emb, classifier=False, tied_weights=tied_weights,
            causal=True, attention_backend=attention_backend,
        )
        self.output_proj = nn.Linear(qkv_dim, vocab_size, bias=False)
        if tied_weights:
            self.output_proj.weight = self.embedding.embedding

    def forward(self, inputs, inputs_positions=None, inputs_segmentation=None, incremental_state=None):
        features = super().forward(
            inputs, inputs_positions, inputs_segmentation, incremental_state
        )
        return self.output_proj(features)
    
