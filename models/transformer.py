# Copyright 2021 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Transformer model (PyTorch version)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import common_layers


class TransformerBlock(nn.Module):
    """Transformer layer."""

    def __init__(self, qkv_dim, mlp_dim, num_heads, dropout_rate=0.1, attention_dropout_rate=0.1):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(qkv_dim)
        self.attn = nn.MultiheadAttention(qkv_dim, num_heads, dropout=attention_dropout_rate, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.layernorm2 = nn.LayerNorm(qkv_dim)
        self.mlp = common_layers.MlpBlock(mlp_dim, out_dim=qkv_dim, dropout_rate=dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x, src_key_padding_mask=None):
        # Attention block
        residual = x
        x = self.layernorm1(x)
        attn_output, _ = self.attn(x, x, x, key_padding_mask=src_key_padding_mask)
        x = self.dropout1(attn_output)
        x = x + residual

        # MLP block
        residual2 = x
        x = self.layernorm2(x)
        x = self.mlp(x)
        x = self.dropout2(x)
        x = x + residual2
        return x


class TransformerEncoder(nn.Module):
    """Transformer Model Encoder."""

    def __init__(self, vocab_size, emb_dim=512, num_heads=8, num_layers=6, qkv_dim=512, mlp_dim=2048,
                 max_len=512, dropout_rate=0.1, attention_dropout_rate=0.1, learn_pos_emb=False,
                 classifier=False, classifier_pool='CLS', num_classes=10, tied_weights=False):
        super().__init__()
        self.emb_dim = emb_dim
        self.classifier = classifier
        self.classifier_pool = classifier_pool
        self.num_classes = num_classes
        self.tied_weights = tied_weights
        self.embedding = common_layers.Embed(vocab_size, emb_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.max_len = max_len+1 if classifier_pool=='CLS' else max_len
        self.pos_emb = common_layers.AddPositionEmbs(max_len=self.max_len, posemb_init=None if not learn_pos_emb else nn.init.normal_)
        self.dropout = nn.Dropout(dropout_rate)
        self.layers = nn.ModuleList([
            TransformerBlock(qkv_dim, mlp_dim, num_heads, dropout_rate, attention_dropout_rate)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(qkv_dim)
        if self.classifier:
            self.classifier_layer = common_layers.ClassifierHead(num_classes, qkv_dim, mlp_dim, self.classifier_pool)

    def forward(self, inputs, inputs_positions=None, inputs_segmentation=None):
        # inputs: [batch, seq_len]
        src_padding_mask = (inputs == 0)  # True for PAD tokens

        x = self.embedding(inputs)
        if self.classifier and self.classifier_pool == 'CLS':
            batch_size = x.shape[0]
            cls = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls, x], dim=1)
            # update mask for CLS token (not masked)
            src_padding_mask = torch.cat([torch.zeros((batch_size, 1), dtype=torch.bool, device=x.device), src_padding_mask], dim=1)

        x = self.pos_emb(x, inputs_positions)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_padding_mask)

        x = self.norm(x)

        if self.classifier:
            x = self.classifier_layer(x)
        return x


class TransformerDualEncoder(nn.Module):
    """Transformer Model for Matching (dual encoding) tasks."""

    def __init__(self, vocab_size, emb_dim=512, num_heads=8, num_layers=6, qkv_dim=512, mlp_dim=2048,
                 max_len=2048, dropout_rate=0.1, attention_dropout_rate=0.1, classifier=True,
                 classifier_pool='CLS', num_classes=2, interaction=None):
        super().__init__()
        self.encoder = TransformerEncoder(
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            qkv_dim=qkv_dim,
            mlp_dim=mlp_dim,
            max_len=max_len,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            classifier=False,
            classifier_pool=classifier_pool,
            num_classes=num_classes
        )
        self.classifier = classifier
        self.classifier_pool = classifier_pool
        self.num_classes = num_classes
        self.mlp_dim = mlp_dim
        self.interaction = interaction
        if self.classifier:
            self.classifier_layer = common_layers.ClassifierHead_Dual(num_classes, qkv_dim, qkv_dim, mlp_dim, self.interaction, self.classifier_pool)

    def forward(self, inputs1, inputs2, inputs1_positions=None, inputs2_positions=None,
                inputs1_segmentation=None, inputs2_segmentation=None):
        inputs1_encoded = self.encoder(inputs1, inputs1_positions, inputs1_segmentation)
        inputs2_encoded = self.encoder(inputs2, inputs2_positions, inputs2_segmentation)
        if self.classifier:
            x = self.classifier_layer(inputs1_encoded, inputs2_encoded)
            return x
        else:
            return inputs1_encoded,inputs2_encoded