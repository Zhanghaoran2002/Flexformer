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
"""Common layers used in models (PyTorch version)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Embed(nn.Module):
    """Embedding Module."""
    def __init__(self, num_embeddings, features, mode='input', emb_init=None):
        super().__init__()
        self.mode = mode
        self.embedding = nn.Parameter(torch.empty(num_embeddings, features))
        if emb_init is None:
            nn.init.normal_(self.embedding, std=1.0)
        else:
            emb_init(self.embedding)

    def forward(self, inputs):
        if self.mode == 'input':
            if inputs.dtype.is_floating_point or inputs.dtype.is_complex:
                raise ValueError('Input type must be an integer or unsigned integer.')
            return F.embedding(inputs, self.embedding)
        elif self.mode == 'output':
            # inputs: [batch, len, d], embedding: [vocab, d]
            # output: [batch, len, vocab]
            return torch.einsum('bld,vd->blv', inputs, self.embedding)
        else:
            raise ValueError('Unknown mode: {}'.format(self.mode))


def sinusoidal_init(max_len=2048):
    """1D Sinusoidal Position Embedding Initializer."""
    def init(shape, dtype=torch.float32):
        d_feature = shape[-1]
        pe = np.zeros((max_len, d_feature), dtype=np.float32)
        position = np.arange(0, max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_feature, 2) * -(np.log(10000.0) / d_feature))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        pe = pe[np.newaxis, :, :]  # [1, max_len, d_feature]
        return torch.tensor(pe, dtype=dtype)
    return init


class AddPositionEmbs(nn.Module):
    """Adds (optionally learned) positional embeddings to the inputs."""

    def __init__(self, max_len=512, posemb_init=None):
        super().__init__()
        self.max_len = max_len
        self.posemb_init = posemb_init
        self.pos_embedding = None

    def build(self, emb_dim, device=None):
        pos_emb_shape = (1, self.max_len, emb_dim)
        
        if self.posemb_init is None:
            init_tensor = sinusoidal_init(self.max_len)(pos_emb_shape)
        else:
            init_tensor = torch.empty(pos_emb_shape)
            self.posemb_init(init_tensor)

        if device is not None:
            init_tensor = init_tensor.to(device)
        
        self.pos_embedding = nn.Parameter(init_tensor, requires_grad=(self.posemb_init is not None))
        self.emb_dim = emb_dim 

    def forward(self, inputs, inputs_positions=None):
        # inputs: [batch, seq_len, emb_dim]
        if self.pos_embedding is None or self.pos_embedding.shape[-1] != inputs.shape[-1] or self.pos_embedding.device != inputs.device:          
            self.build(inputs.shape[-1], device=inputs.device)
        length = inputs.shape[1]
        pe = self.pos_embedding[:, :length, :]
        if inputs_positions is None:
            return inputs + pe
        else:
            # for packed data we need to use known position indices:
            pe = pe[0]
            pe = pe[inputs_positions]
            return inputs + pe


class MlpBlock(nn.Module):
    """Transformer MLP block."""

    def __init__(self, mlp_dim, out_dim=None, dropout_rate=0.1):
        super().__init__()
        self.mlp_dim = mlp_dim
        self.out_dim = out_dim
        self.dropout_rate = dropout_rate
        self.dense1 = None
        self.dense2 = None
        self.bulit = False

    def build(self, input_dim, device=None):
        out_dim = input_dim if self.out_dim is None else self.out_dim
        self.dense1 = nn.Linear(input_dim, self.mlp_dim)
        self.dense2 = nn.Linear(self.mlp_dim, out_dim)
        if device is not None:
            self.dense1 = self.dense1.to(device)
            self.dense2 = self.dense2.to(device)
        self.bulit = True

    def forward(self, inputs, deterministic=False):
        if self.dense1 is None or self.dense1.in_features != inputs.shape[-1] and not self.bulit:
            self.build(inputs.shape[-1], device=inputs.device)
        x = self.dense1(inputs)
        x = F.gelu(x)
        x = F.dropout(x, p=self.dropout_rate, training=not deterministic)
        output = self.dense2(x)
        output = F.dropout(output, p=self.dropout_rate, training=not deterministic)
        return output

class ClassifierHead(nn.Module):
    """Classifier head module."""
    def __init__(self, num_classes, input_dim, mlp_dim, pooling_mode='MEAN'):
        super().__init__()
        self.pooling_mode = pooling_mode
        self.mlp = nn.Linear(input_dim, mlp_dim)
        self.logits = nn.Linear(mlp_dim, num_classes)
    
    def forward(self, x):
        # Pooling operations
        if self.pooling_mode == 'MEAN':
            x = x.mean(dim=1)
        elif self.pooling_mode == 'SUM':
            x = x.sum(dim=1)
        elif self.pooling_mode == 'FLATTEN':
            x = x.view(x.size(0), -1)
        elif self.pooling_mode == 'CLS':
            x = x[:, 0]
        else:
            raise NotImplementedError(f'Pooling mode "{self.pooling_mode}" not supported.')
        
        x = F.relu(self.mlp(x))
        return self.logits(x)

    
class ClassifierHead_Dual(nn.Module):
    """Classifier head module."""
    def __init__(self, num_classes, input_dim1, input_dim2, mlp_dim, interaction= None,pooling_mode='MEAN'):
        super().__init__()
        self.pooling_mode = pooling_mode
        self.mlp = nn.Linear(input_dim1+input_dim2, mlp_dim)
        self.mlp2 = nn.Linear(mlp_dim, mlp_dim//2)
        self.logits = nn.Linear(mlp_dim//2, num_classes)
        self.interaction = interaction
    
    def forward(self, x, y):
        # Pooling operations
        if self.pooling_mode == 'MEAN':
            x = x.mean(dim=1)
            y = y.mean(dim=1)
        elif self.pooling_mode == 'SUM':
            x = x.sum(dim=1)
            y = y.sum(dim=1)
        elif self.pooling_mode == 'FLATTEN':
            x = x.view(x.size(0), -1)
            y = y.view(y.size(0), -1)
        elif self.pooling_mode == 'CLS':
            x = x[:, 0]
            y = y[:, 0]
        else:
            raise NotImplementedError(f'Pooling mode "{self.pooling_mode}" not supported.')
        if self.interaction == 'NLI':
            encoded = torch.cat([x, y, x * y, x - y], dim=1)
        else:
            encoded = torch.cat([x, y], dim=1)
        x = F.relu(self.mlp(encoded))
        x = F.relu(self.mlp2(x))
        return self.logits(x)