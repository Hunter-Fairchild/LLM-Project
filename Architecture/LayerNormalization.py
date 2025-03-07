import torch
import torch.nn as nn
import math

class LayerNorm(nn.Module):
    """defines the Layer Normalization to be used"""

    def __init__(self, emb_dim):
        """constructor for Layer-Normalization Unit"""
        super().__init__()

        self.eps = 1e-5

        self.scale = nn.Parameter(torch.ones(emb_dim))      # Training Parameters
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        """computation each call"""

        mean = x.mean(dim=-1, keepdim=True)                  # compute mean
        var = x.var(dim=-1, keepdim=True, unbiased=False)    # compute variance

        norm_x = (x - mean) / torch.sqrt(var + self.eps)     # adjust to standard normal

        return self.scale * norm_x + self.shift              # apply weights and return output