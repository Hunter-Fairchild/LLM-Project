import torch.nn as nn
import torch
import math


class GELU(nn.Module):
    """Gaussian Error Linear Unit (approximation)"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / math.pi)) *(x + 0.044715 * torch.pow(x, 3))))

class FeedForward(nn.Module):
    """A feed forward neural network module"""

    def __init__(self, cfg):
        super().__init__()

        # sequence of operators to apply
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),    # training weights
            GELU(),                                           # Gaussian error Linear Unit
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),    # training weights
        )

    def forward(self, x):
        """apply sequence of operators to input"""
        return self.layers(x)