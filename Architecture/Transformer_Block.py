import torch.nn as nn
from Architecture.Attention_Mechanism import MultiHeadAttention
from Architecture.FeedForward import FeedForward
from Architecture.LayerNormalization import LayerNorm

class TransformerBlock(nn.Module):
    """Transformer block component of GPT"""

    def __init__(self, cfg):
        """constructor for the transformer block"""
        super().__init__()

        self.att = MultiHeadAttention(                      # Multi-Attention Head
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])

        self.ff = FeedForward(cfg)                           # Feed-Forward Module

        self.norm1 = LayerNorm(cfg["emb_dim"])               # Normalize Outputs
        self.norm2 = LayerNorm(cfg["emb_dim"])

        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])    # (random) dropout

    def forward(self, x):
        """computation performed in each Transformer Block pass"""
        shortcut = x   # first input

        x = self.norm1(x)            # apply (pre) layer norm
        x = self.att(x)              # apply Multi-Head Attention
        x = self.drop_shortcut(x)    # Apply Dropout
        x = x + shortcut             # apply shortcut (add first input)

        shortcut = x                 # first output

        x = self.norm2(x)            # apply (pre) layer norm
        x = self.ff(x)               # apply feed-forward module
        x = self.drop_shortcut(x)    # apply Dropout
        x = x + shortcut             # apply shortcut (add first output)

        return x