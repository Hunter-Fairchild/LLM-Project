import torch
import torch.nn as nn
from Architecture.Transformer_Block import TransformerBlock
from Architecture.LayerNormalization import LayerNorm

class GPTModel(nn.Module):
    """GPT model architecture implementation"""

    def __init__(self, cfg):
        """Constructor for GPT Model"""
        super().__init__()

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])

        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential( *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])

        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        """computation for each forward pass"""
        batch_size, seq_len = in_idx.shape

        tok_embeds = self.tok_emb(in_idx)                  # token embedding

        pos_embeds = self.pos_emb(                         # positional embedding
            torch.arange(seq_len, device=in_idx.device)
        )

        x = tok_embeds + pos_embeds                        # final embedding

        x = self.drop_emb(x)                               # drop (random) cells

        x = self.trf_blocks(x)                             # apply sequence of transformer blocks

        x = self.final_norm(x)                             # normalize output

        logits = self.out_head(x)                          # apply final linear transformation

        return logits                                      # return scores for vocab