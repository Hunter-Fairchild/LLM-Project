import torch
import torch.nn as nn
import math



class MultiHeadAttention(nn.Module):
    """Causal, Multi-head Attention Unit with dropout"""

    def __init__(self, d_in, d_out,context_length, dropout, num_heads, qkv_bias=False):
        """constructor for efficient MultiHeadAttention"""
        super().__init__()
        assert (d_out % num_heads == 0)    # d_out must be divisible by num_heads

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias) # weighted linear change of variables
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out)              # combines heads together???

        self.dropout = nn.Dropout(dropout)                   # mask to drop random values

        self.register_buffer(                                # used for better computation in torch (gpu)
            "mask",
            torch.triu(torch.ones(context_length, context_length),
            diagonal=1)
        )

    def forward(self, x):
        """computes one-step of the module"""
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attn_scores.masked_fill_(mask_bool, -math.inf)

        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)

        context_vec = context_vec.contiguous().view(
            b, num_tokens, self.d_out
        )
        context_vec = self.out_proj(context_vec)

        return context_vec



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

def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    """"""
    for _ in range(max_new_tokens):              # loop for number of tokens

        idx_cond = idx[:, -context_size:]        # last (context_size) inputs

        with torch.no_grad():
            logits = model(idx_cond)             # compute scores

        logits = logits[:, -1, :]                # check last score (for next word)

        if top_k is not None:                                            # apply top-k sampling
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(                                        # apply threshold
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )

        if temperature > 0.0:                                            # temperature scaling
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)        # greedy decoding

        if idx_next == eos_id:break                                      # failed step (???)

        idx = torch.cat((idx, idx_next), dim=1)                          # add the new word to the (growing) list

    return idx                                    # return the list of token id's

