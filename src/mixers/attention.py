import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import math

try:
    from flash_attn import flash_attn_qkvpacked_func
    _flash_attn_available = True
except:
    print("Flash Attention is not available")
    _flash_attn_available = False

class SelfAttention(nn.Module):
    def __init__(self, attention_dropout=0.0, log_scores: bool = False):
        super().__init__()
        self.dropout_p = attention_dropout
        self.log_scores = log_scores

    def forward(self, qkv):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value. (B, S, 3, H, D)
            causal: if passed, will override self.causal
        """
        softmax_scale = 1.0 / math.sqrt(qkv.shape[-1])
        if _flash_attn_available and qkv.dtype in (torch.float16, torch.bfloat16):
            output = flash_attn_qkvpacked_func(qkv, dropout_p=self.dropout_p if self.training else 0.0, softmax_scale=softmax_scale, causal=True)
        else:
            seqlen = qkv.shape[1]
            q, k, v = qkv.unbind(dim=2)
            scores = torch.einsum("bthd,bshd->bhts", q, k * softmax_scale)
            causal_mask = torch.triu(
                torch.full((seqlen, seqlen), -10000.0, device=scores.device), 1
            )
            if self.log_scores:
                if not hasattr(self, "saved_scores_matrix"):
                    self.saved_scores_matrix = []
                self.saved_scores_matrix.append(scores.detach().cpu())
                
            scores = scores + causal_mask.to(dtype=scores.dtype)
            attention = torch.softmax(scores, dim=-1, dtype=v.dtype)
            if self.log_scores:
                self.saved_probs_matrix = attention.detach().cpu()
                
            attention_drop = F.dropout(attention, self.dropout_p if self.training else 0.0)
            output = torch.einsum("bhts,bshd->bthd", attention_drop, v)
        return output


class MHA(nn.Module):
    """Multi-head self-attention"""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
        layer_idx: int = None,
        log_scores: bool = False,
        **kwargs
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        assert self.d_model % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.d_model // num_heads
        self.Wqkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.inner_attn = SelfAttention(attention_dropout=dropout, log_scores=log_scores)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor):
        """"""
        qkv = self.Wqkv(x)
        qkv = rearrange(
            qkv, "... (three h d) -> ... three h d", three=3, d=self.head_dim
        )
        context = self.inner_attn(qkv)
        out = self.out_proj(rearrange(context, "... h d -> ... (h d)"))
        return out
