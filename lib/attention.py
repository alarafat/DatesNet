import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self,
                 n_embd: int = None,
                 n_head: int = None,
                 has_causal_mask: bool = False,
                 has_attn_bias: bool = False,
                 has_proj_bias: bool = False,
                 attn_dropout: float = 0.0,
                 out_dropout: float = 0.0
                 ):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.has_causal_mask = has_causal_mask

        assert self.n_embd % self.n_head == 0, "Embedding dimension must be a multiple of number of heads"

        self.dk = self.n_embd // self.n_head

        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=has_attn_bias)  # (b, t, c) -> (b, t, 3c)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=has_proj_bias)  # (b, t, c) -> (b, t, c)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.out_dropout = nn.Dropout(out_dropout)

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        self.flash = False

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.c_attn.weight)
        self.c_attn.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.c_proj.weight)
        self.c_proj.bias.data.fill_(0)

    def forward(self, x):
        b, t, c = x.shape  # here, "c == n_embd" and "t" is sequence_length

        # Compute the query, key, and value
        q, k, v = self.c_attn(x).split(self.n_embd, dim=-1)

        # So, we want to do group dot-product, the result will be in (b, t, n_head, c // n_head)
        intermediate_shape = (b, t, self.n_head, self.dk)
        q = q.view(intermediate_shape).transpose(1, 2)  # (b, nh, t, c//nh)
        k = k.view(intermediate_shape).transpose(1, 2)  # (b, nh, t, c//nh)
        v = v.view(intermediate_shape).transpose(1, 2)  # (b, nh, t, c//nh)

        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v,
                                                                   attn_mask=None,
                                                                   dropout_p=self.attn_dropout if self.training else 0,
                                                                   is_causal=self.has_causal_mask)
        else:
            # Dot product between Q and K as it's self attention
            attn = q @ k.transpose(-2, -1)  # (b, nh, t, t)
            scale_factor = (self.dk ** -0.5)
            attn *= scale_factor

            if self.has_causal_mask:
                mask = torch.ones_like(attn, dtype=torch.bool).triu(1)
                attn = attn.masked_fill_(mask, -torch.inf)

            attn = F.softmax(attn, dim=-1)

            attn = self.attn_dropout(attn)  # (b, nh, t, t)
            out = attn @ v  # (b, nh, t, c//nh)

        out = out.transpose(1, 2).contiguous().view(b, t, c)

        out = self.out_dropout(self.c_proj(out))        # (b, t, c)
        return out


class CrossAttention(nn.Module):
    def __init__(self,
                 n_embd: int = None,
                 n_embd_kv: int = None,
                 n_head: int = None,
                 has_causal_mask: bool = False,
                 has_attn_bias: bool = False,
                 has_proj_bias: bool = False,
                 attn_dropout: float = 0.0,
                 out_dropout: float = 0.0
                 ):
        super().__init__()
        self.n_embd_q = n_embd
        self.n_embd_kv = n_embd_kv
        self.n_head = n_head
        self.has_causal_mask = has_causal_mask

        assert n_embd % n_head == 0, "Embedding dimension must be a multiple of number of heads"

        self.dk = self.n_embd // self.n_head

        self.q = nn.Linear(self.n_embd_q, self.n_embd_q, bias=has_attn_bias)  # (b, t, c) -> (b, t, c)
        self.k = nn.Linear(self.n_embd_kv, self.n_embd_q, bias=has_attn_bias)  # (b, t, c) -> (b, t, c)
        self.v = nn.Linear(self.n_embd_kv, self.n_embd_q, bias=has_attn_bias)  # (b, t, c) -> (b, t, c)

        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=has_proj_bias)  # (b, t, c) -> (b, t, c)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.out_dropout = nn.Dropout(out_dropout)

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

    def forward(self, x, y):
        """
        x is the input to the decoder, goes as Q
        y is the output of the encoder, go as KV
        """
        b, t, c = x.shape  # here, "c == n_embd" and "t" is sequence_length

        # Compute the query, key, and value
        q = self.q(x)
        k = self.k(y)
        v = self.v(y)

        # So, we want to do group dot-product, the result will be in (b, t, n_head, c // n_head==dk)
        intermediate_shape = (b, t, self.n_head, self.dk)
        q = q.view(intermediate_shape).transpose(1, 2)  # (b, nh, t, dk)
        k = k.view(intermediate_shape).transpose(1, 2)  # (b, nh, t, dk)
        v = v.view(intermediate_shape).transpose(1, 2)  # (b, nh, t, dk)

        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v,
                                                                   attn_mask=None,
                                                                   dropout_p=self.attn_dropout if self.training else 0,
                                                                   is_causal=self.has_causal_mask)
        else:
            # Dot product between Q and K as it's self attention
            attn = q @ k.transpose(-2, -1)  # (b, nh, t, t)
            scale_factor = (self.dk ** -0.5)
            attn *= scale_factor

            if self.has_causal_mask:
                mask = torch.ones_like(attn, dtype=torch.bool).triu(1)
                attn = attn.masked_fill_(mask, -torch.inf)

            attn = F.softmax(attn, dim=-1)

            attn = self.attn_dropout(attn)  # (b, nh, t, t)
            out = attn @ v  # (b, nh, t, c//nh==dk)

        out = out.transpose(1, 2).contiguous().view(b, t, c)

        out = self.out_dropout(self.c_proj(out))
        return out
