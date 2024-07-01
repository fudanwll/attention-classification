import torch
import torch.nn as nn
import torch.nn.functional as F

# class SelfAttention(nn.Module):
#     def __init__(self, dim_q, dim_k, dim_v, **args):
#         super(SelfAttention,self).__init__()
#         self.dim_q = dim_q
#         self.dim_k = dim_k
#         self.dim_v = dim_v

#         self.linear_q = nn.Linear(dim_q, dim_k, bias=False)
#         self.linear_k = nn.Linear(dim_q, dim_k, bias=False)
#         self.linear_v = nn.Linear(dim_q, dim_v, bias=False)
#         self._norm_fact = 1 / sqrt(dim_k)
    
#     def forward(self,x):
#         # x: batch, n , dim_q
#         batch, n , dim_q = x.shape
#         assert dim_q == self.dim_q

#         q = self.linear_q(x)  # batch, n, dim_k
#         k = self.linear_k(x)  # batch, n, dim_k
#         v = self.linear_v(x)  # batch, n, dim_v

#         dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact
#         dist = torch.softmax(dist, dim=-1)

#         att = torch.bmm(dist,v)
#         return att

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embed size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask=None):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Attention mechanism
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out
