import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AttentionHead(nn.Module):
    def __init__(self, emb_dim, att_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.att_dim = att_dim
        self.norm_const = math.sqrt(att_dim)
        self.key = nn.Linear(emb_dim, att_dim, bias=False)
        self.query = nn.Linear(emb_dim, att_dim, bias=False)
        self.value = nn.Linear(emb_dim, att_dim, bias=False)

    def forward(self, x, mask=None):
        assert x.shape[1] == self.emb_dim

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        assert Q.shape == (x.shape[0], self.att_dim)
        assert K.shape == (x.shape[0], self.att_dim)
        assert V.shape == (x.shape[0], self.att_dim)

        scores = torch.matmul(Q, K.transpose(1, 0)) / self.norm_const

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        scores = F.softmax(scores, dim=0)

        assert scores.shape == (x.shape[0], x.shape[0])
        assert torch.allclose(torch.sum(scores, dim=0), torch.ones(x.shape[0]))

        attention = torch.matmul(scores, V)
        assert attention.shape == (x.shape[0], self.att_dim)

        return attention


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, emb_dim, att_dim, out_dim):
        super().__init__()
        self.n_heads = n_heads
        self.emb_dim = emb_dim
        self.att_dim = att_dim
        self.out_dim = out_dim
        self.heads = nn.ModuleList([AttentionHead(emb_dim, att_dim) for _ in range(n_heads)])
        self.O = nn.Linear(n_heads * att_dim, out_dim)

    def forward(self, x, mask=None):
        out = torch.concat([head(x, mask) for head in self.heads], dim=-1)
        out = self.O(out)

        assert out.shape == (x.shape[0], self.out_dim)

        return out


class FeedForward(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, 4*in_features),
            nn.ReLU(),
            nn.Linear(4*in_features, out_features)
        )

    def forward(self, x):
        return self.model(x)


class LayerNorm(nn.Module):
    def __init__(self, emb_dim, eps = 1e-5):
        super().__init__()
        self.emb_dim = emb_dim
        self.eps = eps
        self.gamma = torch.ones(emb_dim)
        self.beta = torch.zeros(emb_dim)

    def forward(self, x):
        x_mean = torch.mean(x, dim=1, keepdims=True)
        x_var = torch.var(x, dim=1, keepdims=True)
        return (x - x_mean) / (torch.sqrt(x_var + self.eps)) * self.gamma + self.beta


class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, n_heads, att_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.att_dim = att_dim
        self.mha = MultiHeadAttention(n_heads, emb_dim, att_dim, att_dim)
        self.ln1 = LayerNorm(att_dim)
        self.ffwd = FeedForward(att_dim, emb_dim)
        self.ln2 = LayerNorm(emb_dim)

    def forward(self, x):
        tmp = self.ln1(self.mha(x)) + x
        return self.ln2(self.ffwd(tmp)) + tmp


class Decoder(nn.Module):
    def __init__(self, n_blocks):
        super().__init__()
        self.blocks = nn.ModuleList(TransformerBlock())
        self.linear = nn.Linear()

    def forward(self, x):
        for block in self.block:
            x = block(x)
        logits = self.Linear(x)
        return F.softmax(logits, dim=0)