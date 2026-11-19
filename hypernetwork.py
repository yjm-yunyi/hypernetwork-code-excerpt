import math

import torch
import torch.nn as nn


class HyperNet(nn.Module):
    def __init__(self, in_dim, targets, d=1024, r=8, k=16, nq=4, heads=8):
        super().__init__()
        self.names = list(targets)
        self.r, self.k = r, k
        self.inorm = nn.LayerNorm(in_dim)
        self.enc = nn.Sequential(nn.Linear(in_dim, d), nn.GELU(), nn.Linear(d, d))
        self.null = nn.Parameter(torch.zeros(1, d))
        self.queries = nn.Parameter(torch.randn(nq, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.to_z = nn.Sequential(nn.LayerNorm(nq * d), nn.Linear(nq * d, d), nn.GELU(), nn.Linear(d, d))
        self.headA = nn.ModuleDict(); self.headB = nn.ModuleDict()
        self.PA = nn.ParameterDict(); self.PB = nn.ParameterDict()
        for i, (n, (fin, fout)) in enumerate(targets.items()):
            key = str(i)
            self.headA[key] = nn.Linear(d, r * k)
            hb = nn.Linear(d, k * r); nn.init.zeros_(hb.weight); nn.init.zeros_(hb.bias); self.headB[key] = hb
            self.PA[key] = nn.Parameter(torch.randn(k, fin) / math.sqrt(fin))
            self.PB[key] = nn.Parameter(torch.randn(fout, k) / math.sqrt(k))

    def encode(self, feats):
        """feats [N, in_dim] (N may be 0) -> z [d]"""
        tok = self.null
        if feats.shape[0] > 0:
            tok = torch.cat([self.null, self.enc(self.inorm(feats))], 0)
        o, _ = self.attn(self.queries[None], tok[None], tok[None])
        return self.to_z(o.reshape(1, -1))[0]

    def deltas(self, z, dtype=torch.bfloat16):
        out = {}
        for i, n in enumerate(self.names):
            key = str(i)
            a = self.headA[key](z).view(self.r, self.k) @ self.PA[key]          # [r, in]
            b = self.PB[key] @ self.headB[key](z).view(self.k, self.r)          # [out, r]
            out[n] = (a.to(dtype), b.to(dtype))
        return out
