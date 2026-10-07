import math
import os
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRAExpert(nn.Module):
    def __init__(self, in_dim, out_dim, r=8):
        super().__init__()
        self.r = r
        self.B = nn.Parameter(torch.randn(in_dim, r) / math.sqrt(in_dim))
        self.A = nn.Parameter(torch.zeros(r, out_dim))

    def forward(self, x):
        return x @ self.B @ self.A


class DevelopmentalMoE(nn.Module):
    def __init__(self, original_ffn, in_dim, out_dim, r=8, tau=1.0, delta=0.15):
        super().__init__()
        self.original_ffn = original_ffn
        self.tau, self.delta = tau, delta
        self.r = r

        self.experts = nn.ModuleList([LoRAExpert(in_dim, out_dim, r),
                                      LoRAExpert(in_dim, out_dim, r)])
        self.router = nn.Linear(in_dim, 2, bias=False)
        nn.init.zeros_(self.router.weight)

        self.cur_labels = None
        self.llb = None
        self.inp = None

    def add_expert(self, in_dim, out_dim, r=None):
        r = r or self.r
        dev = self.router.weight.device
        new_exp = LoRAExpert(in_dim, out_dim, r).to(dev)

        with torch.no_grad():
            prev = torch.cat([e.B for e in self.experts[1:]], dim=1)
            Q, _ = torch.linalg.qr(prev)
            new_exp.B -= Q @ (Q.T @ new_exp.B)
        self.experts.append(new_exp)

        new_router = nn.Linear(in_dim, len(self.experts), bias=False).to(dev)
        with torch.no_grad():
            new_router.weight.zero_()
            new_router.weight[:-1] = self.router.weight
        self.router = new_router

    def forward(self, x):
        base_out = self.original_ffn(x)
        expert_outputs = torch.stack([e(x) for e in self.experts], dim=0)
        I = F.softmax(self.router(x) / self.tau, dim=-1)
        moe_out = torch.einsum('bse,ebsd->bsd', I, expert_outputs)

        self.llb = torch.zeros((), device=x.device)
        labels = self.cur_labels
        if labels is not None:
            E = len(self.experts)
            is_fake = (labels == 0)
            match = torch.zeros(labels.size(0), E, device=x.device)
            match[:, 0] = (~is_fake).float()
            match[:, 1:] = is_fake.float().unsqueeze(1)
            C = 1 + self.delta - 2 * self.delta * match
            W = I.mean(dim=1) * C
            self.llb = torch.var(W) / (torch.mean(W) + 1e-8)

        self.inp = x.detach()
        return base_out + moe_out


class DevFDViT(nn.Module):
    def __init__(self, rank=8, sbi_ckpt='clip_sbi.pt'):
        super().__init__()
        self.rank = rank
        self.vit = timm.create_model('vit_base_patch16_clip_224.openai',
                                     pretrained=True, num_classes=0)

        ckpt = None
        if sbi_ckpt and os.path.exists(sbi_ckpt):
            ckpt = torch.load(sbi_ckpt, map_location='cpu')
            self.vit.load_state_dict(ckpt['vit'])
            print(f"Loaded SBI-pretrained backbone from {sbi_ckpt}")

        for p in self.vit.parameters():
            p.requires_grad = False

        self.hidden_dim = 768
        self.moe_layers = nn.ModuleList()
        for block in self.vit.blocks:
            moe = DevelopmentalMoE(block.mlp, self.hidden_dim, self.hidden_dim, r=rank)
            block.mlp = moe
            self.moe_layers.append(moe)

        self.head = nn.Linear(self.hidden_dim, 1)
        if ckpt is not None:
            self.head.load_state_dict(ckpt['head'])

    def add_task(self, freeze_real=False):
        for moe in self.moe_layers:
            if freeze_real:
                for p in moe.experts[0].parameters():
                    p.requires_grad = False
            for e in moe.experts[1:]:
                for p in e.parameters():
                    p.requires_grad = False
            moe.add_expert(self.hidden_dim, self.hidden_dim, self.rank)

    def freeze_old_router_rows(self):
        for moe in self.moe_layers:
            g = moe.router.weight.grad
            if g is not None:
                g[:-1] = 0

    def forward(self, x, labels=None):
        for m in self.moe_layers:
            m.cur_labels = labels
        feats = self.vit(x)
        probs = torch.sigmoid(self.head(feats)).squeeze(-1)
        total_llb = sum(m.llb for m in self.moe_layers)
        moe_inputs = [m.inp for m in self.moe_layers]
        return probs, total_llb, moe_inputs
