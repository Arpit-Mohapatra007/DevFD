import math
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRAExpert(nn.Module):
    def __init__(self, in_dim, out_dim, r=8):
        super().__init__()
        self.r = r
        # B: down-projection (random, ~unit-norm columns), A: up-projection (zeros)
        self.B = nn.Parameter(torch.randn(in_dim, r) / math.sqrt(in_dim))
        self.A = nn.Parameter(torch.zeros(r, out_dim))

    def forward(self, x):
        return x @ self.B @ self.A


class DevelopmentalMoE(nn.Module):
    def __init__(self, original_ffn, in_dim, out_dim, r=8, tau=1.0, delta=0.15):
        super().__init__()
        self.original_ffn = original_ffn
        self.tau, self.delta = tau, delta

        # expert 0 = Real-LoRA, expert 1 = Fake-LoRA for task 1
        self.experts = nn.ModuleList([LoRAExpert(in_dim, out_dim, r),
                                      LoRAExpert(in_dim, out_dim, r)])
        self.router = nn.Linear(in_dim, 2, bias=False)
        nn.init.zeros_(self.router.weight)

        self.cur_labels = None   # set by DevFDViT.forward
        self.llb = None
        self.inp = None

    def add_expert(self, in_dim, out_dim, r=8):
        dev = self.router.weight.device
        new_exp = LoRAExpert(in_dim, out_dim, r).to(dev)

        # start the new B orthogonal to all old Fake-LoRA B's
        with torch.no_grad():
            prev = torch.cat([e.B for e in self.experts[1:]], dim=1)
            Q, _ = torch.linalg.qr(prev)
            new_exp.B -= Q @ (Q.T @ new_exp.B)
        self.experts.append(new_exp)

        # expand router, keep old rows
        new_router = nn.Linear(in_dim, len(self.experts), bias=False).to(dev)
        with torch.no_grad():
            new_router.weight.zero_()
            new_router.weight[:-1] = self.router.weight
        self.router = new_router

    def forward(self, x):
        base_out = self.original_ffn(x)
        expert_outputs = torch.stack([e(x) for e in self.experts], dim=0)   # [E,B,S,D]
        I = F.softmax(self.router(x) / self.tau, dim=-1)                    # [B,S,E]
        moe_out = torch.einsum('bse,ebsd->bsd', I, expert_outputs)

        # Label-guided localized balancing loss (per-sample response)
        self.llb = torch.zeros((), device=x.device)
        labels = self.cur_labels
        if labels is not None:
            E = len(self.experts)
            is_fake = (labels == 0)                                  # fake=0, real=1
            match = torch.zeros(labels.size(0), E, device=x.device)
            match[:, 0] = (~is_fake).float()                         # Real-LoRA <-> real
            match[:, 1:] = is_fake.float().unsqueeze(1)              # Fake-LoRAs <-> fake
            C = 1 + self.delta - 2 * self.delta * match              # 1-δ match, 1+δ mismatch
            W = I.mean(dim=1) * C                                    # [B,E]
            self.llb = torch.var(W) / (torch.mean(W) + 1e-8)

        self.inp = x.detach()
        return base_out + moe_out


class DevFDViT(nn.Module):
    def __init__(self):
        super().__init__()
        # CLIP ViT-B/16 as in the paper (pooled 768-d output with num_classes=0)
        self.vit = timm.create_model('vit_base_patch16_clip_224.openai',
                                     pretrained=True, num_classes=0)
        for p in self.vit.parameters():
            p.requires_grad = False

        self.hidden_dim = 768
        self.moe_layers = nn.ModuleList()
        for block in self.vit.blocks:
            moe = DevelopmentalMoE(block.mlp, self.hidden_dim, self.hidden_dim)
            block.mlp = moe
            self.moe_layers.append(moe)

        self.head = nn.Linear(self.hidden_dim, 1)

    def add_task(self):
        for moe in self.moe_layers:
            for e in moe.experts[1:]:      # freeze old Fake-LoRAs, keep Real-LoRA trainable
                for p in e.parameters():
                    p.requires_grad = False
            moe.add_expert(self.hidden_dim, self.hidden_dim)

    def freeze_old_router_rows(self):      # call after backward(), before optimizer.step()
        for moe in self.moe_layers:
            g = moe.router.weight.grad
            if g is not None:
                g[:-1] = 0

    def forward(self, x, labels=None):
        for m in self.moe_layers:
            m.cur_labels = labels
        feats = self.vit(x)                                        # [B,768]
        probs = torch.sigmoid(self.head(feats)).squeeze(-1)
        total_llb = sum(m.llb for m in self.moe_layers)
        moe_inputs = [m.inp for m in self.moe_layers]
        return probs, total_llb, moe_inputs
