#!/usr/bin/env python3
"""
ERA V5 Assignment 11
Adam by hand, bias correction, warmup ratios, cosine vs WSD, width–LR sweep.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# MiniGPT: Pre-LN RMSNorm, causal MHA, SwiGLU, tied embeddings
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * (x / rms)


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        hs = C // self.n_head
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        att = att.masked_fill(~mask, float("-inf"))
        y = (F.softmax(att, dim=-1) @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class SwiGLUFFN(nn.Module):
    def __init__(self, n_embd: int, mult: int = 4):
        super().__init__()
        hidden = mult * n_embd
        self.w_gate = nn.Linear(n_embd, hidden, bias=False)
        self.w_up = nn.Linear(n_embd, hidden, bias=False)
        self.w_down = nn.Linear(hidden, n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        self.n1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.n2 = RMSNorm(n_embd)
        self.ffn = SwiGLUFFN(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.n1(x))
        x = x + self.ffn(self.n2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab=64, block_size=32, n_layer=2, n_head=4, n_embd=128):
        super().__init__()
        self.block_size = block_size
        self.n_embd = n_embd
        self.tok_emb = nn.Embedding(vocab, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList(Block(n_embd, n_head) for _ in range(n_layer))
        self.n_f = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        logits = self.lm_head(self.n_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


def nparams(m: nn.Module) -> int:
    # tied weights counted once
    seen = set()
    total = 0
    for p in m.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel()
    return total


def make_batch(vocab, T, B, device, rng, noise=0.0):
    starts = torch.randint(0, vocab, (B,), generator=rng)
    seq = torch.zeros(B, T + 1, dtype=torch.long)
    for b in range(B):
        s = int(starts[b])
        seq[b] = torch.arange(s, s + T + 1) % vocab
    if noise > 0:
        rnd = torch.randint(0, vocab, seq.shape, generator=rng)
        mask = torch.rand(seq.shape, generator=rng) < noise
        seq = torch.where(mask, rnd, seq)
    return seq[:, :-1].to(device), seq[:, 1:].to(device)


# ---------------------------------------------------------------------------
# Task 1 — Adam by hand vs PyTorch
# ---------------------------------------------------------------------------
def task1():
    print("\n========== TASK 1: Adam by hand vs PyTorch ==========")
    beta1, beta2, eps, eta = 0.9, 0.999, 1e-8, 1e-3
    w0 = 0.5
    grads = [0.4, -0.2, 0.1, 0.05, -0.3]

    # PyTorch
    w = torch.nn.Parameter(torch.tensor([w0], dtype=torch.float64))
    opt = torch.optim.Adam([w], lr=eta, betas=(beta1, beta2), eps=eps)
    pt_rows = []
    for g in grads:
        if w.grad is not None:
            w.grad.zero_()
        w.grad = torch.tensor([g], dtype=torch.float64)
        opt.step()
        st = opt.state[w]
        pt_rows.append(
            {
                "m": float(st["exp_avg"]),
                "v": float(st["exp_avg_sq"]),
                "w": float(w.detach()),
            }
        )

    # Hand
    m = v = 0.0
    wh = w0
    print(f"{'t':>3} {'g':>8} {'m':>12} {'v':>14} {'mhat':>12} {'vhat':>14} {'w_hand':>12} {'w_pt':>12} {'|dw|':>10}")
    max_err = 0.0
    for t, g in enumerate(grads, start=1):
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g * g)
        mhat = m / (1 - beta1**t)
        vhat = v / (1 - beta2**t)
        wh = wh - eta * mhat / (math.sqrt(vhat) + eps)
        err = abs(wh - pt_rows[t - 1]["w"])
        max_err = max(max_err, err)
        print(
            f"{t:3d} {g:8.3f} {m:12.8f} {v:14.8e} {mhat:12.8f} {vhat:14.8e} "
            f"{wh:12.8f} {pt_rows[t-1]['w']:12.8f} {err:10.2e}"
        )
    print(f"max |w_hand - w_pytorch| = {max_err:.3e}")
    print("MATCH" if max_err < 1e-10 else "CHECK")
    return grads, pt_rows


# ---------------------------------------------------------------------------
# Task 2 — bias correction on / off, 20 steps + longer horizon
# ---------------------------------------------------------------------------
def task2():
    print("\n========== TASK 2: Bias correction on vs off ==========")
    beta1, beta2, eps, eta = 0.9, 0.999, 1e-8, 1e-3
    g = 0.25  # constant gradient so the schedule is isolated
    n_plot = 20
    n_long = 400

    def run(correct: bool, steps: int):
        m = v = 0.0
        w = 0.5
        ws, dus = [w], []
        for t in range(1, steps + 1):
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * (g * g)
            if correct:
                mh, vh = m / (1 - beta1**t), v / (1 - beta2**t)
            else:
                mh, vh = m, v
            du = eta * mh / (math.sqrt(vh) + eps)
            w = w - du
            ws.append(w)
            dus.append(du)
        return ws, dus

    w_on, du_on = run(True, n_long)
    w_off, du_off = run(False, n_long)

    print(f"{'t':>4} {'du_corrected':>14} {'du_raw':>14} {'rel_diff':>12} {'factor':>10}")
    first_below_1pct = None
    first_below_10pct = None
    for t in range(1, n_long + 1):
        rel = abs(du_on[t - 1] - du_off[t - 1]) / (abs(du_on[t - 1]) + 1e-30)
        factor = math.sqrt(1 - beta2**t) / (1 - beta1**t)
        if t <= n_plot or t in (50, 100, 200, 300, 400):
            print(f"{t:4d} {du_on[t-1]:14.6e} {du_off[t-1]:14.6e} {rel:12.4%} {factor:10.4f}")
        if first_below_10pct is None and rel < 0.10:
            first_below_10pct = t
        if first_below_1pct is None and rel < 0.01:
            first_below_1pct = t

    # 1-beta2^t → 1 slowly; solve sqrt(1-0.999^t) ≈ 0.99 with beta1^t≈0
    t_1pct_theory = int(math.log(1 - 0.99**2) / math.log(beta2))
    print(f"relative update gap first <10% at step {first_below_10pct}")
    print(f"relative update gap first <1%  at step {first_below_1pct}")
    print(f"theory: |du| gap <1% needs t≈{t_1pct_theory} when β2=0.999")
    print("Note: raw Adam (no correction) takes a LARGER first step because v is still tiny.")
    print("Bias correction on v inflates hat_v early → smaller, safer first steps.")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ts = list(range(1, n_plot + 1))
    ax[0].plot(ts, du_on[:n_plot], "o-", label="bias-corrected Adam")
    ax[0].plot(ts, du_off[:n_plot], "s--", label="no bias correction")
    ax[0].set_title("Update size, first 20 steps")
    ax[0].set_xlabel("step")
    ax[0].legend()
    ax[1].plot(range(1, n_long + 1), [abs(a - b) / (abs(a) + 1e-30) for a, b in zip(du_on, du_off)])
    ax[1].axhline(0.01, color="k", ls=":", label="1%")
    if first_below_1pct:
        ax[1].axvline(first_below_1pct, color="C3", ls="--", label=f"<1% at {first_below_1pct}")
    ax[1].set_title("Relative update difference")
    ax[1].set_xlabel("step")
    ax[1].set_yscale("log")
    ax[1].legend()
    fig.tight_layout()
    path = "/home/workdir/artifacts/task2_bias_correction.png"
    fig.savefig(path, dpi=130)
    plt.close()
    print(f"wrote {path}")
    return first_below_1pct


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------
def lr_cosine(step, total, warmup, base):
    if step <= warmup:
        return base * step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return base * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def lr_wsd(step, total, warmup, base, stable_frac=0.8):
    stable_end = int(total * stable_frac)
    if step <= warmup:
        return base * step / max(warmup, 1)
    if step <= stable_end:
        return base
    decay_len = max(total - stable_end, 1)
    progress = (step - stable_end) / decay_len
    return base * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


# ---------------------------------------------------------------------------
# Task 3 — update-to-weight ratio + warmup
# ---------------------------------------------------------------------------
def layer_groups(model: MiniGPT):
    return {
        "tok_emb": [model.tok_emb.weight],
        "pos_emb": [model.pos_emb.weight],
        "attn": [p for n, p in model.named_parameters() if "attn" in n],
        "ffn": [p for n, p in model.named_parameters() if "ffn" in n],
        "rmsnorm": [p for n, p in model.named_parameters() if n.endswith("weight") and ("n1" in n or "n2" in n or "n_f" in n)],
    }


def task3(device):
    print("\n========== TASK 3: Update / weight ratio + warmup ==========")
    torch.manual_seed(0)
    vocab, T, B = 64, 32, 8
    warmup, total, base_lr = 30, 80, 3e-3
    model = MiniGPT(vocab=vocab, block_size=T, n_layer=2, n_head=4, n_embd=128).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, betas=(0.9, 0.999), weight_decay=0.01)
    rng = torch.Generator().manual_seed(1)
    groups = layer_groups(model)
    history = {k: [] for k in groups}
    history["all"] = []

    for step in range(1, total + 1):
        lr = lr_cosine(step, total, warmup, base_lr)
        for pg in opt.param_groups:
            pg["lr"] = lr
        x, y = make_batch(vocab, T, B, device, rng)
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()

        # snapshot weights, step, measure delta
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        opt.step()
        for name, params in groups.items():
            dw2 = 0.0
            w2 = 0.0
            for p in params:
                dw2 += (p.detach() - before[name_of(model, p)]).pow(2).sum().item() if False else 0.0
            # robust: match by identity
        for name, params in groups.items():
            dw2 = w2 = 0.0
            for p in params:
                old = None
                for n, q in model.named_parameters():
                    if q is p:
                        old = before[n]
                        break
                if old is None:
                    continue
                dw2 += (p.detach() - old).pow(2).sum().item()
                w2 += p.detach().pow(2).sum().item()
            ratio = math.sqrt(dw2) / (math.sqrt(w2) + 1e-12)
            history[name].append(ratio)
        dw2 = w2 = 0.0
        for n, p in model.named_parameters():
            dw2 += (p.detach() - before[n]).pow(2).sum().item()
            w2 += p.detach().pow(2).sum().item()
        history["all"].append(math.sqrt(dw2) / (math.sqrt(w2) + 1e-12))
        if step in (1, 10, 30, 31, 50, 80):
            print(f"step {step:3d}  lr={lr:.5f}  loss={loss.item():.4f}  ||Δw||/||w||={history['all'][-1]:.5e}")

    # warmup stops changing the *scale* at step=warmup
    peak_step = 1 + max(range(len(history["all"])), key=lambda i: history["all"][i])
    print(f"global ratio peaks at step {peak_step} (warmup ends at {warmup})")
    print("After warmup the ratio is no longer scaled up by rising η; it follows cosine decay.")

    fig, ax = plt.subplots(figsize=(8, 4))
    for name, series in history.items():
        ax.plot(range(1, total + 1), series, label=name)
    ax.axvline(warmup, color="k", ls="--", label=f"warmup end ({warmup})")
    ax.set_xlabel("step")
    ax.set_ylabel(r"$\Vert\Delta w\Vert_2 / \Vert w\Vert_2$")
    ax.set_title("Layer-wise update-to-weight ratio")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = "/home/workdir/artifacts/task3_update_ratio.png"
    fig.savefig(path, dpi=130)
    plt.close()
    print(f"wrote {path}")
    return warmup, peak_step


def name_of(model, p):
    for n, q in model.named_parameters():
        if q is p:
            return n
    return None


# ---------------------------------------------------------------------------
# Task 4 — cosine vs WSD, budget 300, stop at 200
# ---------------------------------------------------------------------------
def train_schedule(kind, device, total=300, stop=200, warmup=30, base_lr=3e-3, seed=0):
    torch.manual_seed(seed)
    vocab, T, B = 64, 32, 8
    model = MiniGPT(vocab=vocab, block_size=T, n_layer=2, n_head=4, n_embd=128).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, betas=(0.9, 0.999), weight_decay=0.01)
    rng = torch.Generator().manual_seed(seed + 11)
    losses, lrs = [], []
    sched = lr_cosine if kind == "cosine" else lr_wsd
    for step in range(1, stop + 1):
        lr = sched(step, total, warmup, base_lr)
        for pg in opt.param_groups:
            pg["lr"] = lr
        x, y = make_batch(vocab, T, B, device, rng, noise=0.35)
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        lrs.append(lr)
    return model, losses, lrs


def task4(device):
    print("\n========== TASK 4: Cosine vs WSD, stop at 200 / 300 budget ==========")
    m_cos, loss_c, lr_c = train_schedule("cosine", device)
    m_wsd, loss_w, lr_w = train_schedule("wsd", device)
    print(f"cosine  loss@200 = {loss_c[-1]:.4f}   lr@200 = {lr_c[-1]:.5f}")
    print(f"WSD     loss@200 = {loss_w[-1]:.4f}   lr@200 = {lr_w[-1]:.5f}")
    print("WSD still on the stable high-LR plateau at step 200 (decay starts at 240).")
    print("Cosine has already decayed → often lower instantaneous loss, less remaining mobility.")
    print("KEEP: WSD checkpoint. You can still decay from 200, or continue to 300, without a baked-in early anneal.")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(loss_c, label="cosine")
    ax[0].plot(loss_w, label="WSD")
    ax[0].set_title("Train loss (stop 200)")
    ax[0].legend()
    ax[1].plot(lr_c, label="cosine")
    ax[1].plot(lr_w, label="WSD")
    ax[1].axvline(240, color="k", ls=":", label="WSD decay start")
    ax[1].set_title("Learning rate (budget 300)")
    ax[1].legend()
    fig.tight_layout()
    path = "/home/workdir/artifacts/task4_cosine_wsd.png"
    fig.savefig(path, dpi=130)
    plt.close()
    print(f"wrote {path}")
    return loss_c[-1], loss_w[-1]


# ---------------------------------------------------------------------------
# Task 5 — LR sweep at widths 256, 512, 1024
# ---------------------------------------------------------------------------
def run_one(width, lr, device, steps=60, seed=0):
    torch.manual_seed(seed)
    vocab, T, B = 64, 32, 4
    n_head = 4 if width >= 256 else 2
    model = MiniGPT(vocab=vocab, block_size=T, n_layer=1, n_head=n_head, n_embd=width).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.0)
    rng = torch.Generator().manual_seed(seed + 3)
    warmup = 10
    last = None
    for step in range(1, steps + 1):
        eta = lr * min(1.0, step / warmup)
        for pg in opt.param_groups:
            pg["lr"] = eta
        x, y = make_batch(vocab, T, B, device, rng)
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last = loss.item()
    return last


def task5(device):
    print("\n========== TASK 5: LR sweep vs width ==========")
    widths = [256, 512, 1024]
    lrs = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    grid = {}
    minima = {}
    for d in widths:
        grid[d] = []
        print(f"-- width {d}  params≈{nparams(MiniGPT(n_embd=d, n_layer=1, n_head=4))}")
        for lr in lrs:
            val = run_one(d, lr, device)
            grid[d].append(val)
            print(f"   lr={lr:.0e}  loss={val:.4f}")
        best_i = min(range(len(lrs)), key=lambda i: grid[d][i])
        minima[d] = (lrs[best_i], grid[d][best_i])
        print(f"   min at lr={lrs[best_i]:.0e}  loss={grid[d][best_i]:.4f}")

    # SP guess: eta* ~ 1/sqrt(d)
    # fit log eta* = a - 0.5 log d using available minima
    import math as _m

    ds = list(minima.keys())
    etas = [minima[d][0] for d in ds]
    # predicted at 4096 from 1024 using 1/sqrt scaling
    eta_4096_from_1024 = minima[1024][0] * _m.sqrt(1024 / 4096)
    eta_4096_from_256 = minima[256][0] * _m.sqrt(256 / 4096)
    eta_4096_from_512 = minima[512][0] * _m.sqrt(512 / 4096)
    print(f"SP 1/sqrt(d) extrapolations to width 4096:")
    print(f"  from 256  → {eta_4096_from_256:.3e}")
    print(f"  from 512  → {eta_4096_from_512:.3e}")
    print(f"  from 1024 → {eta_4096_from_1024:.3e}")
    pick = eta_4096_from_1024
    print(f"USE at 4096: {pick:.2e}  (nearest of 3e-4 / 5e-4 / 1e-3 depending on min at 1024)")
    print("Confidence: medium-low. Only 3 widths, coarse 5-point grid, toy data, 1 layer, 60 steps.")
    print("Direction (smaller η at larger d under SP) is the reliable part, not the exact digit.")

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for d in widths:
        ax.semilogx(lrs, grid[d], "o-", label=f"d={d}")
        blr, bloss = minima[d]
        ax.scatter([blr], [bloss], s=80, zorder=5)
        ax.annotate(f"min {blr:.0e}", (blr, bloss), textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.set_xlabel("learning rate")
    ax.set_ylabel("loss after 60 steps")
    ax.set_title("LR sweep at three widths")
    ax.legend()
    fig.tight_layout()
    path = "/home/workdir/artifacts/task5_lr_width_sweep.png"
    fig.savefig(path, dpi=130)
    plt.close()
    print(f"wrote {path}")
    return grid, minima, pick


def main():
    device = torch.device("cpu")
    print("ERA V5 Assignment 11")
    print(f"device={device}")
    task1()
    t2 = task2()
    t3 = task3(device)
    t4 = task4(device)
    t5 = task5(device)
    print("\n========== DONE ==========")
    print(f"task2 first <1% step: {t2}")
    print(f"task3 warmup end / peak: {t3}")
    print(f"task4 losses cosine/WSD @200: {t4}")


if __name__ == "__main__":
    main()
