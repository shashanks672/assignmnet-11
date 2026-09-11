# Assignment 11: Deep Dive into Adam Dynamics, Schedules, and Width Scaling

**Course:** ERA V5  
**Code:** `era_v5_assignment11.py`  
**Device:** CPU, PyTorch 2.12  

Plots: `task2_bias_correction.png`, `task3_update_ratio.png`, `task4_cosine_wsd.png`, `task5_lr_width_sweep.png`

---

## Model (`MiniGPT`)

Decoder-only Mini-Transformer:

- Pre-LN **RMSNorm**
- **Causal multi-head attention** (strict causal mask)
- **SwiGLU** FFN (gate × up, SiLU)
- **Tied** `lm_head.weight = tok_emb.weight`

Default for Tasks 3–4: `n_layer=2`, `n_embd=128`, `n_head=4`, `block=32`, `vocab=64`.  
Task 5 uses `n_layer=1` and widths `{256, 512, 1024}`.

Data is synthetic next-token sequences (counting stream; Task 4 adds 35% token noise so loss does not collapse to 0).

---

## Task 1 — Reproduce Adam by hand

One scalar weight \(w_0 = 0.5\), five gradients

\[
g = [0.4,\ -0.2,\ 0.1,\ 0.05,\ -0.3]
\]

with \(\eta=10^{-3}\), \(\beta_1=0.9\), \(\beta_2=0.999\), \(\varepsilon=10^{-8}\).

\[
m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t,\quad
v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^2
\]

\[
\hat m_t=\frac{m_t}{1-\beta_1^t},\quad
\hat v_t=\frac{v_t}{1-\beta_2^t},\quad
w_t=w_{t-1}-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\varepsilon}
\]

| t | g | m | v | m̂ | v̂ | w_hand | w_PyTorch |
|---|---|---|---|---|---|---|---|
| 1 | 0.400 | 0.04000000 | 1.60000000e-04 | 0.40000000 | 1.60000000e-01 | 0.49900000 | 0.49900000 |
| 2 | −0.200 | 0.01600000 | 1.99840000e-04 | 0.08421053 | 9.99699850e-02 | 0.49873366 | 0.49873366 |
| 3 | 0.100 | 0.02440000 | 2.09640160e-04 | 0.09003690 | 6.99499800e-02 | 0.49839323 | 0.49839323 |
| 4 | 0.050 | 0.02696000 | 2.11930520e-04 | 0.07839488 | 5.30621702e-02 | 0.49805291 | 0.49805291 |
| 5 | −0.300 | −0.00573600 | 3.01718589e-04 | −0.01400698 | 6.04645260e-02 | 0.49810987 | 0.49810987 |

**max |w_hand − w_Adam| = 0** (float64). Agrees to all printed decimals, well past \(10^{-7}\).

---

## Task 2 — Disable bias correction

Compared

- corrected: \(\Delta w=\eta\,\hat m/(\sqrt{\hat v}+\varepsilon)\)
- raw: \(\Delta w=\eta\,m/(\sqrt{v}+\varepsilon)\)

on a **constant** gradient so the only difference is the bias-correction factor

\[
\frac{\Delta w_{\text{corr}}}{\Delta w_{\text{raw}}}=\frac{\sqrt{1-\beta_2^t}}{1-\beta_1^t}.
\]

First 20 steps (constant \(g=0.25\), \(\eta=10^{-3}\)):

- Corrected update is **exactly \(\eta=10^{-3}\)** every step (because \(\hat m/\sqrt{\hat v}=\mathrm{sign}(g)\)).
- Raw update is **3.2× to 5.6× larger** (step 1: \(3.16\times10^{-3}\); step 12 peak ~\(6.57\times10^{-3}\)).

**Correction does not damp the first step — it shrinks it.** Early \(v\) is tiny (\((1-\beta_2)g^2\)), so \(m/\sqrt{v}\) explodes. Dividing \(v\) by \(1-\beta_2^t\approx0.001\) inflates \(\hat v\) and makes the first steps safe.

When it stops mattering (\(\beta_2=0.999\)):

| step | relative \(\lvert\Delta w_{\text{corr}}-\Delta w_{\text{raw}}\rvert/\lvert\Delta w_{\text{corr}}\rvert\) |
|---|---|
| 20 | 524% |
| 100 | 224% |
| 200 | 135% |
| 300 | 96% |
| 400 | 74% |

\(\beta_1^t\) is already ~0 by step ~50. \(\sqrt{1-\beta_2^t}\) only reaches 0.99 around **step 3915**. So the 1% horizon is **thousands of steps**, not 20.

**Report:** difference still matters at step 20 and even at 300. It “stops mattering” (~1% on update size) near **step 3900** for \(\beta_2=0.999\). Always leave bias correction on for short runs.

Artifact: `task2_bias_correction.png`

---

## Task 3 — Layer-wise \(\lVert\Delta w\rVert_2/\lVert w\rVert_2\) and warmup

Linear warmup of 30 steps, cosine after, base \(\eta=3\times10^{-3}\).

| step | lr | \(\lVert\Delta w\rVert/\lVert w\rVert\) (all params) |
|---|---|---|
| 1 | 1.0e-4 | 6.17e-4 |
| 10 | 1.0e-3 | 4.26e-3 |
| **30** (warmup end) | 3.0e-3 | 5.03e-3 |
| 31 | 3.0e-3 | 4.73e-3 |
| 50 | 1.96e-3 | 1.94e-3 |
| 80 | ~0 | 0 |

Warmup **stops raising the scale at step 30**. After that the ratio falls with cosine LR (and with shrinking grads).

The *peak* of the curve was step 18, not 30, because loss (hence gradient size) is also dropping during warmup. The quantity warmup *controls* is \(\eta_t\); that control ends at **step 30**.

Logged groups: `tok_emb`, `pos_emb`, `attn`, `ffn`, `rmsnorm`.

Artifact: `task3_update_ratio.png`

---

## Task 4 — Cosine vs WSD, budget 300, stop at 200

Same MiniGPT, same seed, same noisy data. Warmup 30. WSD stays at base LR until step \(0.8\times300=240\), then cosine-decays.

| schedule | loss @ 200 | lr @ 200 |
|---|---|---|
| Cosine | **3.342** | 9.1e-4 |
| WSD | **3.292** | 3.0e-3 |

On this toy the two losses are close (WSD slightly lower). Cosine has already annealed; WSD is still on the high-LR plateau.

**Keep the WSD checkpoint.** At an early stop you want remaining mobility: you can start decay *now* from a healthy LR, or continue to 300, without a schedule that already spent its anneal. Cosine at 200 is a model that thinks training is almost over.

Artifact: `task4_cosine_wsd.png`

---

## Task 5 — LR sweep at widths 256, 512, 1024

Sweep \(\eta\in\{10^{-4},3\cdot10^{-4},10^{-3},3\cdot10^{-3},10^{-2}\}\), 60 steps, 1 layer, SP (PyTorch default init). Loss after 60 steps:

| width | 1e-4 | 3e-4 | 1e-3 | 3e-3 | 1e-2 | **minimum** |
|---|---|---|---|---|---|---|
| 256 | 148.23 | 5.32 | **0.0006** | 0.176 | 4.18 | \(\eta^*=10^{-3}\) |
| 512 | 100.09 | **0.038** | 0.224 | 11.24 | 24.03 | \(\eta^*=3\cdot10^{-4}\) |
| 1024 | 3.49 | **0.060** | 40.37 | 63.60 | 66.14 | \(\eta^*=3\cdot10^{-4}\) |

Minima move **left** as width grows: larger \(d\) wants a smaller LR under SP.

### Width 4096

Standard parameterization: \(\eta^*(d)\propto 1/\sqrt{d}\).

| anchor | \(\eta^*(4096)=\eta^*(d)\sqrt{d/4096}\) |
|---|---|
| 256 @ 1e-3 | 2.5e-4 |
| 512 @ 3e-4 | 1.1e-4 |
| 1024 @ 3e-4 | **1.5e-4** |

**Value I would use at 4096: \(1.5\times10^{-4}\)** (from the widest measured min). I would also try \(3\times10^{-4}\) as a neighbour on the coarse grid.

**Confidence: low–medium.** Reasons to distrust the exact digit:

- only three widths, five LR points
- 1 layer, 60 steps, synthetic tokens, CPU
- grid cannot resolve a min between 1e-4 and 3e-4 at width 1024

What I *am* confident about: under SP you should **not** reuse \(\eta=10^{-3}\) from width 256 at 4096; go down by roughly \(1/\sqrt{16}=1/4\), into the mid \(10^{-4}\) range. µP / width-independent LR would be a different story; this sweep is SP.

Artifact: `task5_lr_width_sweep.png`

---

## Files

| file | what |
|---|---|
| `era_v5_assignment11.py` | all five tasks |
| `task2_bias_correction.png` | 20-step updates + long-horizon relative gap |
| `task3_update_ratio.png` | per-layer \(\lVert\Delta w\rVert/\lVert w\rVert\) |
| `task4_cosine_wsd.png` | loss and LR, cosine vs WSD |
| `task5_lr_width_sweep.png` | loss vs LR, three widths, minima marked |

```bash
python3 era_v5_assignment11.py
```
