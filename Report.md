# ERA V5 — Assignment 11

**Adam Dynamics, Learning-Rate Schedules, and Width Scaling in a MiniGPT Transformer**

| Field | Value |
|---|---|
| Course | ERA V5 |
| Assignment | 11 |
| Runtime | CPU, PyTorch 2.12 |
| Repro | `python3 era_v5_assignment11.py` |

This repository is a complete, self-contained solution. One script runs every required experiment, writes four figures, and prints the numerical claims cited below. No external dataset is required.

---

## 1. Scope

The brief asks for five empirical checks on the optimizer that actually moves LLM weights:

1. Closed-form Adam versus `torch.optim.Adam` on a single scalar.
2. The horizon over which bias correction still changes the step.
3. Layer-wise \(\lVert\Delta w\rVert_2/\lVert w\rVert_2\) through linear warmup.
4. Cosine versus warmup–stable–decay (WSD) when both runs are stopped early.
5. A learning-rate sweep at three widths, with an extrapolation to \(d=4096\).

The model is a decoder-only MiniGPT: pre-LN RMSNorm, causal multi-head attention, SwiGLU FFN, and tied input/output embeddings. That is the smallest architecture that still contains the same parameter groups as a Llama-style block.

---

## 2. How to run

```bash
python3 era_v5_assignment11.py
```

Outputs written next to the script:

| File | Content |
|---|---|
| `task2_bias_correction.png` | Corrected vs raw Adam updates; relative gap vs step |
| `task3_update_ratio.png` | Per-layer update-to-weight ratio |
| `task4_cosine_wsd.png` | Train loss and LR under cosine and WSD |
| `task5_lr_width_sweep.png` | Loss against \(\eta\) at \(d\in\{256,512,1024\}\) |

Expected wall time on CPU: one to two minutes. Task 5 dominates.

---

## 3. Results

### 3.1 Adam by hand

Scalar \(w_0=0.5\), gradients \([0.4,-0.2,0.1,0.05,-0.3]\), \(\eta=10^{-3}\), \(\beta=(0.9,0.999)\), \(\varepsilon=10^{-8}\), float64.

| \(t\) | \(m_t\) | \(v_t\) | \(\hat m_t\) | \(\hat v_t\) | \(w^{\text{hand}}\) | \(w^{\text{Adam}}\) |
|---|---|---|---|---|---|---|
| 1 | 0.04000000 | 1.60000000e-04 | 0.40000000 | 1.60000000e-01 | 0.49900000 | 0.49900000 |
| 2 | 0.01600000 | 1.99840000e-04 | 0.08421053 | 9.99699850e-02 | 0.49873366 | 0.49873366 |
| 3 | 0.02440000 | 2.09640160e-04 | 0.09003690 | 6.99499800e-02 | 0.49839323 | 0.49839323 |
| 4 | 0.02696000 | 2.11930520e-04 | 0.07839488 | 5.30621702e-02 | 0.49805291 | 0.49805291 |
| 5 | −0.00573600 | 3.01718589e-04 | −0.01400698 | 6.04645260e-02 | 0.49810987 | 0.49810987 |

Maximum absolute disagreement: **0**. The two implementations are the same recurrence.

### 3.2 Bias correction

On a constant gradient the corrected step is exactly \(\eta\,\mathrm{sign}(g)\). The uncorrected step is larger, because early \(v_t=(1-\beta_2)g^2\) underestimates second moment. The ratio of the two updates is

\[
\frac{\Delta w_{\text{corr}}}{\Delta w_{\text{raw}}}=\frac{\sqrt{1-\beta_2^t}}{1-\beta_1^t}.
\]

Measured relative gap \(\lvert\Delta w_{\text{corr}}-\Delta w_{\text{raw}}\rvert/\lvert\Delta w_{\text{corr}}\rvert\):

| step | gap |
|---|---|
| 20 | 524% |
| 100 | 224% |
| 300 | 96% |
| 400 | 74% |

With \(\beta_2=0.999\), a 1% gap requires \(\sqrt{1-\beta_2^t}\approx 0.99\), i.e. **\(t\approx 3915\)**. Bias correction still matters at step 20 and at step 300. It should stay enabled for any run shorter than a few thousand steps.

### 3.3 Update-to-weight ratio

Linear warmup of 30 steps, then cosine. Global \(\lVert\Delta w\rVert_2/\lVert w\rVert_2\):

| step | \(\eta_t\) | ratio |
|---|---|---|
| 1 | \(1.0\times 10^{-4}\) | \(6.17\times 10^{-4}\) |
| 30 | \(3.0\times 10^{-3}\) | \(5.03\times 10^{-3}\) |
| 31 | \(3.0\times 10^{-3}\) | \(4.73\times 10^{-3}\) |
| 80 | \(\approx 0\) | \(0\) |

Warmup stops increasing \(\eta\) at **step 30**. After that the ratio tracks the decaying schedule. The curve can peak slightly earlier (here step 18) because gradient norms also fall as loss falls; that is gradient scale, not leftover warmup.

### 3.4 Cosine versus WSD, early stop

Budget 300 steps, both runs halted at 200. WSD holds peak LR until step 240.

| schedule | loss @ 200 | LR @ 200 |
|---|---|---|
| Cosine | 3.342 | \(9.1\times 10^{-4}\) |
| WSD | 3.292 | \(3.0\times 10^{-3}\) |

**Checkpoint kept: WSD.** Instantaneous loss is not the selection criterion at an early stop. Cosine has already spent its anneal. WSD can still decay from a live LR, or continue to the original budget, without a schedule that assumed the run was almost finished.

### 3.5 Width–LR sweep and \(d=4096\)

Sixty steps, one layer, standard PyTorch parameterization. Loss after the fixed budget:

| \(d\) | \(10^{-4}\) | \(3\cdot10^{-4}\) | \(10^{-3}\) | \(3\cdot10^{-3}\) | \(10^{-2}\) | \(\eta^\star\) |
|---|---|---|---|---|---|---|
| 256 | 148.23 | 5.32 | **0.0006** | 0.176 | 4.18 | \(10^{-3}\) |
| 512 | 100.09 | **0.038** | 0.224 | 11.24 | 24.03 | \(3\cdot10^{-4}\) |
| 1024 | 3.49 | **0.060** | 40.37 | 63.60 | 66.14 | \(3\cdot10^{-4}\) |

Minima shift left as width grows, consistent with \(\eta^\star(d)\propto d^{-1/2}\) under SP.

Extrapolation to \(d=4096\) from the measured minima:

| anchor | implied \(\eta^\star(4096)\) |
|---|---|
| 256 | \(2.5\times 10^{-4}\) |
| 512 | \(1.1\times 10^{-4}\) |
| 1024 | \(1.5\times 10^{-4}\) |

**Recommended \(\eta\) at width 4096: \(1.5\times 10^{-4}\).** Neighbour to try: \(3\times 10^{-4}\).

Confidence is **low–medium**. The grid is five points, the model is one layer, the token stream is synthetic, and the 1024 minimum is not resolved between \(10^{-4}\) and \(3\cdot10^{-4}\). The directional claim is reliable: do not reuse \(\eta=10^{-3}\) from width 256 at width 4096. Under SP the step should drop by about \(1/4\).

---

## 4. Design notes

- Token streams are generated in-process. Tasks 1–3 and 5 use a deterministic modular count so the optimizer signal is visible in tens of steps. Task 4 injects 35% token noise so next-token loss cannot collapse to machine zero before step 200.
- Weight tying is applied (`lm_head.weight is tok_emb.weight`). Parameter counts treat the shared tensor once.
- Adam comparisons use float64. Mixing float32 into the hand recurrence is enough to break seventh-decimal agreement.
- Task 5 is intentionally short. It is a location-of-minimum experiment, not a pretrain.

---

## 5. File list

```
era_v5_assignment11.py      executable lab
README.md                   this document
task2_bias_correction.png
task3_update_ratio.png
task4_cosine_wsd.png
task5_lr_width_sweep.png
```
