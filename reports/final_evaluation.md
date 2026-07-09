# Final Evaluation — DPO-Aligned Model

**Model:** Stage 3 DPO, trained on 50 preference pairs on top of the Stage 2 SFT merged model.
**Decoding:** greedy (`do_sample=False`).

---

> ### ⚠ Data Gap
> This report evaluates the DPO model on the **headline question only**. The full
> ten-question evaluation suite has not yet been run against Stage 3.
> Before submitting, run the same question list used for Base/CPT/SFT against the
> DPO model and fill in the "Full Question Suite" section below. Without it, the
> DPO stage cannot be fairly compared against SFT, and any claim about DPO
> improving or degrading general behavior is unsupported.

---

## Training Summary

| Setting | Value |
|---|---|
| Preference pairs | 50 |
| Steps | 15 (≈4 epochs) |
| Learning rate | 5e-5 |
| `beta` | 0.1 |
| Reference model | `None` (frozen base acts as implicit reference under LoRA) |

### Metrics

| Step | Loss | rewards/chosen | rewards/rejected | accuracies | margins |
|---|---|---|---|---|---|
| 1 | 0.6931 | 0.000 | 0.000 | 0.00 | 0.000 |
| 5 | 0.5795 | +0.120 | −0.130 | 1.00 | +0.250 |
| 10 | 0.0964 | +1.001 | −1.851 | 1.00 | +2.853 |
| 15 | 0.0181 | **+1.142** | −3.500 | 1.00 | **+4.642** |

Loss begins at exactly **0.6931 = ln 2**, the point at which the model is indifferent between `chosen` and `rejected`. It descends from there. Accuracy reaches 1.0 by step 5.

Critically, **`rewards/chosen` remains positive (+1.14) at the end.** The model became *more* likely to produce the correct answer, not merely less likely to produce the wrong one.

---

## Result on the Headline Question

**Q: What is the Drug that is the subject of the Antares–AMAG Manufacturing Agreement?**

> **A: It is 17-alpha hydroxyprogesterone caproate.** ✓

Correct, and consistent with the SFT model. DPO preserved the fact rather than degrading it.

---

## Over-Optimization: A Documented Failure

An initial DPO run at **60 steps** over-optimized badly. Comparing the two:

| Metric | DPO-60 (rejected) | DPO-15 (adopted) |
|---|---|---|
| Final loss | 0.000000 — saturated | 0.0181 |
| `rewards/chosen` | **−1.442** ← negative | **+1.142** |
| `rewards/rejected` | −15.216 | −3.500 |
| `rewards/margins` | +13.774 | +4.642 |
| `rewards/accuracies` | 1.000 | 1.000 |

### Why DPO-60 was rejected
DPO optimizes only the **margin** = `chosen − rejected`. Once `rejected` is suppressed far enough, the model can keep growing the margin while making **chosen answers less likely too**. At steps 40 and 60, `rewards/chosen` went negative — the model was learning to produce *neither* answer, merely rejecting one slightly harder.

Suppressing `rewards/rejected` to −15.2 also risks degrading general fluency, since the rejected answers are themselves well-formed English contract prose.

The loss hitting exactly `0.000000` by step 20 meant the objective was saturated; the remaining 40 steps were not merely wasted but actively harmful.

> **Stopping rule: watch `rewards/chosen`, not just margins. Margins can grow while both sides get worse. When `rewards/chosen` starts falling, stop.**

---

## Full Question Suite

*(To be completed — run the ten-question list against the DPO merged model.)*

| # | Question | SFT answer | DPO answer | Verdict |
|---|---|---|---|---|
| 1 | Drug name | 17-alpha hydroxyprogesterone caproate ✓ | 17-alpha hydroxyprogesterone caproate ✓ | Preserved |
| 2 | Parties | Antares and AMAG ✓ | *pending* | |
| 3 | Effective date | September 30, 2016 ✗ | *pending* | |
| 4 | Quality Agreement | fabricated ✗ | *pending* | |
| 5 | Minimum orders | partially correct | *pending* | |
| 6 | Return/recall policy | vague | *pending* | |
| 7 | Transfer Price | "$1" ✗ | *pending* | |
| 8 | Prefilled Syringe | degenerate repetition ✗ | *pending* | |

**The control case to watch:** question 3 (Effective Date). The preference dataset contains a pair for it (chosen: "March 20, 2018" / rejected: "September 30, 2014"), but the SFT model never learned the fact — it appeared in only **1 of 104** training examples.

- If DPO **fixes** it, preference signal alone was sufficient.
- If DPO **does not** fix it, this cleanly demonstrates that **DPO sharpens preferences but does not teach facts.**

The second outcome is the expected one, and is the more instructive result.

---

## Known Limitations

### 1. Length bias in the preference data
`chosen` answers average 202 characters; `rejected` average 149. DPO is known to latch onto length as a proxy for quality. The model may have learned "prefer longer" alongside "prefer correct."

### 2. Sparse coverage of the target fact
Only **1 of 50** preference pairs mentions the drug name. DPO is not heavily reinforcing it — it is one signal among fifty.

### 3. Inherited SFT overfitting
Stage 2 reached loss 0.0018 (near-total memorization). DPO builds on that model, so its confabulation of rare facts is inherited, not corrected.

### 4. Preference pairs test facts, not style
The pairs discriminate *correct fact vs. plausible wrong fact*, not *good style vs. bad style*. DPO can only sharpen a discrimination the underlying model is already capable of representing.

---

## Overall Verdict

| Criterion | Base | CPT | SFT | DPO |
|---|---|---|---|---|
| Answers the question | ✗ | ✗ | ✓ | ✓ |
| Contract vocabulary | ✗ | Partial | ✓ | ✓ |
| Stops cleanly | ✗ | ✗ | ✓ | ✓ |
| Enriched fact correct | ✗ | ✗ | ✓ | ✓ |
| Rare facts correct | ✗ | ✗ | ✗ | *pending* |

**The pipeline works end-to-end**, with each stage contributing a distinct, measurable capability. The DPO stage was validated by its reward dynamics (accuracy 1.0, margins +4.64, chosen reward positive) and by preserving the correct answer on the headline question.

The remaining limitation is factual coverage of rare facts, which is a **data problem, not a training problem**, and is not something DPO is designed to solve.
