# SFT Model Comparison

**Compared:** Base → Stage 1 (CPT) → Stage 2 (SFT)
**Decoding:** greedy (`do_sample=False`) for all models — identical settings, so differences are attributable to training alone.

---

## Training Summary

| Stage | Data | Steps | LR | Loss | `lora_B` max |
|---|---|---|---|---|---|
| 1 — CPT | 25 packed chunks (1 contract) | 100 | 2e-4 | 2.021 → 0.882 | 0.0204 |
| 2 — SFT | 104 instruction examples | 150 | 2e-4 | 1.839 → 0.0018 | ✓ |

Stage 1 gradient norms decayed healthily from 9.28 → ~0.4–0.6, confirming real optimization.

---

## Side-by-Side: The Headline Question

**Q: What is the Drug that is the subject of the Antares–AMAG Manufacturing Agreement?**

| Model | Answer |
|---|---|
| **Base** | "NBUD-BRL (Nebulin Autologous Beta-Lactamase Inhibitor)... a wholly owned subsidiary of Amgen Inc." |
| **CPT** | Contract-shaped text; still fabricates entities |
| **SFT** | **"It is 17-alpha hydroxyprogesterone caproate."** ✓ |

This is the clearest demonstration of what SFT contributes.

---

## What Each Stage Actually Added

### Stage 1 (CPT) — domain style, not instruction-following
CPT trains the model to *continue* domain text. It does not teach question-answering. Tested with a question, the CPT model still rambles and fabricates — which is the correct, expected behavior, not a failure.

Its value is measurable in the loss (2.02 → 0.88 on contract text) rather than in free-form Q&A. On 25 chunks from a single contract, its effect on generation is subtle.

### Stage 2 (SFT) — the transformation
Three changes are immediately visible:

**1. It answers the question.**
Base rambles for 150 tokens. SFT produces a direct, bounded answer and stops.

**2. It uses the contract's actual vocabulary.**
"Trainer Product," "Firm Order Acceptance," "Forecast," "Subcontractor," "sample Product" — these are terms from the agreement, not generic legal boilerplate.

**3. It gets enriched facts right.**
The drug name is correct, and it *carries into other answers*:

> "...the manufacture and specification of a Device Product — the **17-alpha hydroxyprogesterone caproate** Device..."

That is generalization, not row-level memorization of one training example.

---

## Honest Assessment: SFT's Remaining Failures

Stage 2 reached a final loss of **0.0018** — near-total memorization at 22 epochs. This produced a sharp split in behavior.

### Facts it knows (high training frequency)
| Fact | Occurrences | Result |
|---|---|---|
| Drug name | 8 (enriched) | ✓ Correct |
| Parties (Antares/AMAG) | 55/57 | ✓ Correct |
| Prefilled Syringe, Trainer | 16–20 | ✓ Reasonable |

### Facts it invents (low training frequency)
| Fact | Occurrences | SFT output |
|---|---|---|
| Effective Date | **1** | "September 30, 2016" ✗ (true: March 20, 2018) |
| Governing law | low | "International Business Commencement Code" ✗ (invented) |
| Company locations | 0 | "Antares in Miami, AMAG in **Liverpool, England**" ✗ |
| Transfer Price | 5 | "a dollar-denominated percentage, namely **$1**" ✗ |

One answer degenerated into repetition:
> "It produces and produces only a Finished Product..."

---

## The Central Finding

**Fact frequency in the training data — not training duration — determines factual recall.**

The one fact deliberately enriched to 8 varied phrasings is the one fact the model reliably reproduces. Facts appearing once are confabulated *fluently and confidently*, which is more dangerous than obvious nonsense because it looks correct.

Earlier attempts to fix this by training longer (60 → 160 steps) failed entirely: the model produced a different invented drug name each time (NBI-495 → NBI-4157 → NBI-49544). More epochs amplify the whole gradient signal uniformly; they cannot change the *ratio* of a rare fact against everything else.

### Why the chemical name is a hard case
`17-alpha hydroxyprogesterone caproate` tokenizes into ~7 rare sub-word pieces, all of which must be correct in sequence:

| Per-token confidence | Whole-name probability |
|---|---|
| 0.73 | `0.73^7` ≈ **11%** |
| 0.998 | `0.998^7` ≈ **99%** |

That exponent explains why a weakly-learned fact yields a confident fabrication rather than a partially-correct answer: one weak token, and the model continues smoothly down a plausible wrong path.

---

## Verdict

| Criterion | Base | CPT | SFT |
|---|---|---|---|
| Answers the question asked | ✗ | ✗ | ✓ |
| Contract-specific vocabulary | ✗ | Partial | ✓ |
| Stops cleanly | ✗ | ✗ | ✓ |
| Enriched facts correct | ✗ | ✗ | ✓ |
| Rare facts correct | ✗ | ✗ | ✗ |

SFT delivers instruction-following and reliable recall of well-represented facts. It does not solve rare-fact hallucination — that requires more data coverage, not more training.
