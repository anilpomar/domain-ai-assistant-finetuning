# Why full fine-tuning is expensive
Full Fine-Tuning needs training which updates weights on each parameters of a model.
    Ex : unsloth/tinyllama-bnb-4bit which has 1.1 billion parameters
     In a full fine-tuning run, every single one of the 1.1 billion parameters has its weights updated during training.

**Reasons.**    
**High VRAM Cost**-It requires significantly more hardware memory than PEFT (Parameter-Efficient Fine-Tuning)
**Optimizer State Overhead**-The optimizer must track gradients for all 1.1 billion parameters.
## What "learning a fact" means inside a model

Fine-tuning does not store facts in a lookup table. It adjusts weights so that, given some input tokens, the probability of the correct next token increases.

"Learning" that the drug is *17-alpha hydroxyprogesterone caproate* means: after training, when the model sees the tokens for "the drug is," the weights push it to assign high probability to the tokens spelling that name.

Each time a training example containing that fact passes through, the gradient nudges the weights slightly toward making the correct tokens more likely.

> **Memorization is the accumulation of many small gradient nudges toward the same target.**

Everything that follows is a consequence of this one idea.

---

## Values Used for Fine-Tuning

- LoRA rank: 16
- LoRA alpha: 32
- LoRA dropout: 0
- BATCH_SIZE = 2`
- STAGE1_LR = 5e-5`       # Non-Instruction Learning Rate
- STAGE2_LR = 2e-4`       # Instruction Learning Rate
- STAGE3_LR = 5e-5`       # Preference Learning Rate

# Fine-Tuning Explained

A conceptual walkthrough of the three-stage pipeline, why each stage exists, and what the experiments actually demonstrated.

---
## The Three Stages

### Stage 1 — Continued Pretraining (CPT)
**Objective:**  Trains Domain Knowledge to Pretrained Model.
**Data:** 25 packed chunks from the Antares–AMAG contract.
**Teaches:** domain *style* and vocabulary.
**Does not teach:** instruction-following.

CPT trains the model to *continue* text in a domain. Tested with a question, a CPT-only model will still ramble — this is correct behavior, not failure. Its value appears in the loss on contract text (2.02 → 0.88), not in Q&A.

### Stage 2 — Supervised Fine-Tuning (SFT)
**Objective:** Trains Model on top of Domain to respond to question asked.Given an instruction, produce the correct response.
**Data:** 104 Alpaca-format instruction/response pairs.
**Teaches:** instruction-following, plus recall of well-represented facts.

Three implementation details matter enormously:

**1. Prompt template.** Every example is formatted identically, and inference must use the exact same template — including the trailing newline after `### Response:`. A mismatch degrades output badly.

**2. EOS token.** The response must end with `tokenizer.eos_token`, or the model never learns where to stop and will ramble past its answer.

**3. Response-only masking.** Prompt tokens are set to `-100` in the labels so loss is computed *only* on the response. Without this, most gradient signal goes into predicting the instruction text — which the model will never need to generate.

```python
split = text.find("### Response:\n") + len("### Response:\n")
n_prompt = len(tokenizer(text[:split], add_special_tokens=False)["input_ids"])
for i in range(n_prompt):
    labels[i] = -100          # mask the prompt
```

**Verify it worked** by decoding the unmasked tokens:
```
'The Drug is 17-alpha hydroxyprogesterone caproate.</s>'   ← response only ✓
```

### Stage 3 — Direct Preference Optimization (DPO)
**Objective:** Model Gets trained to opt Positive answers on Negative. Given a prompt and two candidate answers, prefer the better one.
**Data:** 50 `(prompt, chosen, rejected)` triples.
**Teaches:** preference between things the model can already represent.

DPO **sharpens**; it does not **teach**. If the SFT model has no internal representation of a fact, showing it "prefer A over B" produces only a shallow preference between two strings it doesn't understand. This is why the stage order matters: SFT must teach the fact before DPO can refine it.

---

## The Central Empirical Finding: Signal Ratio

The drug name appeared in **2 of 100** original training examples.

With an effective batch of 16–32, that fact contributes ~2% of the gradient on any given step. Meanwhile the *general pattern* — "contract questions get answered in fluent legal prose" — appears in **all 100** examples.

The model learns the high-frequency thing (style) and barely moves on the low-frequency thing (the specific name).

### Why more steps could not fix it
More epochs amplify the entire gradient signal uniformly. **They do not change the ratio.** The fact stays at 2/100 whether it's step 5 or step 160.

This was tested directly. Going 60 → 160 steps produced a *different* invented drug name each run:

```
NBI-495  →  NBI-4157  →  NBI-49544  →  fentanyl citrate
```

Prose became more fluent. Facts stayed wrong.

### Why the chemical name is a hard target
`17-alpha hydroxyprogesterone caproate` tokenizes into ~7 rare sub-word pieces. **Every one must be correct in sequence.**

| Per-token confidence | Whole-name probability (7 tokens) |
|---|---|
| 0.73 | `0.73^7` ≈ **11%** |
| 0.998 | `0.998^7` ≈ **99%** |

That exponent explains a counterintuitive observation: a weakly-learned fact produces a *confident fabrication*, not a partially-correct answer. One weak token, and the model continues smoothly down a plausible wrong path.

### The fix: varied phrasings
Enriching the fact to **8 occurrences** across different question phrasings did two things:

1. **Raised the gradient ratio** (2/100 → 8/100).
2. **Created multiple independent gradient paths** to the same target tokens, so the model generalizes across phrasings rather than memorizing one rigid input→output pair.

This is why *varied phrasings* beat *duplicating one example eight times*.

**Result:** the enriched fact is now answered correctly. Facts appearing once (effective date) are still fabricated.

> **A fact's frequency in the training data, relative to everything else, determines whether the model learns it. Change the composition of the data, not the duration of training.**

---

## Decoding: Greedy vs Sampling

`do_sample` is an **inference** parameter with **zero effect on training**.

| | Behavior | Use for |
|---|---|---|
| `do_sample=False` (greedy) | Always takes the highest-probability token. **Deterministic.** | Factual questions |
| `do_sample=True` | Samples from the distribution | Creative variety |

**The tell:** three different invented drug codes for the *same* question across runs. A changing answer is the fingerprint of **sampling**, not of learning. Under greedy, the same model gives the same answer every time — that is how you know you are reading learned behavior rather than dice rolls.

Sampling also compounds the token-chain problem above: one wrong token early, and the whole answer derails.

---

## Reading the Metrics

### Loss can lie
On the T4, loss descended from 2.29 → 1.82 while the model learned **nothing**. That descent came from trivial parts of the objective — where to emit `</s>`, roughly how long a response should be. These require almost no weight movement.

### `grad_norm` cannot lie

| `grad_norm` | Meaning |
|---|---|
| `nan` / `inf` | Overflow; optimizer step skipped. **Nothing learns.** |
| ~0.001 or below | Vanishing gradients. **Effectively nothing learns.** |
| **0.1 – 2.0** | **Healthy.** |
| > 10, growing | Exploding; needs clipping. |

### `lora_B` is the direct proof
LoRA's `lora_B` matrix **initializes to exactly zero**. The merge computes `W + (B @ A) × (α/r)`. If `lora_B` is still ~0 after training, the merged model *is* the base model.

| Run | Hardware | Steps | `lora_B` max |
|---|---|---|---|
| SFT | T4 (fp16) | 60 | 0.00052 ← never left zero |
| CPT | L4 (bf16) | 100 | **0.0204** ✓ |

### DPO metrics are different
Loss starts at exactly **`0.6931 = ln 2`** (the model is indifferent) and descends. Never compare it to SFT loss.

- **`rewards/margins`** = how much the model prefers chosen over rejected. Should grow positive.
- **`rewards/accuracies`** = fraction of pairs where chosen scores higher. Should reach 1.0.
- **`rewards/chosen`** = the one to watch. DPO optimizes only the *margin*, so it can keep growing that margin while making chosen answers **less likely too**. When `rewards/chosen` goes negative, the model is learning to produce *neither* answer.

---

## Precision: Why the GPU Mattered

The single hardest bug in this project was numeric.

| Format | Exponent bits | Mantissa bits | Dynamic range |
|---|---|---|---|
| fp32 | 8 | 23 | ~1e-38 to 1e38 |
| **bf16** | **8** | 7 | **~1e-38 to 1e38** |
| fp16 | 5 | 10 | ~6e-5 to 65504 |

**Exponent bits determine range.** Transformer gradients span many orders of magnitude. In fp16 the large ones **overflow to `nan`**.

PyTorch's GradScaler responds by skipping the optimizer step and halving the loss scale — repeatedly. It eventually lands at a scale where gradients are finite but crushed to ~1e-3 magnitude. Training "runs," loss ticks down, **nothing learns**.

This is exactly what the logs showed:
```
step  1-25:  grad_norm = nan      ← every step SKIPPED
step 26-40:  grad_norm = 0.0014   ← survived, 1000× too small
```

**bf16 has fp32's full range in 16 bits**, sacrificing mantissa precision instead — which neural network training tolerates easily. No overflow, no GradScaler, no crushed gradients.

bf16 is a property of the **GPU**, not the model. The T4 (Turing, compute 7.5) predates bf16 tensor cores; the L4 (Ada, 8.9) has them. Note that `unsloth/tinyllama-**bnb-4bit**` refers to how *frozen base weights are stored*, which is entirely independent of what precision the *training computation* runs in.

---


## Summary of Lessons

1. **`grad_norm` is authoritative; the loss curve is not.**
2. **fp16's problem is range, not precision.** On pre-Ampere GPUs, fp16 + LoRA + 4-bit is a known NaN generator.
3. **"It worked yesterday" means the environment changed.** Pin dependency versions.
4. **Fact frequency, not training duration, drives factual recall.**
5. **Prefer deterministic mechanisms over pattern-matching ones** — offset-based masking over token-subsequence search; greedy decoding over sampling for evaluation.
6. **Distinguish "broken" from "not finished."** A near-zero `lora_B` after 15 steps looks identical to the fp16 bug — but the gradients were healthy and the run was simply too short.
7. **DPO sharpens preferences; it does not teach facts.**
8. **Every metric has a failure mode where it lies.** Loss lied (trivial-objective descent). Margins lied (grew while both rewards fell). Watch multiple signals, and know which is authoritative for the question you are asking.
