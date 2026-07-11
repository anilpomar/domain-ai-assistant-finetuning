# Why full fine-tuning is expensive
Full Fine-Tuning needs training which updates weights on each parameters of a model.
    Ex : unsloth/tinyllama-bnb-4bit which has 1.1 billion parameters
     In a full fine-tuning run, every single one of the 1.1 billion parameters has its weights updated during training.

**Reasons.**    
**High VRAM Cost**-It requires significantly more hardware memory than PEFT (Parameter-Efficient Fine-Tuning)
**Optimizer State Overhead**-The optimizer must track gradients for all 1.1 billion parameters.
## What Lora Does
Its an adapter that freezes actual weights of model and trains a tiny patch that gets added to Model.(Matrix Decomposition is used).
Ex:
Instead of updating all 1.1 billion weights during fine-tuning, LoRA freezes them and trains a tiny "patch" (~12.6 million weights, ~1%) that gets added on top.

---
## What QLora Does
QLoRA(Quantized Low-Rank Adaptation) is LoRA, but the frozen base weights are compressed to 4-bit instead of stored in 16-bit — so the model fits in 4× less GPU memory while training the tiny patch.

---

## Why QLoRA is useful on limited GPU
Normally, a model's weights are stored in 16-bit (FP16/BF16) or 32-bit precision.

QLoRA quantizes the pretrained model to 4-bit which makes GPU 

# Fine-Tuning Explained

A conceptual walkthrough of the three-stage pipeline, why each stage exists, and what the experiments actually demonstrated.

---
## The Three Stages

### Stage 1 — Continued Pretraining (CPT)
**Objective:**  Trains Domain Knowledge to Pretrained Model.
**Teaches:** domain *style* and vocabulary.
**Does not teach:** instruction-following.

CPT trains the model to *continue* text in a domain. Tested with a question, a CPT-only model will still ramble — this is correct behavior, not failure. Its value appears in the loss on contract text (2.02 → 0.88), not in Q&A.

### Stage 2 — Supervised Fine-Tuning (SFT)
**Objective:** Trains Model on top of Domain to respond to question asked.Given an instruction, produce the correct response.
**Data:** 104 Alpaca-format instruction/response pairs.
**Teaches:** instruction-following, plus recall of well-represented facts.

### Stage 3 — Direct Preference Optimization (DPO)
**Objective:** Model Gets trained to opt Positive answers on Negative. Given a prompt and two candidate answers, prefer the better one.
**Data:** 50 `(prompt, chosen, rejected)` triples.
**Teaches:** preference between things the model can already represent.

DPO **sharpens**; it does not **teach**. If the SFT model has no internal representation of a fact, showing it "prefer A over B" produces only a shallow preference between two strings it doesn't understand. This is why the stage order matters: SFT must teach the fact before DPO can refine it.

---
## Difference between SFT and DPO
SFT teaches a model to imitate good responses, while DPO teaches it to prefer better responses over worse ones.
## Values Used for Fine-Tuning

- LoRA rank: 16
- LoRA alpha: 32
- LoRA dropout: 0
- BATCH_SIZE = 2`
- STAGE1_LR = 5e-5`       # Non-Instruction Learning Rate
- STAGE2_LR = 2e-4`       # Instruction Learning Rate
- STAGE3_LR = 5e-5`       # Preference Learning Rate
