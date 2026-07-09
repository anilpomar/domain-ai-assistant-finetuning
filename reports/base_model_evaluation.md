# Base Model Evaluation

**Model:** `unsloth/tinyllama-bnb-4bit` (1.1B params, 4-bit quantized)
**Decoding:** greedy (`do_sample=False`), deterministic
**Purpose:** Establish a baseline before any fine-tuning.

---

## Summary

The base model has **no knowledge of the Antares–AMAG Manufacturing Agreement** and **does not follow instructions**. It responds by continuing text in whatever pattern the prompt resembles, producing fluent, confident, and entirely fabricated content.

This is expected. TinyLlama is a raw pretrained model — it was trained to predict the next token, not to answer questions. Both capabilities are what the subsequent stages add.

---

## Observed Failure Modes

### 1. Confident factual fabrication
Asked to name the drug, it invented a compound, a mechanism, and a citation:

> The drug in question, **NBUD-BRL** (also known as **Nebulin Autologous Beta-Lactamase Inhibitor**). The product was developed by AMAG ... under an agreement with Antares Pharma Inc., a wholly owned subsidiary of Amgen Inc.[1]
> [1] https://www.fda.gov/drugs/guidances/ucm350264.htm

Every element is false: the compound does not exist, Antares is not an Amgen subsidiary, and the FDA URL is invented. The fabricated citation is notable — the model has learned that authoritative text *contains* citations, so it produces one.

### 2. Wrong domain entirely
Asked who the parties are, it named **Amazon**:

> The Parties to this agreement include: **Amazon** (the "Customer") and ______________________________ ("Seller")

Including a literal fill-in-the-blank underline, indicating it is reproducing a generic contract *template* from pretraining rather than reasoning about a specific agreement.

### 3. No instruction-following
Several responses drift into generating *new questions* rather than answering:

> **Question #4:** What are some examples from your experience where the Quality Agreement needs to be reviewed?

The model treats the prompt as a document to continue, not a question to answer.

### 4. Format leakage from pretraining
One response terminated with a stray LaTeX directive:

> \end{document}

Another emitted a Google Drive link. These are artifacts of the pretraining corpus surfacing verbatim.

### 5. Invented dates
- "The effective date will be on or about **January 1, 20XX**"
- "effective date is **10/29/2018**"

Both fabricated (the true date is March 20, 2018). Note the `MM/DD/YY` format — a pretraining prior for legal text, not anything learned from this contract.

---

## Interpretation

None of this indicates a broken model. It indicates a model doing exactly what a base LLM does: producing statistically plausible continuations. The failures cluster into two categories, each addressed by a later stage:

| Deficit | Addressed by |
|---|---|
| No domain knowledge | Stage 1 (CPT) + Stage 2 (SFT) |
| No instruction-following | Stage 2 (SFT) |
| No preference for correct over plausible | Stage 3 (DPO) |

---

## Baseline Verdict

| Criterion | Result |
|---|---|
| Answers the question asked | ✗ |
| Uses contract-specific vocabulary | ✗ |
| Factually correct | ✗ |
| Stops cleanly | Partially (some responses run to token limit) |
| Fluent English | ✓ |

**The only capability the base model has is fluency.** Everything else must be trained in. This makes it an excellent baseline — any improvement in the later stages is unambiguous.
