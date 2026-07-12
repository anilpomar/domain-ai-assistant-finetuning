# Final Evaluation — DPO-Aligned Model

**Model:** Stage 3 DPO, trained on 50 preference pairs on top of the Stage 2 SFT merged model.
**Decoding:** greedy (`do_sample=False`).

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

## Full Question Suite — Base vs SFT vs DPO

| # | Question | Base Model Answer | SFT Answer | DPO Answer | Best | Verdict |
|---|---|---|---|---|---|---|
| 1 | What is the Drug that is the subject of the Antares–AMAG Manufacturing Agreement? | "NBUD-BRL (Nebulin Autologous Beta-Lactamase Inhibitor)... a wholly owned subsidiary of Amgen Inc." Fabricated FDA URL. | "It is 17-alpha hydroxyprogesterone caproate." ✓ | "It is 17-alpha hydroxyprogesterone caproate." ✓ | SFT / DPO (tied) | Both correct. The only fact enriched to 8 training occurrences. DPO preserved it. |
| 2 | Who are the parties to the Manufacturing Agreement? | "The Parties... include Amazon (the 'Customer') and ______ ('Seller')" | "It is the agreement between AMAG and Antares governing... the 17-alpha hydroxyprogesterone caproate Device" ✓ | "...specifies where they happen (and which one is in Sequence 01)... either party must execute as exhibit to it without revision." | SFT | Base names Amazon. SFT is correct and concise. DPO keeps the right parties but adds invented detail ("Sequence 01") and becomes rambling. |
| 3 | What is the effective date of the Agreement? | "on or about January 1, 20XX" | "It becomes effective on the First Firm Order Acceptance, and anticlaudation laws apply exclusively." ✗ | "It becomes effective on the First Forecast Date... both dates are expressed as months." ✗ | Neither | All three wrong (true: March 20, 2018). **CONTROL CASE:** the preference dataset contains a pair for this exact fact, yet DPO did not install it. The fact appeared in 1 of 104 SFT examples — there was no representation to sharpen. |
| 4 | Who are the agreeing parties? | Generic boilerplate about "the parties," no names given. | "Antares and AMAG, each formed under the International Business Commencement Code... Antares is based in Miami, Florida, and AMAG in Liverpool, England." ✗ | "...exclusively addressed under Title 36, Law 108." ✗ | Neither | SFT invents a legal code and two locations. DPO invents a different statute. Both fabricate confidently. |
| 5 | What is the Quality Agreement and when was it originally entered into? | "Quality Agreements were initially signed in 2015... QA Manager position added January 3rd, 2019" | "It is the fourth-year Mutual AMAG Quality Agenda, dated 15 October 2018." ✗ | "It is the Quranic Texts, Product, Trainer, Device or Sample Devices Quality AMAG Manufacturing Services..." ✗ | Base (least wrong) | DPO produces "Quranic Texts" — a religious-text token intruding into a pharma contract. Clear evidence that suppressing rejected answers damaged general language ability. |
| 6 | Are there Minimum Orders AMAG should place from Antares in each shipment? What is the payment policy? | "minimum order quantity is 100 units... contact our sales team at [email]" | "Antares must purchase its entire quantity of Product, sample Product and Trainer Product... after Forecasting it." | "Antares or its Subcontractor must purchase and pay for the entire quantity it delivers within the Forecast — 16-month Product, sample Product and Trainer Product." | SFT | Both SFT and DPO reverse the parties. DPO additionally emits "16-month Product," which is meaningless. |
| 7 | What is the Return and Recall Policy? | A long e-commerce returns policy: RMA forms, wholesale pricing, late fees. | "It addresses returned or exhibited samples, obsolescence and Replacement parts." | "It governs Antares or AMAG parties returning or removing sold or finished Products... including replacing them within a specified time after notice." | DPO (marginally) | DPO answer has more relevant content, though both are vague. Illustrates the length bias — chosen answers in the preference data averaged 202 chars vs 149 for rejected. |
| 8 | State the Effective Date of the Manufacturing Agreement. | "effective date is 10/29/2018", then drifts into generating new exam questions. | "It becomes effective on the First Final Forecast, expressed as September 30, 2016." ✗ | "It becomes effective on the First Acceptance, titled Antares Manufactures and AMAG Specifies Details" ✗ | Neither | DPO is worse than SFT — it no longer even attempts a date. SFT invents a different wrong date than in Q3: a fact with 1 occurrence is not stored, it is re-guessed on every generation. |
| 9 | What is the "Transfer Price" and where is it set out? | "set by the Manufacturer... in a separate agreement," then unrelated confidentiality clauses. | "...expressed as a dollar-denominated percentage, namely $1." ✗ | "...an agreement numbered Raquel after its date (with Ra being the year) and transferred to itself and AMAG exclusively sealed." ✗ | Neither | DPO produces incoherent word salad ("numbered Raquel"). A marked degradation from SFT's merely-wrong answer. |
| 10 | Who incorporates the Prefilled Syringe into the Device to produce the finished Product? | "manufactured by a third party... designed in accordance with cGMPs", then emits a stray LaTeX \end{document}. | "It produces and produces only a Finished Product that is completely constructed as a Device..." ✗ | "...including the Prefilled Shrinkage and Prefilled Packaging components, produced as a Device." ✗ | Neither | SFT degenerates into repetition ("produces and produces"). DPO inherits that AND corrupts "Prefilled Syringe" into "Prefilled Shrinkage." DPO amplified the SFT overfitting rather than correcting it. |

---

## Scorecard

| Metric | Base | SFT | DPO |
|---|---|---|---|
| Answers the question asked | 0/10 | 10/10 | 10/10 |
| Stops cleanly (emits EOS) | 3/10 | 10/10 | 10/10 |
| Uses contract vocabulary | 0/10 | 10/10 | 10/10 |
| **Factually correct** | **0/10** | **2/10** | **1/10** |
| **Coherent (no word salad)** | **8/10** | **9/10** | **5/10** |

DPO made the model worse on both axes that matter — factual accuracy dropped from 2/10 to 1/10, and coherence dropped from 9/10 to 5/10. It retained only the one enriched fact (drug name) and degraded nearly everything else.

---

## Why DPO Degraded the Model

### 1. Fluency degradation from suppressing rejected answers
`rewards/rejected` was driven to −3.50. The rejected answers are themselves fluent, well-formed English contract prose. Suppressing them damages the model's general language ability — which is why DPO produced "Quranic Texts" (Q5), "numbered Raquel" (Q9), and "Prefilled Shrinkage" (Q10).

### 2. Length bias in the preference data
Chosen answers averaged 202 characters against 149 for rejected. Every DPO answer is systematically longer and more rambling than its SFT counterpart (compare Q7). The model learned "prefer longer" alongside "prefer correct" — a well-documented DPO failure mode, predicted from the data before training.

### 3. Inherited SFT overfitting
Stage 2 reached loss 0.0018 (near-total memorization at 22 epochs). DPO built on that overfit model and amplified the degradation rather than correcting it. "anticlaudation" (SFT, Q3) and "Prefilled Shrinkage" (DPO, Q10) are symptoms of the same root cause — the SFT model learned to emit contract-shaped token sequences without semantic grounding.

### 4. DPO cannot teach facts the SFT model never learned
The preference dataset contains an explicit pair for the Effective Date — `chosen: "March 20, 2018"` vs `rejected: "September 30, 2014"`. DPO trained on it and reached 100% preference accuracy. The model still cannot produce the date.

**Preferring the right answer and being able to generate it are different capabilities.** DPO trains the first. Only SFT trains the second.

---

## The Central Finding

The most interesting result is the **Effective Date control case** (Q3 and Q8).

The preference dataset contains the correct answer ("March 20, 2018"). DPO was explicitly trained to prefer it over the wrong answer ("September 30, 2014"). `rewards/accuracies` reached 1.0 — the model does prefer the correct answer when choosing between the two.

Yet it cannot *generate* it, because SFT saw that fact once in 104 examples. There was no internal representation for DPO to sharpen.

Compare to the drug question (Q1): SFT saw that fact 8 times (after enrichment), so the representation exists and DPO preserved it. Same model, same DPO run, same preference mechanism — different data coverage in SFT.

> **DPO sharpens existing knowledge; it does not install new facts. Fact coverage must be established in SFT before DPO can refine it.**

---

## What Would Actually Help

1. **Enrich every key fact to 6–8 varied phrasings in the SFT data** — the same intervention already validated on the drug name.
2. **Balance chosen/rejected lengths** in the preference data to remove the length shortcut.
3. **Stop DPO earlier or raise beta above 0.1** to constrain drift from the SFT reference model.
4. **Add a held-out eval split to Stage 2** so overfitting is measured at minimum eval loss rather than guessed.

---

## Overall Verdict

| Criterion | Base | CPT | SFT | DPO |
|---|---|---|---|---|
| Answers the question | ✗ | ✗ | ✓ | ✓ |
| Contract vocabulary | ✗ | Partial | ✓ | ✓ |
| Stops cleanly | ✗ | ✗ | ✓ | ✓ |
| Enriched fact correct | ✗ | ✗ | ✓ | ✓ |
| Rare facts correct | ✗ | ✗ | ✗ | ✗ |
| Coherent output | ✓ | ✓ | ✓ | Partially |

**The pipeline works end-to-end**, with each stage contributing a distinct, measurable capability. SFT delivered instruction-following completely and factual recall selectively. DPO demonstrated its own structural limit — it can sharpen preferences but cannot substitute for data coverage.

**For deployment, the SFT model is the stronger artifact.** DPO's value in this project is as a finding, not as an improvement.
