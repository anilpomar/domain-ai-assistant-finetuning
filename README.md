# Domain AI Assistant Fine-tuning

## 1. Project Title
Domain AI Assistant Fine-tuning

## 2. Domain Selected
Pharmaceutical and manufacturing contract domain

This project focuses on the Antares–AMAG Manufacturing Agreement, a commercial/legal document related to pharmaceutical manufacturing, device supply, quality agreements, recalls, and product delivery terms.

## 3. Business Problem
The business challenge is to build a domain-specific assistant that can accurately answer questions about a complex contract document. A general-purpose language model tends to be fluent but often fabricates facts. The goal is to make the model more reliable for contract Q&A in a regulated and specialized domain.

## 4. Dataset Details
The project uses two main datasets:

- data/instruction_dataset.jsonl: instruction-response pairs used for supervised fine-tuning
- data/preference_dataset.jsonl: chosen/rejected response pairs used for DPO alignment

Dataset sample counts (from data/):
- instruction_dataset.jsonl: 104 samples
- preference_dataset.jsonl: 50 samples

The data contains questions such as:
- drug name
- parties to the agreement
- effective date
- quality agreement details
- delivery and recall terms
- transfer price and minimum order requirements

Reference dataset and synthetic data:
- Referred to the Hugging Face CUAD contract corpus for domain context and structure: [CUAD Part I](https://huggingface.co/datasets/theatticusproject/cuad/tree/main/CUAD_v1/full_contract_txt/Part_I)
- Created synthetic training data using Claude to expand the instruction and preference datasets for fine-tuning.

## 5. Base Model Used
Base model: unsloth/tinyllama-bnb-4bit

This is a compact TinyLlama-based model adapted for efficient fine-tuning using 4-bit quantization.

## 6. Non-instruction Fine-tuning Approach
The non-instruction fine-tuning stage is a continued pretraining-style adaptation on domain text.

Approach:
- train on contract-domain text extracted from the agreement
- teach domain vocabulary, style, and terminology
- improve familiarity with legal/commercial phrasing

This stage helps the model better understand the target domain before it learns to answer questions.

**Stage 1 — Continued Pretraining (CPT).** 25 packed chunks from the Antares–AMAG contract, 100 steps at LR 2e-4, `packing=True`. This teaches the model to *continue text* in the contract's domain and style. Loss descended 2.02 → 0.88; `lora_B` reached 0.0204, confirming the adapter genuinely learned. CPT does **not** teach instruction-following — tested with a question, this model still rambles, which is correct behavior.

## 7. Instruction Fine-tuning Approach
The instruction fine-tuning stage uses supervised fine-tuning (SFT) on instruction-response data.

Approach:
- format prompts in an instruction-following style
- train the model to answer questions directly
- teach response structure and factual grounding for contract-related questions

**Stage 2 — Supervised Fine-Tuning (SFT).** 104 Alpaca-format instruction/response pairs, 150 steps at LR 2e-4, `packing=False`. Three implementation details were essential:

- **Prompt template** applied identically at training and inference, including the trailing newline after `### Response:`
- **EOS token** appended to each response, or the model never learns where to stop
- **Response-only masking** — prompt tokens set to `-100` so loss is computed only on the answer, implemented via deterministic offset masking rather than token-pattern search

Loss descended 1.84 → 0.0018.

**Data format (Instruction SFT):**
```json
{
  "instruction": "What is the Drug that is the subject of the Antares–AMAG Manufacturing Agreement?",
  "input": "",
  "response": "The Drug is 17-alpha hydroxyprogesterone caproate."
}
```

## 8. DPO Alignment Approach
The DPO stage aligns the model using preference pairs.

Approach:
- compare a preferred answer vs. a rejected answer
- train the model to prefer more accurate and higher-quality responses
- sharpen the behavior of the already instruction-tuned model

**Stage 3 — Direct Preference Optimization (DPO).** 50 `(prompt, chosen, rejected)` triples, 15 steps at LR 5e-5, beta 0.1, loading the Stage 2 merged model with `ref_model=None` (the frozen base acts as implicit reference under LoRA).

Each stage's adapter was merged into the base weights before the next stage loaded it.

**Data format (DPO Alignment):**
```json
{
  "prompt": "What is the Effective Date of the Manufacturing Agreement?",
  "chosen": "The Effective Date is March 20, 2018.",
  "rejected": "The Effective Date is September 30, 2014, the same date as the underlying Development and License Agreement."
}
```

## 9. LoRA / QLoRA Configuration
The notebooks use LoRA-based fine-tuning with 4-bit quantization.

Key settings:
- LoRA rank: 16
- LoRA alpha: 32
- LoRA dropout: 0
- base model: 4-bit quantized TinyLlama
- training style: Unsloth + PEFT + TRL workflow

These settings make training memory-efficient while preserving strong adaptation performance.

## 10. Training Screenshots or Logs
Training artifacts and supporting reports are stored in the reports/ folder, including the additional reports referenced below.

Additional report files:
- [reports/Additional_Reports/Fine Tuning Results.docx](reports/Additional_Reports/Fine%20Tuning%20Results.docx)
- [reports/Additional_Reports/Final_Comparison_Table.docx](reports/Additional_Reports/Final_Comparison_Table.docx)

Other supporting outputs:
- [reports/base_model_evaluation.md](reports/base_model_evaluation.md)
- [reports/sft_model_comparison.md](reports/sft_model_comparison.md)
- [reports/fine_tuning_explanation.md](reports/fine_tuning_explanation.md)
- [reports/final_evaluation.md](reports/final_evaluation.md)

These materials document the training logs, training loss behavior, and qualitative before/after evaluation results.

## 11. Before vs After Output Comparison
The referenced report files show a clear difference between the base model and the fine-tuned models.

### Before fine-tuning (base model)
The base model produced fluent but fabricated answers. For example, when asked about the drug in the agreement, it generated an invented drug name and even included a fabricated FDA-style citation.

Example:
- Question: What is the Drug that is the subject of the Antares–AMAG Manufacturing Agreement?
- Base output: a fabricated answer involving "NBUD-BRL" and a made-up citation

It also failed on other contract questions, such as the parties, by responding with a generic contract template and naming Amazon instead of the correct contracting parties.

### After fine-tuning (SFT / DPO)
After instruction tuning, the model became much better at answering directly and using contract-specific vocabulary. The SFT model correctly answered the drug question with the real drug name: 17-alpha hydroxyprogesterone caproate.

The DPO model preserved this fact but showed mixed behavior on other questions. It maintained the corrected drug answer, yet on some other questions it became more verbose and introduced invented details, which suggests DPO sharpened preference but did not fully solve rare factual recall.

### Summary of comparison
- Base model: fluent, confident, but fabricated
- SFT model: strongest overall improvement in direct question answering
- DPO model: preserved the enriched fact, but did not consistently improve every rare fact

## 12. Final Observations
The final comparison report indicates that the overall pipeline worked, but with an important limitation.

Key observations:
- SFT provided the clearest improvement in instruction-following and domain-specific question answering.
- DPO preserved the one well-trained fact that had been explicitly reinforced in the training data.
- DPO did not reliably fix rare facts such as the effective date, even when the preference dataset contained a relevant comparison pair.
- The project demonstrates that preference optimization can sharpen behavior, but it does not replace the need for strong factual coverage in the training data.

Overall conclusion:
The fine-tuned assistant is significantly better than the base model, especially for the targeted contract domain, but factual accuracy for rare or underrepresented details remains a data challenge rather than a purely training-algorithm challenge.

## 13. Challenges faced

1. **Silent Training failure fp16 gradient collapse** → Early training runs produced a loss curve that descended plausibly (2.29 → 1.82) while the model learned nothing. Merged models were behaviourally identical to the untrained base model.Migrate to bf16-capable GPU (L4) 
### Resolution
Migrated to an L4 GPU (Ada architecture, native bf16). bf16 has fp32's full dynamic range in 16 bits — no overflow, no GradScaler, no crushed gradients. Gradient norms rose to a healthy 0.35–0.65 and lora_B reached 0.0204

2. **Version drift** → A working notebook broke with no code change.The new Transformers was incompatible with the installed Unsloth build. 
### Resolution
Reinstalled the stack in a fresh session. Pinned dependency versions at the top of the notebook so Colab cannot move the foundation underneath the code.

3. **Tokenizer boundary masking** →Used train_on_responses_only which searches for the token-ID subsequence of the marker, not the string.  SentencePiece merges the trailing newline with the first character of the following word, so the marker's standalone tokens never appear as a contiguous run inthe tokenized sequence. Removing the newline from the marker did not help either
### Resolution
Moved to Deterministic offset-based masking by locate the marker by string position

4. **Diluted gradient from masking** → At STAGE2_LR = 2e-5 , Stage 2's lora_B reached only 0.0017 after 150 steps — against 0.0204 for Stage 1 at just 100 steps. Response-only masking sets prompt tokens to -100 , so only the response (roughly 10–20% of each sequence) contributes to the loss
### Resolution
Raise Stage 2 LR from 2e-5 to 2e-4.  A masked SFT stage needs a higher learning rate than an unmasked CPT stage running the same number of steps

5. **"Broken" vs "not finished"** → After migrating to the L4, lora_B still read 0.0008. The run was only 15 steps. lora_B starts at exactly zero and cannot grow meaningfully. Checked gradient norms first. 
### Resolution
Extending Stage 1 to 100 steps grew lora_B to 0.0204 

6. **Fact hallucination** → Once training genuinely worked, the model still fabricated the drug name —producing a different invented compound on each run ( NBI-495 → NBI-4157 → NBI-49544 ). Increasing steps from 60 to 160 made the prose more fluent but the facts no less wrong. 
### Resolution
Enriched the fact to 8 occurrences across varied question phrasings

7. **Misleading evaluation from sampling** → Same question produced three different invented drug names across runs because of Inference used do_sample=True with temperature=0.7 injecting randomness.
### Resolution
Switched all factual evaluation to greedy decoding ( do_sample=False ), which is deterministic and returns the model's single most-confident answer.

8. **DPO over-optimization** → An initial 60-step DPO run appeared successful by the headline metric —rewards/margins grew to +13.8.  Reduced to 15 steps. rewards/chosen stayed positive (+1.14) and  and margins reached a healthy +4.64. Watch rewards/chosen , not just margins. Margins can grow while both sides get worse.

9. **DPO cannot teach facts** → Fix upstream in SFT data coverage (1 of 104 SFT examples had date sample so The aligned model still could not produce the date.)
### Resolution
Will be fixed for future improvement.

## 14. Future Improvements
- The model reliably recalls facts appearing 8+ times in the training data but confidently fabricates those appearing 1–6 times. The next step is to enrich the remaining key facts — Effective Date (1 occurrence), Return/Recall Policy (3), Transfer Price (5), and Quality Agreement (6) — to 6–8 varied phrasings each, applying the same intervention already validated on the drug name, which flipped from fabrication to reliable recall after enrichment from 2 to 8 occurrences.

- Design a website that helps user to interact

## Repository Structure

- data/ - datasets used for training
- notebooks/ - training notebooks for CPT/SFT/DPO
- reports/ - evaluation reports and notes
- src/ - inference script

## Running Inference

```bash
python src/inference.py
```

One-shot example:

```bash
python src/inference.py --ask "What is the drug in the agreement?"
```

If needed, set a Hugging Face token:

```bash
export HF_TOKEN=hf_your_token
```

## Output

Below is a short sample run of the inference script. See the implementation in [src/inference.py](src/inference.py#L1-L74).

```
Ready.

Q1: What is the Drug that is the subject of the Antares-AMAG Manufacturing Agreement?
A1: It is 17-alpha hydroxyprogesterone caproate.
--------------------------------------------------------------------------------
Q2: Who are the parties to the Manufacturing Agreement?
A2: They are Antares and AMAG, respectively.
--------------------------------------------------------------------------------
Q3: State the Effective Date of the Manufacturing Agreement.
A3: It becomes an agreement effectiveness date ahead of the parties both becoming parties to it.
--------------------------------------------------------------------------------
```

## Summary
 
![Inference run screenshot](reports/images/inference-run.png)
This project demonstrates how a small language model can be adapted into a practical domain assistant for contract understanding using a staged workflow: domain adaptation, instruction tuning, and preference alignment.
