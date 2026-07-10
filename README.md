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

Dataset sample counts discovered in the data folder:
- instruction_dataset.jsonl: 104 samples
- preference_dataset.jsonl: 50 samples

The data contains questions such as:
- drug name
- parties to the agreement
- effective date
- quality agreement details
- delivery and recall terms
- transfer price and minimum order requirements

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

## 7. Instruction Fine-tuning Approach
The instruction fine-tuning stage uses supervised fine-tuning (SFT) on instruction-response data.

Approach:
- format prompts in an instruction-following style
- train the model to answer questions directly
- teach response structure and factual grounding for contract-related questions

Data format used for instruction fine-tuning:
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

Data format used for DPO alignment:
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

## 13. Future Improvements
The model reliably recalls facts appearing 8+ times in the training data but confidently fabricates those appearing 1–6 times. The next step is to enrich the remaining key facts — Effective Date (1 occurrence), Return/Recall Policy (3), Transfer Price (5), and Quality Agreement (6) — to 6–8 varied phrasings each, applying the same intervention already validated on the drug name, which flipped from fabrication to reliable recall after enrichment from 2 to 8 occurrences.

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

## Summary
This project demonstrates how a small language model can be adapted into a practical domain assistant for contract understanding using a staged workflow: domain adaptation, instruction tuning, and preference alignment.
