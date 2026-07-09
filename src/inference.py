"""
Simple inference script for the DPO-aligned TinyLlama pharma contract model.

Usage:
    python inference.py                       # interactive Q&A loop
    python inference.py --ask "your question" # one-shot
    python inference.py --model <id_or_path>  # different model or local dir

Authentication (only needed for PRIVATE HuggingFace repos):
    export HF_TOKEN=hf_xxx          # then run normally
    -- or --
    huggingface-cli login           # one-time, cached in ~/.cache/huggingface
    -- or --
    python inference.py --model /local/path/to/merged_model   # no auth at all
"""

import argparse
import os
import sys

import torch
from unsloth import FastLanguageModel

# --- Config: must match training -------------------------------------------

MODEL_ID = "AnilPomar14/tinyllama-pharma-dpo-merged"
MAX_SEQ_LENGTH = 512

# NOT optional. This is the exact prompt format the model was trained on.
# A mismatch here (even the trailing newline) badly degrades output.
ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)


def hf_login():
    """
    Authenticate with HuggingFace if a token is available.

    Only required for private repos. Order of preference:
      1. HF_TOKEN environment variable
      2. Colab secret named 'HuggingFace_Write'
      3. Cached credentials from `huggingface-cli login`  (nothing to do)

    Local model paths need none of this.
    """
    token = os.environ.get("HF_TOKEN")

    if not token:
        try:                                        # Colab secrets, if present
            from google.colab import userdata
            token = userdata.get("HuggingFace_Write")
        except Exception:
            pass

    if token:
        from huggingface_hub import login
        login(token=token)
        print("Authenticated with HuggingFace.")
    else:
        # Either the repo is public, the model is local, or `huggingface-cli
        # login` was already run. from_pretrained will surface a clear error
        # if credentials turn out to be required.
        pass


def load_model(model_id):
    is_local = os.path.isdir(model_id)

    if not is_local:
        hf_login()

    print(f"Loading {model_id} ...")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,           # auto-selects bf16 where supported
            load_in_4bit=True,
        )
    except Exception as e:
        msg = str(e)
        if "401" in msg or "403" in msg or "gated" in msg.lower():
            print(
                f"\nAuth error loading '{model_id}'.\n"
                "If this repo is private, set a token:\n"
                "    export HF_TOKEN=hf_your_read_token\n"
                "or make the repo public, or pass a local path with --model.\n",
                file=sys.stderr,
            )
        raise

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    FastLanguageModel.for_inference(model)
    print("Ready.\n")
    return model, tokenizer


def ask(model, tokenizer, question, max_new_tokens=150):
    """
    Greedy decoding (do_sample=False) is deliberate: it returns the model's
    single most-confident answer and is deterministic. Sampling would inject
    randomness and produce fluent-sounding fabrications.
    """
    prompt = ALPACA_TEMPLATE.format(instruction=question.strip())
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the new tokens, dropping the echoed prompt.
    generated = output[0][inputs["input_ids"].shape[-1]:]
    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()

    # If the model tries to start a new turn, cut it there.
    for stop in ("### Instruction:", "### Response:"):
        if stop in answer:
            answer = answer.split(stop)[0].strip()

    return answer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL_ID,
                   help="HF repo id or local directory")
    p.add_argument("--ask", default=None, help="Ask one question and exit")
    args = p.parse_args()

    model, tokenizer = load_model(args.model)

    if args.ask:
        print(ask(model, tokenizer, args.ask))
        return

    print("Pharma Contract Assistant — type 'quit' to exit.\n")
    while True:
        try:
            q = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if not q:
            continue
        if q.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break
        print(f"\nModel > {ask(model, tokenizer, q)}\n")


if __name__ == "__main__":
    main()
