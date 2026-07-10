import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
MODEL_ID = "AnilPomar14/tinyllama-pharma-dpo-merged"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN,
    dtype=torch.float32,
    low_cpu_mem_usage=True,      # loads weights in chunks — also needs accelerate
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100,
    do_sample=False,              # greedy — deterministic, correct for facts
    repetition_penalty=1.1,
    return_full_text=False,       # strip the echoed prompt
)

llm = HuggingFacePipeline(pipeline=pipe)

# Must match the training format exactly
prompt = PromptTemplate.from_template(
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{question}\n\n### Response:\n"
)

chain = prompt | llm | StrOutputParser()
questions = [
    "What is the Drug that is the subject of the Antares-AMAG Manufacturing Agreement?",
    "Who are the parties to the Manufacturing Agreement?",
    "State the Effective Date of the Manufacturing Agreement.",
]

print("Ready.\n")

for i, question in enumerate(questions, 1):
    answer = chain.invoke({"question": question}).strip()
    print(f"Q{i}: {question}")
    print(f"A{i}: {answer}")
    print("-" * 80)