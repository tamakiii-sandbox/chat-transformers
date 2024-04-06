from transformers import LlamaForCausalLM, LlamaTokenizer
from vllm import LLM, SamplingParams
import bitsandbytes as bnb
import torch.nn as nn

# Load the Llama-2-7b-chat-hf model
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

# Quantize the model
quantized_model = bnb.nn.Linear4bit(
    model.base_model.embed_tokens.weight,
    model.base_model.embed_tokens.weight.dtype,
    has_fp16_weights=True,
    threshold=6.0,
    perchannel=True,
    bias=True
)
model.base_model.embed_tokens.weight = nn.Parameter(quantized_model.weight)

# Initialize the vLLM engine with the quantized model
llm = LLM(
    model=model,
    tokenizer_name_or_path="meta-llama/Llama-2-7b-chat-hf",
    device_config="cuda"
)

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
outputs = llm.generate(prompts, sampling_params)

# Print the outputs
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
