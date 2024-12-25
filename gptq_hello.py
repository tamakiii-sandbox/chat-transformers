import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "TheBloke/Llama-2-7B-GPTQ"

# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# Move the model to the GPU
model = model.to(device)

# Generate text using the model
prompt = "Once upon a time, in a land far away, there was a"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
output = model.generate(input_ids, max_length=100, num_return_sequences=1)

# Print the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
