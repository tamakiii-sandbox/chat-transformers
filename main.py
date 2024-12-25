from transformers import LlamaForCausalLM, LlamaTokenizer
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Load the model and tokenizer
# model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
# tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

# Move the model to the GPU
model.to(device)

# Set up the conversational context
context = ""

# Conversation loop
while True:
    user_input = input("You: ")

    # Append the user's input to the context
    input_text = context + " " + user_input

    # Encode the input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    # Generate a response
    output_ids = model.generate(input_ids, max_length=100, num_return_sequences=1, do_sample=True, top_k=50, top_p=0.95, num_beams=5, device=device)[0]
    response = tokenizer.decode(output_ids, skip_special_tokens=True)

    # Print the model's response
    print("Model:", response)

    # Update the conversational context
    context = input_text + " " + response
