from transformers import LlamaForCausalLM, LlamaTokenizer

model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

prompt = "Once upon a time, there was a"
input_ids = tokenizer.encode(prompt, return_tensors="pt")
output = model.generate(input_ids, max_length=50, num_return_sequences=1, do_sample=True, top_k=50, top_p=0.95, num_beams=5)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
