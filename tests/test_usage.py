from transformers import AutoTokenizer, FlaxAutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("fukugawa/transformer-lm-japanese-0.1b", trust_remote_code=True)
model = FlaxAutoModelForCausalLM.from_pretrained("fukugawa/transformer-lm-japanese-0.1b", trust_remote_code=True)

text = "日本の首都は、"
token_ids = tokenizer.encode(text, return_tensors="jax", add_special_tokens=False)

output_ids = model.generate(
  token_ids,
  do_sample=True,
  temperature=0.6,
  top_k=20,
  max_new_tokens=100
)

output = tokenizer.decode(output_ids[0][0], skip_special_tokens=True)
print(output)
