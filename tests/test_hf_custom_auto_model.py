import unittest

import jax.numpy as jnp
from transformers import AutoTokenizer, FlaxAutoModelForCausalLM


class TestTransformerLM(unittest.TestCase):

  def test_flax_auto_model(self):
    tokenizer = AutoTokenizer.from_pretrained("fukugawa/transformer-lm-japanese-0.1b", trust_remote_code=True)
    model = FlaxAutoModelForCausalLM.from_pretrained("fukugawa/transformer-lm-japanese-0.1b", trust_remote_code=True, dtype=jnp.float32)
    input_text = "日本の首都は、"
    input_ids = tokenizer.encode(input_text, return_tensors="jax", add_special_tokens=False)
    output = model.generate(input_ids, do_sample=True, temperature=0.6, top_k=20, max_length=100)
    generated_text = tokenizer.decode(output[0].flatten(), skip_special_tokens=True)
    print(generated_text)


if __name__=='__main__':
  unittest.main()
