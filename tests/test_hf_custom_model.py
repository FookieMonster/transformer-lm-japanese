import os
import textwrap
import unittest

import jax
import jax.numpy as jnp
from flax.training import checkpoints
from huggingface_hub import hf_hub_download

from hf_custom_model.transformer_lm.configuration_transformerlm import TransformerLMConfig
from hf_custom_model.transformer_lm.modeling_transformerlm_flax import FlaxTransformerLMForCausalLM
from hf_custom_model.transformer_lm.tokenization_transformerlm import TransformerLMTokenizer

REPO_ID = "fukugawa/transformer-lm-japanese-0.1b"
REPO_VER = "v1"
LOCAL_DIR = "../hf_custom_model/transformer_lm/files"
VOCAB_FILE = "../hf_custom_model/transformer_lm/files/sentencepiece_model"
CHECKPOINT_FILE = "../hf_custom_model/transformer_lm/files/checkpoint_499999"


def copy_nested_params(left_params, right_params):
  new_params = {}
  for key, value in left_params.items():
    if key in right_params:
      if isinstance(value, dict):
        new_params[key] = copy_nested_params(value, right_params[key])
      else:
        new_params[key] = right_params[key]
    else:
      new_params[key] = value
  return new_params


class TestFlaxTransformerLMForCausalLM(unittest.TestCase):

  def setUp(self):
    if not os.path.exists(VOCAB_FILE):
      hf_hub_download(
        repo_id=REPO_ID,
        filename="sentencepiece_model",
        revision=REPO_VER,
        local_dir=LOCAL_DIR,
      )
    if not os.path.exists(CHECKPOINT_FILE):
      hf_hub_download(
        repo_id=REPO_ID,
        filename="checkpoint_499999",
        revision=REPO_VER,
        local_dir=LOCAL_DIR,
      )
    self.config = TransformerLMConfig(
      vocab_size=30000,
      output_vocab_size=30000,
      logits_via_embedding=False,
      emb_dim=768,
      num_heads=12,
      num_layers=12,
      qkv_dim=768,
      mlp_dim=3072,
      max_len=max(256, 512),
      dropout_rate=0.1,
      attention_dropout_rate=0.1,
      deterministic=True,
      decode=True,
    )
    self.tokenizer = TransformerLMTokenizer(
      vocab_file=VOCAB_FILE
    )
    restored_params = checkpoints.restore_checkpoint(CHECKPOINT_FILE, target=None)
    self.model = FlaxTransformerLMForCausalLM(self.config, dtype=jnp.float32)
    self.model.params = copy_nested_params(self.model.params, restored_params['params'])
    self.model.params['logitdense'] = restored_params['params']['decoder']['logitdense']

  def test_config(self):
    print(self.config)

  def test_tokenizer(self):
    input_text = "日本の首都は、"
    input_ids = self.tokenizer.encode(input_text, return_tensors="jax", add_special_tokens=False)
    output = input_ids
    generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)

    print(self.tokenizer.tokenize(input_text))
    print(input_ids)
    print(generated_text)

  def test_next_word_prediction(self):
    input_text = "日本の首都は、"
    input_ids = self.tokenizer.encode(input_text, return_tensors="jax", add_special_tokens=False)
    outputs = self.model(input_ids)
    logits = outputs.logits

    print("input_tokens:", self.tokenizer.tokenize(input_text))
    print("input_ids:", input_ids)
    print("input_ids.shape:", input_ids.shape)
    print("logits.shape:", logits.shape)

    self.printTopNTokens(logits, self.tokenizer, 10)

  def test_log_likelihood(self):
    texts = [
      "日本の首都は東京",
      "日本の首都はニューヨーク",
      "日本の首都はロンドン",
      "日本の首都は京都",
    ]
    likelihoods = []

    for text in texts:
      tokens = self.tokenizer.encode(text, return_tensors="jax", add_special_tokens=False)
      outputs = self.model(tokens)
      logits = outputs.logits

      probs = jax.nn.softmax(logits, axis=-1)
      true_probs = jnp.log(jnp.take_along_axis(probs, tokens[:, :, None], axis=-1).squeeze(-1))
      likelihood = jnp.sum(true_probs, axis=-1)
      likelihoods.append(likelihood.item())

    for (text, likelihood) in zip(texts, likelihoods):
      print(f"Log Likelihood for '{text}': {likelihood}")

  def test_greedy_search(self):
    max_length = 20
    input_text = "日本の首都は、"
    input_ids = self.tokenizer.encode(input_text, return_tensors="jax", add_special_tokens=False)

    model_kwargs = self.model.prepare_inputs_for_generation(input_ids, max_length, None)

    sequences = input_ids
    running_token = input_ids

    for i in range(max_length):
      model_outputs = self.model(running_token, params=None, **model_kwargs)
      logits = model_outputs.logits[:, -1]

      next_token = jnp.argmax(logits, axis=-1)
      sequences = jnp.append(sequences, next_token)
      running_token = jnp.array([next_token])

      model_kwargs = self.model.update_inputs_for_generation(model_outputs, model_kwargs)

    print(sequences)

    generated_text = self.tokenizer.decode(sequences, skip_special_tokens=True)
    print(generated_text)

  def test_text_generation_by_sampling(self):
    input_text = "日本の首都は、"
    input_ids = self.tokenizer.encode(input_text, return_tensors="jax", add_special_tokens=False)
    output = self.model.generate(input_ids, do_sample=True, temperature=0.6, top_k=20, max_length=100)
    generated_text = self.tokenizer.decode(output[0][0], skip_special_tokens=True)
    print(generated_text)

  def test_text_generation_by_greedy_search(self):
    input_text = "日本の首都は、"
    input_ids = self.tokenizer.encode(input_text, return_tensors="jax", add_special_tokens=False)
    output = self.model.generate(input_ids, do_sample=False, max_length=100)
    generated_text = self.tokenizer.decode(output[0][0], skip_special_tokens=True)
    print(generated_text)

  def test_text_generation_with_jcommonsenseqa(self):
    input_text = textwrap.dedent("""
      与えられた選択肢の中から、最適な答えを選んでください。
      
      質問：生理現象なのは？
      選択肢：
      - 準備する
      - おしっこする
      - 風
      - 雨
      - ベッドに入る
      回答：
    """)
    input_ids = self.tokenizer.encode(input_text, return_tensors="jax", add_special_tokens=False)
    output = self.model.generate(input_ids, do_sample=False, max_new_tokens=100)
    generated_text = self.tokenizer.decode(output[0][0], skip_special_tokens=True)
    print(generated_text)

  def test_text_generation_with_jsquad(self):
    input_text = textwrap.dedent("""
      以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。
      
      ### 指示:
      与えられた文脈から、質問に対する答えを抜き出してください。
      
      ### 入力:
      文脈：リスボン、ポルト、ファロがおもな国際空港。またこれらの空港から、マデイラ諸島やアソーレス諸島などの離島への路線も出ている。
      質問：マデイラ諸島やアソーレス諸島などの離島への路線も出ているのは？
      
      ### 応答:
    """)
    input_ids = self.tokenizer.encode(input_text, return_tensors="jax", add_special_tokens=False)
    output = self.model.generate(input_ids, do_sample=False, max_new_tokens=100)
    generated_text = self.tokenizer.decode(output[0][0], skip_special_tokens=True)
    print(generated_text)

  @staticmethod
  def printTopNTokens(logits, tokenizer, topN):
    probs = jax.nn.softmax(logits[0, -1, :])

    index = jax.numpy.argmax(probs, axis=-1)
    token = tokenizer.decode(index)
    prob = probs[index]
    prob = prob.item()
    print("----------------------------------------------")
    print(f"Next word: {token}, id: {index}, prob: {prob}")
    print("----------------------------------------------")

    def get_top_k_probs(logits, k=10):
      probs = jax.nn.softmax(logits[0, -1, :])
      top_k_probs, top_k_ids = jax.lax.top_k(probs, k)
      return top_k_probs, top_k_ids

    top_k_probs, top_k_ids = get_top_k_probs(logits, k=topN)

    decoded_tokens = [tokenizer.decode(idx) for idx in top_k_ids]
    for token, prob in zip(decoded_tokens, top_k_probs):
      print(f"Token: {token}, Prob: {prob}")
