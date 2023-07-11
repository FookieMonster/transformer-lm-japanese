# Copyright 2023 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate text using pre-trained weights."""

import functools
import os

from absl import app
from absl import flags
from absl import logging
from clu import platform
from flax import jax_utils
from flax import linen as nn
from flax.training import checkpoints
from flax.training import train_state
import jax
from jax import random
import jax.numpy as jnp
import ml_collections
from ml_collections import config_flags
import numpy as np
import optax
import tensorflow as tf

import models
import temperature_sampler
import tokenizer
import train

FLAGS = flags.FLAGS

flags.DEFINE_string('workdir', None, 'Directory to store model data.')
flags.DEFINE_integer('num_generated_texts', 10, 'Number of texts to generate.')
config_flags.DEFINE_config_file(
  'config',
  'configs/default.py',
  'File path to the training hyperparameter configuration.',
  lock_config=True)
flags.mark_flags_as_required(['config', 'workdir'])


def generate_text(config: ml_collections.ConfigDict, workdir: str, num_generated_texts: int):
  """Generate text.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints.
    num_generated_texts: Number of texts to generate.
  """
  tf.io.gfile.makedirs(workdir)

  vocab_path = config.vocab_path
  if vocab_path is None:
    vocab_path = os.path.join(workdir, "sentencepiece_model")
    config.vocab_path = vocab_path
  tf.io.gfile.makedirs(os.path.split(vocab_path)[0])

  # Load Tokenizer
  # ---------------------------------------------------------------------------
  logging.info("Loading tokenizer.")
  encoder = tokenizer._load_sentencepiece_tokenizer(model_path=vocab_path)

  vocab_size = int(encoder.vocab_size())
  eos_id = temperature_sampler.EOS_ID  # Default Sentencepiece EOS token.

  def decode_tokens(toks):
    valid_toks = toks[:np.argmax(toks==eos_id) + 1].astype(np.int32)
    return encoder.detokenize(valid_toks).numpy().decode("utf-8")

  def encode_strings(strs, max_len):
    tokenized_batch = np.zeros((len(strs), max_len), np.int32)
    for i, s in enumerate(strs):
      toks = encoder.tokenize(s).numpy()
      # Remove EOS token in prompt.
      tokenized_batch[i, :toks.shape[0] - 1] = toks[:-1]
    return tokenized_batch

  tokenized_prompts = encode_strings(
    [config.prompts], config.max_predict_length)

  logging.info("Initializing model, optimizer, and step functions.")
  # Build Model and Optimizer
  # ---------------------------------------------------------------------------
  train_config = models.TransformerConfig(
    vocab_size=vocab_size,
    output_vocab_size=vocab_size,
    logits_via_embedding=config.logits_via_embedding,
    dtype=jnp.bfloat16 if config.use_bfloat16 else jnp.float32,
    emb_dim=config.emb_dim,
    num_heads=config.num_heads,
    num_layers=config.num_layers,
    qkv_dim=config.qkv_dim,
    mlp_dim=config.mlp_dim,
    max_len=max(config.max_target_length, config.max_eval_target_length),
    dropout_rate=config.dropout_rate,
    attention_dropout_rate=config.attention_dropout_rate,
    deterministic=False,
    decode=False,
    kernel_init=nn.initializers.xavier_uniform(),
    bias_init=nn.initializers.normal(stddev=1e-6))
  eval_config = train_config.replace(deterministic=True)
  predict_config = train_config.replace(deterministic=True, decode=True)

  rng = jax.random.PRNGKey(config.seed)
  rng, init_rng = jax.random.split(rng)
  rng, inference_rng = random.split(rng)
  input_shape = (config.per_device_batch_size, config.max_target_length)

  m = models.TransformerLM(eval_config)
  initial_variables = jax.jit(m.init)(init_rng,
    jnp.ones(input_shape, jnp.float32))

  learning_rate_fn = train.create_learning_rate_schedule(
    learning_rate=config.learning_rate, warmup_steps=config.warmup_steps)

  optimizer = optax.adamw(
    learning_rate_fn, b1=0.9, b2=0.98, eps=1e-9,
    weight_decay=config.weight_decay
  )
  state = train_state.TrainState.create(
    apply_fn=m.apply,
    params=initial_variables["params"],
    tx=optimizer
  )

  # Restore unreplicated optimizer + model state from last checkpoint.
  state = checkpoints.restore_checkpoint(workdir, state)
  # Grab last step.
  int(state.step)

  # Replicate optimizer.
  state = jax_utils.replicate(state)

  p_pred_step = jax.pmap(
    functools.partial(
      train.predict_step, config=predict_config,
      temperature=config.sampling_temperature,
      top_k=config.sampling_top_k),
    axis_name="batch",
    static_broadcasted_argnums=(3, 4))  # eos token, max_length are constant

  keys = jax.random.split(inference_rng, num=num_generated_texts)
  for key in keys:
    train.generate_prediction(
      p_pred_step=p_pred_step,
      params=state.params,
      tokenized_prompts=tokenized_prompts,
      eos_id=eos_id,
      inference_rng=key,
      decode_tokens=decode_tokens,
      max_predict_length=config.max_predict_length)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
  # it unavailable to JAX.
  tf.config.experimental.set_visible_devices([], 'GPU')

  logging.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
  logging.info('JAX local devices: %r', jax.local_devices())

  # Add a note so that we can tell which task is which JAX host.
  # (Depending on the platform task 0 is not guaranteed to be host 0)
  platform.work_unit().set_task_status(f'process_index: {jax.process_index()}, '
                                       f'process_count: {jax.process_count()}')
  platform.work_unit().create_artifact(platform.ArtifactType.DIRECTORY,
    FLAGS.workdir, 'workdir')

  generate_text(FLAGS.config, FLAGS.workdir, FLAGS.num_generated_texts)


if __name__=='__main__':
  jax.config.config_with_absl()
  app.run(main)