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

from typing import Callable, Any, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax import struct
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from transformers.modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutput
from transformers.modeling_flax_utils import FlaxPreTrainedModel

from .configuration_transformerlm import TransformerLMConfig


@struct.dataclass
class TransformerConfig:
  """Global hyperparameters used to minimize obnoxious kwarg plumbing."""
  vocab_size: int
  output_vocab_size: int
  share_embeddings: bool = False
  logits_via_embedding: bool = False
  dtype: Any = jnp.float32
  emb_dim: int = 512
  num_heads: int = 8
  num_layers: int = 6
  qkv_dim: int = 512
  mlp_dim: int = 2048
  max_len: int = 2048
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  deterministic: bool = False
  decode: bool = False
  kernel_init: Callable = nn.initializers.xavier_uniform()
  bias_init: Callable = nn.initializers.normal(stddev=1e-6)
  posemb_init: Optional[Callable] = None


def shift_right(x, axis=1):
  """Shift the input to the right by padding and slicing on axis."""
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[axis] = (1, 0)
  padded = jnp.pad(
    x, pad_widths, mode='constant', constant_values=x.dtype.type(0))
  return lax.dynamic_slice_in_dim(padded, 0, padded.shape[axis] - 1, axis)


def shift_inputs(x, segment_ids=None, axis=1):
  """Shift inputs and replace EOS by 0 for packed inputs."""
  shifted = shift_right(x, axis=axis)
  # For packed targets, the first shifted token of a new sequence is made
  # 0, rather than being the EOS token for the last sequence.
  if segment_ids is not None:
    shifted *= (segment_ids==shift_right(segment_ids, axis=axis))
  return shifted


def sinusoidal_init(max_len=2048,
                    min_scale=1.0,
                    max_scale=10000.0):
  """1D Sinusoidal Position Embedding Initializer.

  Args:
      max_len: maximum possible length for the input.
      min_scale: float: minimum frequency-scale in sine grating.
      max_scale: float: maximum frequency-scale in sine grating.

  Returns:
      output: init function returning `(1, max_len, d_feature)`
  """

  def init(key, shape, dtype=np.float32):
    """Sinusoidal init."""
    del key, dtype
    d_feature = shape[-1]
    pe = np.zeros((max_len, d_feature), dtype=np.float32)
    position = np.arange(0, max_len)[:, np.newaxis]
    scale_factor = -np.log(max_scale / min_scale) / (d_feature // 2 - 1)
    div_term = min_scale * np.exp(np.arange(0, d_feature // 2) * scale_factor)
    pe[:, :d_feature // 2] = np.sin(position * div_term)
    pe[:, d_feature // 2: 2 * (d_feature // 2)] = np.cos(position * div_term)
    pe = pe[np.newaxis, :, :]  # [1, max_len, d_feature]
    return jnp.array(pe)

  return init


class AddPositionEmbs(nn.Module):
  """Adds (optionally learned) positional embeddings to the inputs.

  Args:
    config: TransformerConfig dataclass containing hyperparameters.
    decode: whether to run in single-position autoregressive mode.
  """
  config: TransformerConfig
  decode: bool = False

  @nn.compact
  def __call__(self,
               inputs,
               inputs_positions=None):
    """Applies AddPositionEmbs module.

    By default this layer uses a fixed sinusoidal embedding table. If a
    learned position embedding is desired, pass an initializer to
    posemb_init in the configuration.

    Args:
      inputs: input data.
      inputs_positions: input position indices for packed sequences.

    Returns:
      output: `(bs, timesteps, in_dim)`
    """
    config = self.config
    # inputs.shape is (batch_size, seq_len, emb_dim)
    assert inputs.ndim==3, ('Number of dimensions should be 3,'
                            ' but it is: %d' % inputs.ndim)
    length = inputs.shape[1]
    pos_emb_shape = (1, config.max_len, inputs.shape[-1])
    if config.posemb_init is None:
      # Use a fixed (non-learned) sinusoidal position embedding.
      pos_embedding = sinusoidal_init(max_len=config.max_len)(None,
        pos_emb_shape,
        None)
    else:
      pos_embedding = self.param('pos_embedding', config.posemb_init,
        pos_emb_shape)
    pe = pos_embedding[:, :length, :]

    # We use a cache position index for tracking decoding position.
    if self.decode:
      is_initialized = self.has_variable('cache', 'cache_index')
      cache_index = self.variable('cache', 'cache_index',
        lambda: jnp.array(0, dtype=jnp.uint32))
      if is_initialized:
        i = cache_index.value
        cache_index.value = i + 1
        _, _, df = pos_embedding.shape
        pe = lax.dynamic_slice(pos_embedding,
          jnp.array((0, i, 0)),
          (1, 1, df))
    if inputs_positions is None:
      # normal unpacked case:
      return inputs + pe
    else:
      # for packed data we need to use known position indices:
      return inputs + jnp.take(pe[0], inputs_positions, axis=0)


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block.

  Args:
    config: TransformerConfig dataclass containing hyperparameters.
    out_dim: optionally specify out dimension.
  """
  config: TransformerConfig
  out_dim: Optional[int] = None

  @nn.compact
  def __call__(self, inputs):
    """Applies Transformer MlpBlock module."""
    config = self.config
    actual_out_dim = (inputs.shape[-1] if self.out_dim is None
                      else self.out_dim)
    x = nn.Dense(
      config.mlp_dim,
      dtype=config.dtype,
      kernel_init=config.kernel_init,
      bias_init=config.bias_init)(
      inputs)
    x = nn.relu(x)
    x = nn.Dropout(rate=config.dropout_rate)(
      x, deterministic=config.deterministic)
    output = nn.Dense(
      actual_out_dim,
      dtype=config.dtype,
      kernel_init=config.kernel_init,
      bias_init=config.bias_init)(
      x)
    output = nn.Dropout(rate=config.dropout_rate)(
      output, deterministic=config.deterministic)
    return output


class EncoderDecoder1DBlock(nn.Module):
  """Transformer encoder-decoder layer.

  Args:
    config: TransformerConfig dataclass containing hyperparameters.
  """
  config: TransformerConfig

  @nn.compact
  def __call__(self,
               inputs,
               decoder_mask=None,
               encoder_decoder_mask=None):
    """Applies EncoderDecoder1DBlock module.

    Args:
      inputs: input data for decoder
      decoder_mask: decoder self-attention mask.
      encoder_decoder_mask: encoder-decoder attention mask.

    Returns:
      output after transformer encoder-decoder block.
    """
    config = self.config

    # Decoder block.
    assert inputs.ndim==3
    x = nn.LayerNorm(dtype=config.dtype)(inputs)
    x = nn.SelfAttention(
      num_heads=config.num_heads,
      dtype=config.dtype,
      qkv_features=config.qkv_dim,
      kernel_init=config.kernel_init,
      bias_init=config.bias_init,
      use_bias=False,
      broadcast_dropout=False,
      dropout_rate=config.attention_dropout_rate,
      deterministic=config.deterministic,
      decode=config.decode)(x, decoder_mask)
    x = nn.Dropout(rate=config.dropout_rate)(
      x, deterministic=config.deterministic)
    x = x + inputs

    # MLP block.
    z = nn.LayerNorm(dtype=config.dtype)(x)
    z = MlpBlock(config=config)(z)

    return x + z


# Copyright 2021 The Eleuther AI and The Google Flax Team Authors and The HuggingFace Inc. team.
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

class FlaxTransformerLMPreTrainedModel(FlaxPreTrainedModel):
  config_class = TransformerLMConfig
  base_model_prefix = "decoder"
  module_class: nn.Module = None

  def __init__(
      self,
      config: TransformerLMConfig,
      input_shape: Tuple = (1, 1),
      seed: int = 0,
      dtype: jnp.dtype = jnp.bfloat16,
      _do_init: bool = True,
      **kwargs,
  ):
    module = self.module_class(config=config, dtype=dtype, **kwargs)
    super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

  def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
    # init input tensors
    input_ids = jnp.zeros(input_shape, dtype="i4")
    attention_mask = jnp.ones_like(input_ids)
    position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape)
    params_rng, dropout_rng = jax.random.split(rng)
    rngs = {"params": params_rng, "dropout": dropout_rng}

    random_params = self.module.init(rngs, input_ids, attention_mask, position_ids, return_dict=False)["params"]

    if params is not None:
      random_params = flatten_dict(unfreeze(random_params))
      params = flatten_dict(unfreeze(params))
      for missing_key in self._missing_keys:
        params[missing_key] = random_params[missing_key]
      self._missing_keys = set()
      return freeze(unflatten_dict(params))
    else:
      return random_params

  def init_cache(self, batch_size, max_length):
    r"""
    Args:
        batch_size (`int`):
            batch_size used for fast auto-regressive decoding. Defines the batch size of the initialized cache.
        max_length (`int`):
            maximum possible length for auto-regressive decoding. Defines the sequence length of the initialized
            cache.
    """
    # init input variables to retrieve cache
    input_ids = jnp.ones((batch_size, max_length))
    attention_mask = jnp.ones_like(input_ids)
    position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

    init_variables = self.module.init(
      jax.random.PRNGKey(0), input_ids, attention_mask, position_ids, return_dict=False, init_cache=True
    )
    return unfreeze(init_variables["cache"])

  def __call__(
      self,
      input_ids,
      attention_mask=None,
      position_ids=None,
      params: dict = None,
      past_key_values: dict = None,
      dropout_rng: jax.random.PRNGKey = None,
      train: bool = False,
      output_attentions: Optional[bool] = None,
      output_hidden_states: Optional[bool] = None,
      return_dict: Optional[bool] = None,
  ):
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
      output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.return_dict

    batch_size, sequence_length = input_ids.shape

    if position_ids is None:
      if past_key_values is not None:
        raise ValueError("Make sure to provide `position_ids` when passing `past_key_values`.")

      position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

    if attention_mask is None:
      attention_mask = jnp.ones((batch_size, sequence_length))

    # Handle any PRNG if needed
    rngs = {}
    if dropout_rng is not None:
      rngs["dropout"] = dropout_rng

    inputs = {"params": params or self.params}

    # if past_key_values are passed then cache is already initialized a private flag init_cache has to be passed down to ensure cache is used. It has to be made sure that cache is marked as mutable so that it can be changed by FlaxGPTNeoAttention module
    if past_key_values:
      inputs["cache"] = past_key_values
      mutable = ["cache"]
    else:
      mutable = False

    if input_ids.shape[1] > 1:
      input_ids = jnp.insert(input_ids, 0, 0, axis=1) # Insert 0 at the beginning of prompt

    # Progressive cache loop
    if self.module.use_cache:
      batch_size, seq_length = input_ids.shape
      shape = (batch_size, seq_length, self.module.config.vocab_size)
      logits = jnp.zeros(shape, dtype=self.dtype)

      def loop_body_fn(i, state):
        logits, cache = state
        input_id = lax.dynamic_slice(input_ids, (0, i), (input_ids.shape[0], 1))
        output = self.module.apply(
          {
            "params": inputs["params"],
            "cache": cache
          },
          jnp.array(input_id, dtype="i4"),
          jnp.array(attention_mask, dtype="i4"),
          jnp.array(position_ids, dtype="i4"),
          not train,
          False,
          output_attentions,
          output_hidden_states,
          return_dict,
          rngs=rngs,
          mutable=mutable,
        )
        lm_output, new_vars = output
        logits = logits.at[:, i, :].set(lm_output.logits.squeeze(1))
        return logits, new_vars["cache"]

      cache = freeze(inputs["cache"])
      initial_state = (logits, cache)
      lm_logits, lm_cache = lax.fori_loop(0, seq_length, loop_body_fn, initial_state)

      if seq_length > 1:
        lm_logits = lm_logits[:, 1:, :] # Ignore leading zeros in prompts

      lm_cache = {"cache": lm_cache}

      if not return_dict:
        outputs = (lm_logits,) + lm_cache["cache"]
      else:
        outputs = (FlaxCausalLMOutput(logits=lm_logits, hidden_states=None, attentions=None), lm_cache)
    else:
      output = self.module.apply(
        inputs,
        jnp.array(input_ids, dtype="i4"),
        jnp.array(attention_mask, dtype="i4"),
        jnp.array(position_ids, dtype="i4"),
        not train,
        False,
        output_attentions,
        output_hidden_states,
        return_dict,
        rngs=rngs,
        mutable=mutable,
      )
      lm_logits = output.logits
      if input_ids.shape[1] > 1:
        lm_logits = lm_logits[:, 1:, :] # Ignore leading zeros in prompts

      if not return_dict:
        outputs = (lm_logits,) + output[1:]
      else:
        outputs = FlaxCausalLMOutput(logits=lm_logits, hidden_states=output.hidden_states, attentions=output.attentions)

    # add updated cache to model output
    if past_key_values is not None and return_dict:
      outputs, past_key_values = outputs
      outputs["past_key_values"] = unfreeze(past_key_values["cache"])
      return outputs
    elif past_key_values is not None and not return_dict:
      outputs, past_key_values = outputs
      outputs = outputs[:1] + (unfreeze(past_key_values["cache"]),) + outputs[1:]

    return outputs


class FlaxTransformerLMModule(nn.Module):
  config: TransformerConfig

  def setup(self):
    config = self.config
    self.output_embed = nn.Embed(
      num_embeddings=config.output_vocab_size,
      features=config.emb_dim,
      embedding_init=nn.initializers.normal(stddev=1.0),
      name='Embed_0'
    )
    self.pos_embed = AddPositionEmbs(config=config, decode=config.decode, name='posembed_output')
    self.dropout = nn.Dropout(rate=config.dropout_rate)
    self.h_layers = [EncoderDecoder1DBlock(config=config, name=f'encoderdecoderblock_{i}')
                     for i in range(config.num_layers)]
    self.ln_f = nn.LayerNorm(dtype=config.dtype, name='encoderdecoder_norm')

  @nn.compact
  def __call__(
      self,
      input_ids,
      attention_mask,
      position_ids,
      deterministic=True,
      init_cache: bool = False,
      output_attentions: bool = False,
      output_hidden_states: bool = False,
      return_dict: bool = True,
  ):
    config = self.config

    y = input_ids.astype('int32')

    y = self.output_embed(y)
    y = self.pos_embed(y, inputs_positions=position_ids)
    y = self.dropout(y, deterministic=config.deterministic)
    y = y.astype(config.dtype)

    for h in self.h_layers:
      y = h(y, decoder_mask=attention_mask, encoder_decoder_mask=None)

    outputs = (y, None, None)

    hidden_states = outputs[0]
    hidden_states = self.ln_f(hidden_states)

    if output_hidden_states:
      all_hidden_states = outputs[1] + (hidden_states,)
      outputs = (hidden_states, all_hidden_states) + outputs[2:]
    else:
      outputs = (hidden_states,) + outputs[1:]

    if not return_dict:
      return tuple(v for v in outputs if v is not None)

    return FlaxBaseModelOutput(
      last_hidden_state=hidden_states,
      hidden_states=outputs[1],
      attentions=outputs[-1],
    )


class FlaxTransformerLMModel(FlaxTransformerLMPreTrainedModel):
  module_class = FlaxTransformerLMModule


class FlaxTransformerLMForCausalLMModule(nn.Module):
  config: TransformerLMConfig
  dtype: jnp.dtype = jnp.bfloat16
  kernel_init: Callable = nn.initializers.xavier_uniform()
  bias_init: Callable = nn.initializers.normal(stddev=1e-6)
  posemb_init: Callable = None
  use_cache = False

  def convert_config(self, cfg: TransformerLMConfig):
    return TransformerConfig(
      vocab_size=cfg.vocab_size,
      output_vocab_size=cfg.vocab_size,
      logits_via_embedding=cfg.logits_via_embedding,
      dtype=self.dtype,
      emb_dim=cfg.emb_dim,
      num_heads=cfg.num_heads,
      num_layers=cfg.num_layers,
      qkv_dim=cfg.qkv_dim,
      mlp_dim=cfg.mlp_dim,
      max_len=cfg.max_len,
      dropout_rate=cfg.dropout_rate,
      attention_dropout_rate=cfg.attention_dropout_rate,
      deterministic=cfg.deterministic,
      decode=cfg.decode and self.use_cache,
      kernel_init=self.kernel_init,
      bias_init=self.bias_init,
      posemb_init=self.posemb_init,
    )

  def setup(self):
    config_ext = self.convert_config(self.config)
    self.transformer = FlaxTransformerLMModule(config_ext, name='decoder')
    self.lm_head = nn.Dense(
      self.config.output_vocab_size,
      dtype=self.dtype,
      kernel_init=self.kernel_init,
      bias_init=self.bias_init,
      name='logitdense',
    )

  @nn.compact
  def __call__(
      self,
      input_ids,
      attention_mask,
      position_ids,
      deterministic: bool = True,
      init_cache: bool = False,
      output_attentions: bool = False,
      output_hidden_states: bool = False,
      return_dict: bool = True,
  ):
    decoder_mask = None
    inputs_positions = None

    outputs = self.transformer(
      input_ids,
      decoder_mask,
      inputs_positions,
      deterministic=deterministic,
      init_cache=init_cache,
      output_attentions=output_attentions,
      output_hidden_states=output_hidden_states,
      return_dict=return_dict,
    )
    hidden_states = outputs[0]
    lm_logits = self.lm_head(hidden_states)

    if not return_dict:
      return (lm_logits,) + outputs[1:]

    return FlaxCausalLMOutput(logits=lm_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)


class FlaxTransformerLMForCausalLM(FlaxTransformerLMPreTrainedModel):
  module_class = FlaxTransformerLMForCausalLMModule

  def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[jax.Array] = None):

    self.module_class.use_cache = True

    # initializing the cache
    batch_size, seq_length = input_ids.shape

    past_key_values = self.init_cache(batch_size, max_length)
    # Note that usually one would have to put 0's in the attention_mask for x > input_ids.shape[-1] and x < cache_length.
    # But since GPTNeo uses a causal mask, those positions are masked anyways.
    # Thus we can create a single static attention_mask here, which is more efficient for compilation
    extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
    if attention_mask is not None:
      position_ids = attention_mask.cumsum(axis=-1) - 1
      extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))
    else:
      position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length))

    return {
      "past_key_values": past_key_values,
      "attention_mask": extended_attention_mask,
      "position_ids": position_ids,
    }

  def update_inputs_for_generation(self, model_outputs, model_kwargs):
    model_kwargs["past_key_values"] = model_outputs.past_key_values
    model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
    return model_kwargs
