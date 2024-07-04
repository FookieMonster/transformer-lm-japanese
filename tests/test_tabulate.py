import jax
import jax.numpy as jnp
from flax import linen as nn

from transformer_lm.models import TransformerConfig
from transformer_lm.models import TransformerLM

train_config = TransformerConfig(
  vocab_size=30_000,
  output_vocab_size=30_000,
  logits_via_embedding=False,
  dtype=jnp.bfloat16,
  emb_dim=2048,
  num_heads=16,
  num_layers=24,
  qkv_dim=2048,
  mlp_dim=8192,
  max_len=max(256, 512),
  dropout_rate=0.1,
  attention_dropout_rate=0.1,
  deterministic=False,
  decode=False,
  kernel_init=nn.initializers.xavier_uniform(),
  bias_init=nn.initializers.normal(stddev=1e-6))

model = TransformerLM(train_config)
per_device_batch_size = 32
max_target_length = 256
input_shape = (per_device_batch_size, max_target_length)
input = jnp.ones(input_shape, jnp.float32)
init_rngs = {'params': jax.random.PRNGKey(0), 'dropout': jax.random.PRNGKey(1)}
print(model.tabulate(init_rngs, input))
