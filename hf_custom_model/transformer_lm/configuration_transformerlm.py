from transformers import PretrainedConfig


class TransformerLMConfig(PretrainedConfig):
  model_type = "transformerlm"

  def __init__(
      self,
      vocab_size: int = 30000,
      output_vocab_size: int = 30000,
      share_embeddings: bool = False,
      logits_via_embedding: bool = False,
      emb_dim: int = 512,
      num_heads: int = 8,
      num_layers: int = 6,
      qkv_dim: int = 512,
      mlp_dim: int = 2048,
      max_len: int = 2048,
      dropout_rate: float = 0.1,
      attention_dropout_rate: float = 0.1,
      deterministic: bool = False,
      decode: bool = False,
      bos_token_id=50256,
      eos_token_id=50256,
      **kwargs,
  ):
    self.vocab_size = vocab_size
    self.output_vocab_size = output_vocab_size
    self.share_embeddings = share_embeddings
    self.logits_via_embedding = logits_via_embedding
    self.emb_dim = emb_dim
    self.num_heads = num_heads
    self.num_layers = num_layers
    self.qkv_dim = qkv_dim
    self.mlp_dim = mlp_dim
    self.max_len = max_len
    self.dropout_rate = dropout_rate
    self.attention_dropout_rate = attention_dropout_rate
    self.deterministic = deterministic
    self.decode = decode
    self.bos_token_id = bos_token_id
    self.eos_token_id = eos_token_id
    super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
