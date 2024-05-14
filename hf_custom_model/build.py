import os

from flax.training import checkpoints
from huggingface_hub import hf_hub_download

from transformer_lm.configuration_transformerlm import TransformerLMConfig
from transformer_lm.modeling_transformerlm_flax import FlaxTransformerLMModel, FlaxTransformerLMForCausalLM
from transformer_lm.tokenization_transformerlm import TransformerLMTokenizer


REPO_ID = "fukugawa/transformer-lm-japanese-0.1b"
REPO_VER = "v1"
LOCAL_DIR = "./transformer_lm/files"
VOCAB_FILE = "./transformer_lm/files/sentencepiece_model"
CHECKPOINT_FILE = "./transformer_lm/files/checkpoint_499999"
PUSH_REPO_ID = "transformer-lm-japanese-0.1b"

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

TransformerLMConfig.register_for_auto_class()
FlaxTransformerLMModel.register_for_auto_class("AutoModel")
FlaxTransformerLMForCausalLM.register_for_auto_class("FlaxAutoModelForCausalLM")
TransformerLMTokenizer.register_for_auto_class("AutoTokenizer")

config = TransformerLMConfig(
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
  tokenizer_class="TransformerLMTokenizer",
)
model_flax = FlaxTransformerLMForCausalLM(config)

tokenizer = TransformerLMTokenizer(vocab_file=VOCAB_FILE)
tokenizer.push_to_hub(PUSH_REPO_ID)

restored_params = checkpoints.restore_checkpoint(CHECKPOINT_FILE, target=None)


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


model_flax.params = copy_nested_params(model_flax.params, restored_params['params'])
model_flax.params['logitdense'] = restored_params['params']['decoder']['logitdense']
model_flax.save_pretrained(PUSH_REPO_ID, push_to_hub=True)
