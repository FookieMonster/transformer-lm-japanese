# transformer-lm-japanese

<h4 align="center">
    <p>
        <b>English</b> |
        <a href="https://github.com/FookieMonster/transformer-lm-japanese/blob/main/README_ja.md">日本語</a>
    <p>
</h4>

This is a JAX/Flax-based transformer language model trained on a Japanese dataset. It is based on the official Flax example code (lm1b).

In the official example code of Flax, there exists [lm1b](https://github.com/google/flax/tree/main/examples/lm1b), a transformer decoder type language model. The original example code is trained with the English dataset called [the One Billion Word Benchmark](https://arxiv.org/abs/1312.3005), but this repository modifies the code to train a language model using a Japanese dataset.

This repository includes the code for training a language model using a Japanese dataset, and its configuration files. It also includes code to generate text from the trained weights. You can download the pre-trained weights (checkpoints) from Hugging Face's [model hub](https://huggingface.co/fukugawa).

---
### Model Overview

#### Training Environment

| Model | Hardware | Code | Config | Dataset | Note |
|-|-|-|-|-|-|
| lm1b-default | TPU v3-8 | 1.0.0.RC1 | lm1b_default | lm1b | Reproduction of the original |
| transformer-lm-japanese-default | TPU v3-8 | 1.0.0.RC1 | japanese_default_v1 | cc100/ja | 6 layers |
| [transformer-lm-japanese-0.1b](https://huggingface.co/fukugawa/transformer-lm-japanese-0.1b) | TPU v3-8 | 1.0.0.RC1 | japanese_0.1b_v1 | wiki40b/ja | 12 layers, referring to GPT-2 small |

#### Training Results

| Model | Params | Layers | Dim | Heads | Loss | PPL | Training time |
|-|-|-|-|-|-|-|-|
| lm1b-default | 0.05B | 6 | 512 | 8 | 3.121 | 22.67 | 0.5 days |
| transformer-lm-japanese-default | 0.05B | 6 | 512 | 8 | 4.195 | 66.38 | 0.5 days |
| [transformer-lm-japanese-0.1b](https://huggingface.co/fukugawa/transformer-lm-japanese-0.1b) | 0.1B | 12 | 768 | 12 | 3.562 | 35.22 | 1.5 days |

#### TensorBoard

<img src="/images/tensorboard-2.png" width="860">

#### Tokenizer

The tokenizer uses SentencePiece, the same as the original, for subword learning. However, to easily add options specific to Japanese without modifying the source code, the following configuration item (config.spm_train_options) has been added so that they can be added simply from the configuration file.

```
config.spm_train_options = "--character_coverage=0.9995 --byte_fallback=true"
```

#### Dataset

The dataset uses data from TensorFlow Datasets, the same as the original.
The Japanese dataset currently supports the following two types:

* wiki40b/ja
* huggingface:cc100/lang=ja

For example, you can use cc100(ja) by including it in the configuration file as follows.  
(Since cc100 only has a 'train' split, it is set up by dividing it into sub-splits.)

```
config.dataset_name = "huggingface:cc100/lang=ja"
config.train_split = "train[:98%]"
config.eval_dataset_name = "huggingface:cc100/lang=ja"
config.eval_split = "train[98%:]"
```

#### Preprocessing

The dataset_preprocessor.py performs on-the-fly preprocessing of the Japanese dataset.
If you want to support Japanese datasets other than the above two types, you can easily adapt by adding preprocessor code for preprocessing to the dataset_preprocessor.py.

```python
class DatasetPreprocessor:
  def __init__(self):
    self.preprocessors = {
      'lm1b': Lm1bPreprocessor(),
      'wiki40b/ja': Wiki40bJaPreprocessor(),
      'huggingface:cc100/lang=ja': Cc100JaPreprocessor(),
      # Add more datasets as needed
    }
```
---

### How to Train on Cloud TPU

Creating a Cloud TPU VM with gcloud

```
ZONE=us-central1-a
TPU_TYPE=v3-8
TPU_NAME=my-tpu-vm
TPU_SOFTWARE_VERSION=v2-alpha

gcloud compute tpus tpu-vm create $TPU_NAME \
    --zone $ZONE \
    --accelerator-type $TPU_TYPE \
    --version $TPU_SOFTWARE_VERSION
```

Connecting to your Cloud TPU VM

```
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone $ZONE
```

It is found that the TPU VM is actually a VM instance of Ubuntu 20.04.

```
Welcome to Ubuntu 20.04.2 LTS (GNU/Linux 5.4.0-1043-gcp x86_64)
```

Clone the source code of this repository and install the necessary Python packages.

```
git clone -b 1.0.0.RC2 https://github.com/FookieMonster/transformer-lm-japanese
cd ./transformer-lm-japanese/transformer_lm
pip install -r requirements.txt
pip install "jax[tpu]==0.3.2" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

Run the Python interpreter and pre-download the necessary datasets.

```
python3
```

```
>>> import tensorflow_datasets as tfds
>>> tfds.load('lm1b')
>>> tfds.load('wiki40b/ja')
```
The cc100 dataset requires several hundred GB of disk space. If you want to train with the cc100 dataset, please create the dataset on Google Cloud Storage following the procedures in [this](https://github.com/FookieMonster/transformer-lm-japanese/blob/main/docs/train_with_gcs.md) document.

Start training by specifying the working directory, where TensorBoard event logs and checkpoints are saved, and the configuration file.

```
python3 main.py --workdir=$HOME/logs/japanese_0.1b_v1 --config=configs/japanese_0.1b_v1.py
```

---
### How to Train on Cloud TPU (Google Cloud Storage)

For the training procedure when setting the working directory and dataset directory to GCS, refer to [this](https://github.com/FookieMonster/transformer-lm-japanese/blob/main/docs/train_with_gcs.md).

---
### Text Generation

You can generate text by specifying a working directory where the trained weights (checkpoints) are saved and the configuration file.

```
python3 generate_text.py --workdir=$HOME/logs/japanese_0.1b_v1 \
    --config=configs/japanese_0.1b_v1.py \
    --config.sampling_temperature=0.6 \
    --config.sampling_top_k=20 \
    --config.seed=0 \
    --config.prompts="夏目漱石は、" \
    --num_generated_texts=10
```

```
I0711 07:22:35.803195 140565925375040 train.py:323] Generating text.
I0711 07:22:46.564662 140565925375040 train.py:344] Sample: 夏目漱石は、自分の作品を「文学の本」として出版することを構想していた。
I0711 07:22:46.564912 140565925375040 train.py:323] Generating text.
I0711 07:22:47.605942 140565925375040 train.py:344] Sample: 夏目漱石は、明治の文学運動を「文学の原点に立ち帰る」と位置づけ、漱石が「文学の本質をあらわすのが文学である」との認識を、当時の知識人たちが持っていたことを指摘している。
I0711 07:22:47.606220 140565925375040 train.py:323] Generating text.
I0711 07:22:47.989197 140565925375040 train.py:344] Sample: 夏目漱石は、小説『坊っちゃん』で、この「坊っちゃん」を「坊っちゃん」に置き換えた。「坊っちゃん」は、坊っちゃんの「坊」の字を、「坊」は「坊」の字をもじってつけられた。
I0711 07:22:47.989488 140565925375040 train.py:323] Generating text.
I0711 07:22:48.145813 140565925375040 train.py:344] Sample: 夏目漱石は、漱石の『坊っちゃん』を読んで、「漱石は、私に『坊っちゃん』をおもしろおかしく書かせた。これは、私に『坊っちゃん』を書かせるのを、私に教えてくれたからだ」と述懐している。
I0711 07:22:48.146037 140565925375040 train.py:323] Generating text.
I0711 07:22:48.355056 140565925375040 train.py:344] Sample: 夏目漱石は、自身の著作『漱石全集』の中で「漱石が生涯のほとんどを漱石の文学に捧げた」と評価している。
I0711 07:22:48.355254 140565925375040 train.py:323] Generating text.
I0711 07:22:48.524219 140565925375040 train.py:344] Sample: 夏目漱石は、漱石が「『吾輩は猫』を観るのが嫌だ」と言ったのを、漱石が「あんなに怖いとは思わなかった」と返している。
I0711 07:22:48.524435 140565925375040 train.py:323] Generating text.
I0711 07:22:48.758128 140565925375040 train.py:344] Sample: 夏目漱石は、自身の日記の中で「文学の本質と現実との間には、対立関係があり、また対立関係があっても、それが文学の本質と現実との間には関係がある」と書いている。
I0711 07:22:48.758367 140565925375040 train.py:323] Generating text.
I0711 07:22:49.008113 140565925375040 train.py:344] Sample: 夏目漱石は、夏目が漱石の『吾輩は猫である』を読んでいた時に、漱石の『吾輩は猫である』を読んだという。漱石は「猫は猫である」と書いていたが、漱石は「猫である」と書いた。
I0711 07:22:49.008389 140565925375040 train.py:323] Generating text.
I0711 07:22:50.048584 140565925375040 train.py:344] Sample: 夏目漱石は、小説『坊っちゃん』の中で、主人公が「おばあさん」と「おばあさん」の2人で暮らしていると、その家から「おばあさん」と「おばあさん」が飛び出してくるという話を紹介している。
I0711 07:22:50.048887 140565925375040 train.py:323] Generating text.
I0711 07:22:50.240439 140565925375040 train.py:344] Sample: 夏目漱石は、漱石の「吾輩は猫である」という言葉を、漱石が「猫を飼っている人は猫である」という誤解から誤解したのだろうと、著書『猫の散歩道』で述べている。
```
