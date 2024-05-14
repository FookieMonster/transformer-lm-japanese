# transformer-lm-japanese

<h4 align="center">
    <p>
        <b>日本語</b> |
        <a href="https://github.com/FookieMonster/transformer-lm-japanese/blob/main/README_en.md">English</a>
    <p>
</h4>

日本語のデータセットで学習するJAX/FlaxベースのTransformer言語モデルです。Flax公式のサンプルコード（lm1b）をベースにしています。

Flaxの公式サンプルコードには、Transformerのデコーダー型の言語モデルである[lm1b](https://github.com/google/flax/tree/main/examples/lm1b)が存在します。その元々のサンプルコードは英文データセットの[One Billion Word Benchmark](https://arxiv.org/abs/1312.3005)で学習が行われていますが、本リポジトリではそのコードを一部修正し、日本語データセットを用いて言語モデルの学習が可能となっています。

このリポジトリには、日本語データセットを使用して言語モデルを訓練するためのコードと、その設定ファイルが含まれています。学習済みの重みからテキストを生成するコードも含まれています。学習済みの重み（チェックポイント）は、[Hugging Faceのモデルハブ](https://huggingface.co/fukugawa)からダウンロードすることもできます。

その他の詳細な情報は、こちらの[技術ブログ](https://zenn.dev/fukugawa/articles/4446573ec0f697)でも公開しています。

---
### 更新履歴

* 2024/05/13 FlaxAutoModelForCausalLMに対応したカスタムモデルコード（hf_custom_model）を追加しました。

---
### モデルの概要

#### トレーニング環境

| Model | Hardware | Code | Config | Dataset | Note |
|-|-|-|-|-|-|
| lm1b-default | TPU v3-8 | 1.0.0.RC1 | lm1b_default | lm1b | オリジナルの再現 |
| transformer-lm-japanese-default | TPU v3-8 | 1.0.0.RC1 | japanese_default_v1 | cc100/ja | オリジナルと同じ6層 |
| [transformer-lm-japanese-0.1b](https://huggingface.co/fukugawa/transformer-lm-japanese-0.1b) | TPU v3-8 | 1.0.0.RC1 | japanese_0.1b_v1 | wiki40b/ja | GPT-2 samllを参考に12層 |

#### トレーニング結果

| Model | Params | Layers | Dim | Heads | Loss | PPL | Training time |
|-|-|-|-|-|-|-|-|
| lm1b-default | 0.05B | 6 | 512 | 8 | 3.121 | 22.67 | 0.5 days |
| transformer-lm-japanese-default | 0.05B | 6 | 512 | 8 | 4.195 | 66.38 | 0.5 days |
| [transformer-lm-japanese-0.1b](https://huggingface.co/fukugawa/transformer-lm-japanese-0.1b) | 0.1B | 12 | 768 | 12 | 3.562 | 35.22 | 1.5 days |

#### TensorBoard

<img src="/images/tensorboard-2.png" width="860">

#### トークンナイザー

トークンナイザーはオリジナルと同じくSentencePieceを使ってサブワードの学習を行っていますが、日本語特有なオプションをソースコードを修正することなく、設定ファイルから簡単に追加できるように以下の設定項目（config.spm_train_options）を追加しています。

```
config.spm_train_options = "--character_coverage=0.9995 --byte_fallback=true"
```

#### データセット

データセットはオリジナルと同じくTensorFlow Datasetsのデータを使っています。  
日本語データセットは、現在のところ以下の２種類に対応しています。

* wiki40b/ja
* huggingface:cc100/lang=ja

例えばcc100(ja)は、以下のように設定ファイルに記載することで利用可能です。  
（cc100はスプリットがtrainしかありませんのでサブスプリット分割して設定しています）

```
config.dataset_name = "huggingface:cc100/lang=ja"
config.train_split = "train[:98%]"
config.eval_dataset_name = "huggingface:cc100/lang=ja"
config.eval_split = "train[98%:]"
```

#### 前処理

dataset_preprocessor.pyで日本語データセットの前処理をオンザフライに行います。  
上記２種類以外の日本語データセットに対応したい場合、dataset_preprocessor.pyに前処理用のプリプロセッサコードを追加することで簡単に適応できます。

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

### Cloud TPUによるトレーニング手順

TPU-VMの作成

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

TPU-VMにSSHでアクセス

```
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone $ZONE
```

TPU-VMは実際にはUbuntu 20.04のVMインスタンスです。

```
Welcome to Ubuntu 20.04.2 LTS (GNU/Linux 5.4.0-1043-gcp x86_64)
```

このリポジトリのソースコードをクローンし、必要なPythonパッケージをインストールします。

```
git clone -b 1.0.0.RC3 https://github.com/FookieMonster/transformer-lm-japanese
cd ./transformer-lm-japanese/transformer_lm
pip install -r requirements.txt
pip install "jax[tpu]==0.4.13" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

Pythonインタプリターを起動して、必要なデータセットを事前にダウンロードします。

```
python3
```

```
>>> import tensorflow_datasets as tfds
>>> tfds.load('lm1b')
>>> tfds.load('wiki40b/ja')
```

（cc100は数百GBのサイズがあります。cc100でトレーニングしたい場合、[こちら](https://github.com/FookieMonster/transformer-lm-japanese/blob/main/docs/train_with_gcs.md)の手順でGCS上にデータセットを作成して下さい）

TensorBoardのイベントログやチェックポイントが保存されるワークディレクトリと、  
設定ファイルを指定してトレーニングを開始します。

```
python3 main.py --workdir=$HOME/logs/japanese_0.1b_v1 --config=configs/japanese_0.1b_v1.py
```

---
### Cloud TPUによるトレーニング手順（Googleクラウドストレージ版）

ワークディレクトリやデータセットのディレクトリをGCSにする場合のトレーニング手順は[こちら](https://github.com/FookieMonster/transformer-lm-japanese/blob/main/docs/train_with_gcs.md)を参照

---
### テキスト生成

学習済みの重み（チェックポイント）が保存されているワークディレクトリと設定ファイルを指定してテキストを生成することが可能です。

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

---
### テキスト生成 (HuggingFaceから重みをダウンロード)

GCP上の以下のようなCPUインスタンスで、Python 3.10環境を構築してテキスト生成する手順です。

* マシンタイプ: c2-standard-4 (4 CPUs, 16GB Memory)
* ディスク: 100GB (標準永続ディスク)
* OS: Ubuntu 22.04 LTS x86/64

まず、Python 3.10とpipをインストールします。

```
sudo apt-get update
sudo apt-get install python3.10 python3-pip build-essential
```

huggingface_hubをインストールします。

```
pip install --upgrade huggingface_hub
```

ホームディレクトリに移動して、Pythonインタープリターを起動。

```
cd $HOME
python3
```

重みとトークンナイザをダウンロードします。

```python
>>> from huggingface_hub import hf_hub_download
>>> hf_hub_download(repo_id="fukugawa/transformer-lm-japanese-0.1b", filename="sentencepiece_model", revision="v1", local_dir="./logs/japanese_0.1b_v1")
>>> hf_hub_download(repo_id="fukugawa/transformer-lm-japanese-0.1b", filename="checkpoint_499999", revision="v1", local_dir="./logs/japanese_0.1b_v1")
```

ソースコードをクローンして、必要なPythonパッケージをインストールします。

```
git clone -b 1.0.0.RC3 https://github.com/FookieMonster/transformer-lm-japanese
cd ./transformer-lm-japanese/transformer_lm
pip install -r requirements.txt
```

CPUで実行するために必要なパッケージもインストールします。

```
pip install jax[cpu]==0.4.13
pip install protobuf==3.20.3
```

学習済みの重み（チェックポイント）が保存されているワークディレクトリと設定ファイルを指定してテキストを生成することが可能です。

```
python3 generate_text.py --workdir=$HOME/logs/japanese_0.1b_v1 \
    --config=configs/japanese_0.1b_v1.py \
    --config.sampling_temperature=0.6 \
    --config.sampling_top_k=20 \
    --config.seed=0 \
    --config.prompts="夏目漱石は、" \
    --num_generated_texts=10
```
