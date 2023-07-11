# transformer-lm-japanese

<h4 align="center">
    <p>
        <a href="https://github.com/FookieMonster/transformer-lm-japanese">English</a> |
        <b>日本語</b>
    <p>
</h4>

日本語のデータセットで学習するJAX/Flaxベースのトランスフォーマー言語モデルです。Flax公式のサンプルコード（lm1b）をベースにしています。

Flaxの公式サンプルコードには、トランスフォーマーのデコーダー型の言語モデルである[lm1b](https://github.com/google/flax/tree/main/examples/lm1b)が存在します。その元々のサンプルコードは英文データセットの[One Billion Word Benchmark](https://arxiv.org/abs/1312.3005)で学習が行われていますが、本リポジトリではそのコードを一部修正し、日本語データセットを用いて言語モデルの学習が可能となっています。

このリポジトリには、日本語データセットを使用して言語モデルを訓練するためのコードと、その設定ファイルが含まれています。

学習済みの重みからテキストを生成するコードも含まれています。学習済みの重みはHugging Faceのモデルハブからダウンロードできるようにする予定です。

---
### モデルの概要

#### トレーニング環境

| Model | Hardware | Code | Config | Dataset | Note |
|-|-|-|-|-|-|
| lm1b-default | TPU v3-8 | 1.0.0.RC1 | lm1b_default | lm1b | オリジナルの再現 |
| transformer-lm-japanese-default | TPU v3-8 | 1.0.0.RC1 | japanese_default_v1 | cc100/ja | オリジナルと同じ6層 |
| transformer-lm-japanese-0.1b | TPU v3-8 | 1.0.0.RC1 | japanese_0.1b_v1 | wiki40b/ja | GPT-2 samllを参考に12層 |

#### トレーニング結果

| Model | Params | Layers | Dim | Heads | Loss | PPL | Training time |
|-|-|-|-|-|-|-|-|
| lm1b-default | 0.05B | 6 | 512 | 8 | 3.121 | 22.67 | 0.5 days |
| transformer-lm-japanese-default | 0.05B | 6 | 512 | 8 | 4.195 | 66.38 | 0.5 days |
| transformer-lm-japanese-0.1b | 0.1B | 12 | 768 | 12 | 3.562 | 35.22 | 1.5 days |

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
git clone -b 1.0.0.RC2 https://github.com/FookieMonster/transformer-lm-japanese
cd ./transformer-lm-japanese/transformer_lm
pip install -r requirements.txt
pip install "jax[tpu]==0.3.2" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
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

### Cloud TPUによるテキスト生成

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
