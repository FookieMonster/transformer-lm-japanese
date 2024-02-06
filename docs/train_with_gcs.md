### Cloud TPUによるトレーニング手順（Googleクラウドストレージ版）

ここでは、transformer-lm-japaneseの言語モデルをGoogleクラウドストレージ（GCS）上のデータセットとワークディレクトリを利用してTPUで学習させる手順を説明します。コスト削減のためにプリエンプティブルなTPU-VMを利用します。

プリエンプティブルなTPU-VMはいつ停止されるか分からず、一度PREEMPTED状態になるとデータに一切アクセスできず、VMを削除することしかできません。
なので、ワークディレクトリやデータセットディレクトリはGCS上に作成する必要があります。

また、TPU-VMのディスクサイズは100GB程しかなく、cc100/jaなどの大規模データセットをダウンロードすることが困難です。その場合でも、GCS上のデータセットを使うことでトレーニングを行うことが可能になります。

---
#### GCPプロジェクトの作成とCLIのインストール

まず、Google Cloud Platform（GCP）で新しくプロジェクトを作成し、[gcloud CLI をインストールする](https://cloud.google.com/sdk/docs/install?hl=ja)のドキュメントに従ってgcloudコマンドが使える状態にして下さい。次に、gcloudコマンドが正常にインストール＆セットアップできたことを以下のコマンドで確認します。
新しく作成したプロジェクト名がデフォルト表示されていればセットアップは完了です。

```
(local)$ gcloud config list
```

---
#### Cloud TPU APIの有効化

Google Cloud ConsoleのWeb画面から、[Compute Engine] - [TPU]を選択しCloud TPU APIを有効化します。

<img src="/images/tpu-api.png" width="860">

---
#### ワークディレクトリとデータセット用のGCSバケットを作成

先程作成したGCPプロジェクトと同一プロジェクト内のGoogleクラウドストレージ（GCS）上に２つのバケットを作成します。  
ここでは以下の名前で作成することとします。

* my-lm-work　（ワークディレクトリ用でチェックポイントやTensorBoardログが保存される）
* my-tfds-data　（TensorFlow Datasetsのデータ保存用）

**バケット名はグローバルで一意にする必要があるため、同じ名前は使えませんので適宜変更して下さい。  
以後の説明では該当箇所をご自身のバケット名で読み替えて下さい。**

<img src="/images/gcs-backet.png" width="860">

---
#### APIキーの作成とダウンロード

PythonコードからGCSバケットにAPIアクセスするために必要なAPIキーファイルを作成しダウンロードします。

Google Cloud ConsoleのWeb画面から、[IAMと管理]-[サービスアカウント]を選択します。
そのリストに「Compute Engine default service account」という名前のアカウントがデフォルトで作成されているはずです。
そのリンクを選択し、[キー]-[鍵を追加]-[新しい鍵を作成]-[キーのタイプ JSON]を選択肢キーファイルをダウンロードします。

<img src="/images/api-key.png" width="860">

---
#### データセットの事前ダウンロード

データセットの初回ダウンロードにはとても時間がかかります。
lm1bやwiki40b/jaは数時間で完了しますが、cc100/jaは数十時間かかります。
TPU-VM上でダウンロードを行うとコストがかかるので、別のPython3.8環境で事前ダウンロード（GCSのバケットにアップロード）を行います。

また、cc100/jaはデータセットのサイズが74GB（temporaryが数百GB）になるので、一時的なディスクの空き容量が最低でも数百GB必要です。
私の場合は、以下のようなGCE上のCPU-VMを使ってPython3.8環境を構築しました。

```
名前： my-cpu-vm
マシンタイプ: c2-standard-4 CPUx4 メモリ16GB
ゾーン： us-central1-a
ディスク: 2TB（標準永続ディスク）
OS: Ubuntu 20.04 LTS x86/64
```

CPU-VMにSSHでアクセスします。

```
(local)$ gcloud compute ssh my-cpu-vm --zone=us-central1-a
```

まず、Python3.8とpipをインストールします。

```
(cpu-vm)$ sudo apt-get update
(cpu-vm)$ sudo apt-get install python3.8 python3-pip build-essential
```

次に、tensorflow-datasetsとその関連パッケージをインストールします。  
（datasetsはHuggingface DatasetsのデータセットをTensorFlow Datasetsから使うために必要です）

```
(cpu-vm)$ pip install tensorflow==2.11.1
(cpu-vm)$ pip install tensorflow-datasets==4.8.3
(cpu-vm)$ pip install datasets==2.12.0
```

次に、GCSバケットにAPIアクセスするために、APIキーファイルをアップロードしそのパスを環境変数に設定します。  

例） カレントディレクトリにあるAPIキーファイルを、my-cpu-vm側の/tmp/service-account-api-key.jsonにコピー
```
(local)$ gcloud compute scp ./service-account-api-key.json my-cpu-vm:/tmp/service-account-api-key.json --zone=us-central1-a
```

```
(cpu-vm)$ export GOOGLE_APPLICATION_CREDENTIALS="/tmp/service-account-api-key.json"
```

次に、Pythonインタプリターを起動して、データセットを順番にダウンロードします。  
**（data_dirにはTensorFlow Datasetsのデータ保存用のGCSバケット名を指定して下さい）**

```
(cpu-vm)$ python3
```

```
>>> import tensorflow_datasets as tfds
>>> tfds.load('lm1b', data_dir="gs://my-tfds-data")
>>> tfds.load('wiki40b/ja', data_dir="gs://my-tfds-data")
>>> tfds.load('huggingface:cc100/lang=ja', data_dir="gs://my-tfds-data")
```

データセットの事前ダウンロードが完了すると、以下のようにGCSバケット内に各データセット用のフォルダが作成されているはずです。  

<img src="/images/gcs-tfds.png" width="860">

---
#### TPU-VMインスタンスの作成

Google Cloud ConsoleのWeb画面から、[Compute Engine]-[TPU]-[TPUノードを作成]

1. 名前：my-tpu-vm
2. ゾーン：uscentral1-a
3. TPU VM アーキテクチャ（推奨）
4. TPUタイプ：v3-8
5. TPUソフトウェアバージョン：v2-alpha
6. このTPUノードの**プリエンプティブをオン**にする

上記のように設定し作成ボタンを押す

#### TPU-VMにSSHアクセス

```
(local)$ gcloud compute tpus tpu-vm ssh my-tpu-vm --zone=us-central1-a
```

#### ソースコードのクローンとPythonパッケージのインストール

```
(tpu-vm)$ git clone -b 1.0.0.RC3 https://github.com/FookieMonster/transformer-lm-japanese
(tpu-vm)$ cd ./transformer-lm-japanese/transformer_lm
(tpu-vm)$ pip install -r requirements.txt
(tpu-vm)$ pip install "jax[tpu]==0.4.13" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

#### APIキーのアップロード

APIキーをローカルPC側からTPU-VMインスタンス側へコピーする必要があります。  
gcloud CLIのscpコマンドでコピーします。scpコマンドの書式は以下のとおりです。  
（CPU-VMとTPU-VMでは書式が異なります）

```
gcloud compute tpus tpu-vm scp [ローカルファイルのパス] [TPU-VMインスタンス名]:[リモートファイルのパス] --zone=[ゾーン]
```

例） カレントディレクトリにあるAPIキーファイルを、my-tpu-vm側の/tmp/service-account-api-key.jsonにコピー

```
(local)$ gcloud compute tpus tpu-vm scp ./service-account-api-key.json my-tpu-vm:/tmp/service-account-api-key.json --zone=us-central1-a
```

#### トレーニングの開始

GCSバケットにAPIアクセスするために必要なキーファイルを環境変数にセットします。

```
(tpu-vm)$ export GOOGLE_APPLICATION_CREDENTIALS="/tmp/service-account-api-key.json"
```

TensorFlow DatasetsのデータディレクトリをGCSのバケットに設定します。  

```
(tpu-vm)$ export TFDS_DATA_DIR=gs://my-tfds-data
```

GCS上のワークディレクトリを指定してトレーニングを開始します。

```
(tpu-vm)$ python3 main.py --workdir=gs://my-lm-work/japanese_0.1b_v1 --config=configs/japanese_0.1b_v1.py
```

以上で、  
GCS上のデータセットを使ってトレーニングを行い、GCS上のワークディレクトリにTensorBoardログやチェックポイントが保存されるようになります。  

---
### テキスト生成（Googleクラウドストレージ版）

学習済みの重み（チェックポイント）が保存されているワークディレクトリと設定ファイルを指定してテキストを生成することが可能です。

```
(tpu-vm)$ python3 generate_text.py --workdir=gs://my-lm-work/japanese_0.1b_v1 \
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
