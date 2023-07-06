### Cloud TPUによるトレーニング手順（Googleクラウドストレージ版）

ここでは、transformer-lm-japaneseのモデルをGoogleクラウドストレージ（GCS）上のデータセットとワークディレクトリを利用してTPUで学習させる手順を説明します。コスト削減のためにプリエンプティブルなTPU-VMを利用します。

プリエンプティブルなTPU-VMはいつ停止されるか分からず、一度PREEMPTED状態になるとVMインスタンスのデータに一切アクセスできず、VMを削除することしかできません。
なので、ワークディレクトリやデータセットディレクトリはGCS上に作成する必要があります。

---
#### GCPプロジェクトの作成とCLIのインストール

まず、Google Cloud Platform（GCP）で新しくプロジェクトを作成し、[gcloud CLI をインストールする](https://cloud.google.com/sdk/docs/install?hl=ja)のドキュメントに従ってgcloudコマンドが使える状態にして下さい。次に、gcloudコマンドが正常にインストール＆セットアップできたことを以下のコマンドで確認します。
projectに（新しく作成したプロジェクト名）が入っていればセットアップは完了です。

```
% gcloud config list
```

```
[compute]
region = us-central1
zone = us-central1-a
[core]
account = xxxxxxxxxxx@gmail.com
disable_usage_reporting = False
project = （新しく作成したプロジェクト名）

Your active configuration is: [default]
```

---
#### Cloud TPU APIの有効化

Google Cloud ConsoleのWeb画面から、[Compute Engine] - [TPU]を選択しCloud TPU APIを有効化します。

<img src="/images/tpu-api.png" width="860">

---
#### ワークディレクトリとデータセット用のGCSバケットを作成

先程作成したGCPプロジェクトと同一プロジェクト内のGoogleクラウドストレージ（GCS）上に２つのバケットを作成します。  
ここでは以下の名前で作成することとします。バケット名はグローバルで一意にする必要があるため、同じ名前は使えませんので適宜変更して下さい。以後の説明では該当箇所をご自身のバケット名で読み替えて下さい。

* my-lm-work　（ワークディレクトリ用でチェックポイントやTensorBoardログが保存される）
* my-tfds-data　（TensorFlow Datasetsのデータ保存用）

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

また、cc100/jaはデータセットのサイズが82GB（展開後数百GB）になるので、一時的なディスクの空き容量が最低でも数百GB必要です。
私の場合は、以下のGCEインスタンスを借りてPython3.8環境を構築しました。

```
マシンタイプ: e2-standard-2 CPUx2 メモリ8GB
ディスク: 2TB（標準永続ディスク）
OS: Ubuntu 18 LTS
```

事前ダウンロード用のPython3.8環境が準備できましたら、  
まず、tensorflow-datasetsとその関連パッケージをインストールします。

```
pip install tensorflow==2.11.1
pip install tensorflow-datasets==4.8.3
pip install datasets==2.12.0
```

次に、PythonコードからGCSバケットにAPIアクセスするために、先程ダウンロードしたAPIキーファイルのパスを環境変数に設定します。

```
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/dir/service-account-api-key.json"
```

次に、Pythonインタプリターを起動して、以下のPythonコードを順番に実行していきます。  

```
$ python3
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
6. このTPUノードのプリエンプティブをオンにする

上記のように設定し作成ボタンを押す

#### TPU-VMインスタンスにSSHアクセス

```
gcloud compute tpus tpu-vm ssh my-tpu-vm --zone=us-central1-a
```

#### ソースコードのクローンとPythonパッケージのインストール

```
git clone -b 1.0.0.RC1 https://github.com/FookieMonster/transformer-lm-japanese
cd ./transformer-lm-japanese/transformer_lm
pip install -r requirements.txt
pip install "jax[tpu]==0.3.2" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

#### APIキーのアップロード

APIキーをローカルPC側からTPU-VMインスタンス側へコピーする必要があります。  
gcloud CLIのscpコマンドでコピーします。scpコマンドの書式は以下のとおりです。

```
gcloud compute tpus tpu-vm scp [ローカルファイルのパス] [TPU-VMインスタンス名]:[リモートファイルのパス] --zone=[ゾーン]
```

例）カレントディレクトリにあるAPIキーファイルを、my-tpu-vm側の/tmp/service-account-api-key.jsonにコピー

```
gcloud compute tpus tpu-vm scp ./service-account-api-key.json my-tpu-vm:/tmp/service-account-api-key.json --zone=us-central1-a
```

#### トレーニングの開始

GCSバケットにAPIアクセスするために必要なキーファイルを環境変数にセットします。

```
export GOOGLE_APPLICATION_CREDENTIALS="/tmp/service-account-api-key.json"
```

TensorFlow DatasetsのデータディレクトリをGCSのバケットに設定します。  

```
export TFDS_DATA_DIR=gs://my-tfds-data
```

GCS上のワークディレクトリを指定してトレーニングを開始します。

```
python3 main.py --workdir=gs://my-lm-work/japanese_0.1b_v1 --config=configs/japanese_0.1b_v1.py
```

以上で、  
GCS上のデータセットを使ってトレーニングを行い、GCS上のワークディレクトリにTensorBoardログやチェックポイントが保存されるようになります。  
