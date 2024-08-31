# My LLM Project: カスタムLLM実装の学習ガイド
## プロジェクト概要
このプロジェクトは、GPT（Generative Pre-trained Transformer）アーキテクチャをベースにしたカスタムLLM（Large Language Model）の実装です。OpenWebTextデータセットを使用して学習を行い、テキスト生成タスクを実行します。

## プロジェクト構造
```
my-llm-project/
├── scripts/
│   └── train.py
├── src/
│   ├── data_processing.py
│   ├── model.py
│   ├── training.py
│   └── utils.py
└── pyproject.toml
```

## 環境設定
Python 3.11以上がインストールされていることを確認してください。
Poetryをインストールします（まだの場合）:
```
pip install poetry
```

プロジェクトディレクトリに移動し、依存関係をインストールします:
```
poetry install
```

## 実行方法
### データの準備と前処理
```
poetry run python -c "from src.data_processing import load_and_process_dataset, save_processed_dataset, create_memmap_files; tokenized, _ = load_and_process_dataset(); save_processed_dataset(tokenized); create_memmap_files(tokenized)"
```

### モデルの学習
```
poetry run python scripts/train.py
```

### テキスト生成（学習後）
```
poetry run python -c "import torch; from src.model import GPT; model = GPT(...); model.load_state_dict(torch.load('best_checkpoint.bin')['model']); print(model.generate_sentence('Once upon a time', 1024, 100, tokenizer, 'cuda'))"
```

## 主要コンポーネントの解説
*scripts/train.py*
学習プロセス全体を制御するスクリプトです。主な機能は以下の通りです：
- 設定パラメータの定義
- データセットの準備
- モデルの初期化
- オプティマイザとスケーラーの設定
- 学習の実行

*src/data_processing.py*
データの準備と処理を行うモジュールです。主な機能は以下の通りです：
- OpenWebTextデータセットのロードと前処理
- データのトークン化
- 処理済みデータセットの保存と読み込み
- メモリマップファイルの作成
- バッチデータの取得

*src/model.py*
GPTモデルの定義を含むモジュールです。主なクラスは以下の通りです：
- PreLNGPTDecoderLayer: Pre Layer Normalization GPTデコーダ層
- GPT: GPTモデル全体の構造
- GPTクラスには、テキスト生成のためのgenerate()とgenerate_sentence()メソッドが実装されています。

*src/training.py*
モデルの学習と検証を行うモジュールです。主な機能は以下の通りです：
- モデルの学習ループ
- 検証
- チェックポイントの保存
- 学習詳細の保存

*src/utils.py*
ユーティリティ関数を含むモジュールです。現在は学習率のスケジューリングに関する関数のみが含まれています。
### 設定パラメータ
主な設定パラメータはscripts/train.py内のconfig辞書で定義されています。主要なパラメータは以下の通りです：
- embedding_size: 768
- num_heads: 6
- depth: 6
- sentence_size: 1024
- batch_size: 6
- warmup_iters: 2000
- max_lr: 2.5e-5
- min_lr: 2.5e-6
- max_iters: 10000
- batch_iteration: 128
- val_iteration: 1
- vocab_size: 50257
### 学習プロセス
1. OpenWebTextデータセットをロードし、前処理を行います。
2. データをトークン化し、メモリマップファイルを作成します。
3. GPTモデルを初期化します。
4. 学習ループを開始し、指定されたイテレーション数だけ繰り返します。
5. 各イテレーションでバッチデータを取得し、モデルに入力します。
6. 損失を計算し、勾配を更新します。
7. 定期的に検証を行い、最良のモデルを保存します。
8. 学習の詳細を記録します。
### テキスト生成
GPTクラスのgenerate()メソッドとgenerate_sentence()メソッドを使用してテキストを生成できます。これらのメソッドは、与えられた開始トークンから続きのテキストを生成します。
### 依存関係
プロジェクトの主な依存関係は以下の通りです：
- Python 3.11以上
- PyTorch 2.0.0以上
- NumPy 1.26.4以上
- tqdm 4.62.0以上
- tiktoken 0.6.0以上
- datasets 2.3.0以上
これらの依存関係はpyproject.tomlファイルで管理されています。

## 今後の改善点
1. より大規模なデータセットでの学習
2. モデルアーキテクチャの改良（例：より多くのレイヤーや注意ヘッドの追加）
3. ハイパーパラメータのチューニング
4. 推論速度の最適化
5. マルチGPU学習のサポート
6. より洗練されたテキスト生成アルゴリズムの実装（例：ビームサーチ）

## トラブルシューティング
CUDA関連のエラーが発生した場合、GPUが利用可能であることを確認し、必要に応じてCUDAドライバーをアップデートしてください。
メモリ不足エラーが発生した場合、batch_sizeやsentence_sizeを小さくしてみてください。
学習が進まない場合、学習率や他のハイパーパラメータを調整してみてください。
