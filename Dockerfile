# M1 Mac用のDockerイメージを使用
FROM --platform=linux/arm64 python:3.11-slim

# 作業ディレクトリを設定
WORKDIR /app

# 必要なパッケージをインストール
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Poetryをインストール
RUN pip install poetry

# プロジェクトファイルをコピー
COPY pyproject.toml poetry.lock ./

# Poetryの仮想環境を無効化し、依存関係をインストール
RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi

# ソースコードをコピー
COPY . .

# PYTHONPATHを設定
ENV PYTHONPATH="${PYTHONPATH}:/app"

# コンテナ起動時に実行するコマンド
CMD ["poetry", "run", "python", "scripts/train.py"]
