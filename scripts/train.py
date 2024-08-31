import torch
from src.model import GPT
from src.data_processing import load_and_process_dataset, save_processed_dataset, create_memmap_files, get_batch
from src.training import train
import tiktoken

import sys
sys.path.append('/path/to/your/src')
from src.model import GPT

def main():
    # 設定
    config = {
        'embedding_size': 768,
        'num_heads': 6,
        'depth': 6,
        'sentence_size': 1024,
        'batch_size': 6,
        'warmup_iters': 2000,
        'max_lr': 2.5e-5,
        'min_lr': 2.5e-6,
        'max_iters': 10000,
        'batch_iteration': 128,
        'val_iteration': 1,
        'vocab_size': 50257,
        'begin': 0,
    }

    # デバイスの設定
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # データの準備
    tokenized, tokenizer = load_and_process_dataset()
    save_processed_dataset(tokenized)
    create_memmap_files(tokenized)

    # モデルの初期化
    model = GPT(config['vocab_size'], config['embedding_size'], config['embedding_size']*4, 
                config['num_heads'], 0, batch_first=True, T=config['sentence_size'], N=config['depth']).to(device)

    # オプティマイザとスケーラーの設定
    optimizer = torch.optim.Adam(model.parameters(), lr=config['max_lr'])
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))

    # 学習の実行
    train(model, optimizer, scaler, tokenizer, device, config)

if __name__ == "__main__":
    main()
