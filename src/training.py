import torch
import numpy as np
from tqdm import tqdm
import gc
from src.utils import get_lr
from src.data_processing import get_batch

def train(model, optimizer, scaler, tokenizer, device, config):
    """
    モデルの学習を行う関数。

    引数:
        model: 学習するモデル
        optimizer: オプティマイザ
        scaler: 勾配スケーラー
        tokenizer: トークナイザー
        device: 使用するデバイス
        config: 設定パラメータを含む辞書
    """
    best_loss = float('inf')
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda cur_iter: get_lr(cur_iter, config))

    for cur_iter in tqdm(range(config['begin'], config['max_iters'])):
        scheduler.step()

        for _ in range(config['batch_iteration']):
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                x, y = get_batch("train", config['sentence_size'], config['batch_size'], device)
                padding_mask, mask = model.create_mask(x, 0, device)
                loss, pred = model(x, y, padding_mask, mask)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            del x, y, padding_mask, mask, loss, pred

        valid_loss = validate(model, device, config)

        if best_loss > valid_loss:
            best_loss = valid_loss
            save_checkpoint(model, optimizer, scaler, cur_iter, best_loss, "best_checkpoint.bin")
            print(f"Params updated. Best Loss: {best_loss}")

        print(f"Val loss: {valid_loss}")

        if np.isnan(valid_loss):
            print("Loss is NaN!")
            break

        save_checkpoint(model, optimizer, scaler, cur_iter, best_loss, "latest_checkpoint.bin", valid_loss)
        save_learning_details(cur_iter, config, scheduler.get_last_lr()[0], best_loss, valid_loss)

        gc.collect()
        torch.cuda.empty_cache()

def validate(model, device, config):
    """
    モデルの検証を行う関数。

    引数:
        model: 検証するモデル
        device: 使用するデバイス
        config: 設定パラメータを含む辞書

    戻り値:
        平均検証損失
    """
    valid_loss = 0
    for _ in range(config['val_iteration']):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                x, y = get_batch("val", config['sentence_size'], config['batch_size'], device)
                padding_mask, mask = model.create_mask(x, 0, device)
                loss, pred = model(x, y, padding_mask, mask)
                valid_loss += loss.item()
        del x, y, padding_mask, mask, loss, pred
    return valid_loss / config['val_iteration']

def save_checkpoint(model, optimizer, scaler, cur_iter, best_loss, filename, loss=None):
    """
    チェックポイントを保存する関数。

    引数:
        model: 保存するモデル
        optimizer: オプティマイザ
        scaler: 勾配スケーラー
        cur_iter: 現在のイテレーション
        best_loss: 最良の損失値
        filename: 保存するファイル名
        loss: 現在の損失値（オプション）
    """
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "iter": cur_iter,
        "best_loss": best_loss,
    }
    if loss is not None:
        checkpoint["loss"] = loss
    torch.save(checkpoint, filename)

def save_learning_details(cur_iter, config, lr, best_loss, val_loss):
    """
    学習の詳細を保存する関数。

    引数:
        cur_iter: 現在のイテレーション
        config: 設定パラメータを含む辞書
        lr: 現在の学習率
        best_loss: 最良の損失値
        val_loss: 現在の検証損失
    """
    with open("learning_detail_latest.txt", "w") as f:
        f.write("学習状況\n")
        f.write(f"iter: {cur_iter}\n")
        f.write("hyper params: \n")
        f.write(f"vocab_size: {config['vocab_size']}, embedding size: {config['embedding_size']}, "
                f"ffn: {config['embedding_size'] * 4}, num_heads: {config['num_heads']}, "
                f"Depth: {config['depth']}, sentence_size: {config['sentence_size']}\n")
        f.write(f"lr: {lr}, best_loss: {best_loss}\n")
        f.write(f"val_loss: {val_loss}\n")
