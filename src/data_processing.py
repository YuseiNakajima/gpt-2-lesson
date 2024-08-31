import numpy as np
import torch
from datasets import load_dataset
import tiktoken
import pickle

def load_and_process_dataset(trust_remote_code=8):
    """
    OpenWebTextデータセットをロードし、前処理を行う関数。

    引数:
        num_proc_load_dataset (int): データセットのロードに使用するプロセス数

    戻り値:
        処理済みのデータセット
    """
    num_proc_load_dataset = 8
    # データセットのロード
    dataset = load_dataset("openwebtext", num_proc=num_proc_load_dataset, trust_remote_code=trust_remote_code)
    
    # データセットの分割
    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')  # テスト分割を検証分割にリネーム
    
    # トークナイザーの初期化
    tokenizer = tiktoken.get_encoding("gpt2")
    
    def process(example):
        """
        各テキストサンプルを処理する内部関数。

        引数:
            example (dict): テキストサンプルを含む辞書

        戻り値:
            処理済みのサンプル
        """
        ids = tokenizer.encode_ordinary(example['text'])
        ids.append(tokenizer.eot_token)  # 文末にトークンを追加
        return {'ids': ids, 'len': len(ids)}
    
    # データセットの各サンプルにprocess関数を適用
    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc_load_dataset,
    )
    
    return tokenized, tokenizer

def save_processed_dataset(tokenized, filename="tokenized_dataset.bin"):
    """
    処理済みデータセットをバイナリファイルとして保存する関数。

    引数:
        tokenized: 処理済みデータセット
        filename (str): 保存するファイル名
    """
    with open(filename, "wb") as p:
        pickle.dump(tokenized, p)

def load_processed_dataset(filename="tokenized_dataset.bin"):
    """
    保存された処理済みデータセットを読み込む関数。

    引数:
        filename (str): 読み込むファイル名

    戻り値:
        処理済みデータセット
    """
    with open(filename, "rb") as p:
        return pickle.load(p)

def create_memmap_files(tokenized, train_filename="train.bin", val_filename="val.bin"):
    """
    トークン化されたデータセットからmemmapファイルを作成する関数。

    引数:
        tokenized: トークン化されたデータセット
        train_filename (str): 訓練データ用のmemmapファイル名
        val_filename (str): 検証データ用のmemmapファイル名
    """
    for split, dset in tokenized.items():
        filename = train_filename if split == 'train' else val_filename
        length = np.sum(dset["len"], dtype=np.uint64)
        
        write_data = np.memmap(filename, dtype=np.uint16, mode="w+", shape=(length,))
        
        index = 0
        for iter_index in range(1024):  # 1024は任意の分割数
            add_data = dset.shard(num_shards=1024, index=iter_index, contiguous=True).with_format('numpy')
            add_data = np.concatenate(add_data['ids'])
            add_length = len(add_data)
            write_data[index:index+add_length] = add_data
            index += add_length
        
        write_data.flush()

def get_batch(split, sentence_size, batch_size, device):
    """
    指定されたsplitからバッチデータを取得する関数。

    引数:
        split (str): 'train'または'val'
        sentence_size (int): 各シーケンスの長さ
        batch_size (int): バッチサイズ
        device (str): 使用するデバイス ('cuda'または'cpu')

    戻り値:
        x (torch.Tensor): 入力シーケンス
        y (torch.Tensor): ターゲットシーケンス
    """
    data = np.memmap(f"{split}.bin", dtype=np.uint16, mode="r")
    
    index = torch.randint(len(data) - sentence_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+sentence_size]).astype(np.int64)) for i in index])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+sentence_size]).astype(np.int64)) for i in index])
    
    if device == "cuda":
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)
