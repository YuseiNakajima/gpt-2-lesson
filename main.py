import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import warnings
import subprocess
from time import time, sleep

# 乱数シードの設定
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# ワーニングを無視
warnings.simplefilter('ignore')

# CUDA環境の確認
print("CUDA環境の確認: ", torch.cuda.is_available())
# CUDAが利用できない場合のエラーハンドリングを検討する


class PreLNGPTDecoderLayer(nn.Module):
    """
    Pre Layer Normalization GPT Decoder Layer（事前レイヤー正規化GPTデコーダ層）。

    引数:
        embedding_dim (int): 各埋め込みベクトルのサイズ。
        ffn_dim (int): フィードフォワードネットワークモデルの次元数。
        num_heads (int): マルチヘッドアテンションモデルのヘッド数。
        drop_out_rate (float, optional): ドロップアウト率。デフォルトは0.0。
        layer_eps (float, optional): LayerNormのためのイプシロン値。デフォルトは1e-05。
        batch_first (bool, optional): Trueの場合、入力および出力テンソルは(batch, seq, feature)として提供されます。デフォルトはFalse。
    """
    def __init__(self, embedding_dim, ffn_dim, num_heads, drop_out_rate = 0., layer_eps=1e-05, batch_first = False):
        super().__init__()
        self.masked_multihead_attention = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=batch_first)
        self.dropout_self_attn = nn.Dropout(p=drop_out_rate)
        self.layer_norm_self_attn = nn.LayerNorm(embedding_dim, eps=layer_eps)

        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, ffn_dim), 
            nn.GELU(), 
            nn.Linear(ffn_dim, embedding_dim)
        )
        self.layer_norm_ffn = nn.LayerNorm(embedding_dim, eps=layer_eps)
        self.dropout_ffn = nn.Dropout(p=drop_out_rate)

    def forward(self, x, pad_mask_self=None, mask_self=None):
        """
        レイヤーのフォワードパス。

        引数:
            x: 入力テンソル。
            pad_mask_self: 自己アテンションのためのパディングマスク。
            mask_self: 自己アテンションのためのアテンションマスク。

        戻り値:
            GPTデコーダ層を通過した後のテンソル。
        """
        attention_input = self.layer_norm_self_attn(x)
        attention_output, _ = self.masked_multihead_attention(
            attention_input, attention_input, attention_input,
            key_padding_mask=pad_mask_self, attn_mask=mask_self
        )
        attention_output = self.dropout_self_attn(attention_output)
        x = x + attention_output

        ffn_input = self.layer_norm_ffn(x)
        ffn_output = self.dropout_ffn(self.ffn(ffn_input))
        x = x + ffn_output

        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, embedding_dim, ffn_dim, num_heads, drop_out_rate = 0.,\
                layer_eps=1e-05, batch_first = False, T = 10000, N = 1):
        super().__init__()
        #Tはmax_lenを表している
        self.embedding = nn.Embedding(vocab_size, embedding_dim,)
        self.positional_embedding = nn.Embedding(T, embedding_dim)
        self.decoder = nn.ModuleList([PreLNGPTDecoderLayer(embedding_dim, ffn_dim, num_heads, drop_out_rate,\
                                                            layer_eps, batch_first) for _ in range(N)])
        self.linear = nn.Linear(embedding_dim, vocab_size, bias = False)
        self.vocab_size = vocab_size
    def forward(self, x, y = None,pad_mask_self = None, mask_self=None):
        """
        yはxを1つだけずらしたデータである
        x = data[a:b]なら、y = data[a+1:b+1]となる。
        """
        x = self.embedding(x)
        pos = torch.arange(0,x.size(1),dtype=torch.long).unsqueeze(0).to(x.device)
        pos = self.positional_embedding(pos)
        x = x + pos
        for layer in self.decoder:
            x = layer(x, pad_mask_self = pad_mask_self, mask_self = mask_self)
        x = self.linear(x)
        if y != None:
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-1)
            #ignore_index=-1はyをonehotベクトル化しないでcross_entropyを使うために使用
            pred = x.argmax(dim = -1).detach().cpu()
            return loss,pred
        loss = None
        pred = x[:,[-1],:]
        return loss, pred
    def create_mask(self, x: torch.tensor, x_pad: int, device: str):
        """
        (batch_size, sequence_length, embedding_dim)の入力を想定
        """
        """
        Trueが無視される値であることに注意すること
        """
        seq_len = x.size(1)
        #srcのマスク制作
        padding_mask = (x == x_pad)
        mask = torch.triu(torch.ones(size = (seq_len, seq_len))==1).transpose(0,1) #下三角行列を作る
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask==1.,float(0.0)).to(device)
        return padding_mask, mask

    @torch.no_grad()
    def generate(self,bos: str, sentence_size, tokenizer, device):
        self.eval()
        bos_tokenized = tokenizer.encode_ordinary(bos)
        bos_tokenized = bos_tokenized[-sentence_size:]
        bos_tokenized = torch.LongTensor([bos_tokenized])
        _, add_sentence = self(bos_tokenized.to(device))
        self.train()
        return add_sentence

    @torch.no_grad()
    def generate_sentence(self, bos: str, sentence_size, generate_tokens, tokenizer, device, top_K = None, temperature = 1.0):
        return_sentence = bos
        for i in range(generate_tokens):
            add_sentence = self.generate(return_sentence, sentence_size, tokenizer,device)
            add_sentence = add_sentence[:,-1,:] / temperature #(1, vocab_size)
            if top_K is not None:
                v, _ = torch.topk(add_sentence, min(top_K, add_sentence.size(-1)))
                #v[:, [-1]]がtopkの中でも最小値を取る。これより小さいやつは予想に含めない。
                add_sentence[add_sentence < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(add_sentence, dim = -1)
            idx_next = torch.multinomial(probs, num_samples=1)
            return_sentence += tokenizer.decode_batch(idx_next.tolist())[0]
        return return_sentence

device = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.tensor([[2, 10, 20, 100, 512, 3], [2, 10, 20, 100, 512, 3], [2, 10, 20, 100, 512, 3]], dtype=torch.long).to(device)
# x = x.reshape(3, 6)  # 
embedding_size = 768
num_heads = 12
# Parameters set based on Karpathy's minGPT
gpt = GPT(50257, embedding_size, embedding_size * 4, num_heads, 0.1, batch_first=True, T=1024, N=12).to(device)


# 前提として、xは適切に前処理されたシーケンシャルデータである必要があります。
padding_mask, mask = gpt.create_mask(x[0:2], 0, device)

# x[0:2] を入力とし、x[1:3] をターゲットとして使用
loss, pred = gpt(x[0:2], x[1:3], padding_mask, mask)

# loss が None でない場合にのみ出力
if loss is not None:
    print("Loss: \n", loss.item())  # loss.item() は、もし loss がテンソルの場合に値を取得するために使用
print("Pred: \n", pred)

print(gpt)


count_params = 0
for params in gpt.parameters():
    count_params += params.contiguous().view(-1).size(0)
print("The number of parameters is ", count_params)

# 使用したリソースを解放するためにオブジェクトの参照を削除
import gc
del gpt
del x
del padding_mask, mask

# ガベージコレクションを実行して未使用のメモリを解放
gc.collect()

# GPUメモリキャッシュをクリア（GPUメモリ不足の場合に有用）
torch.cuda.empty_cache()


# トークン化のために、tiktokenをimport
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
tokenizer.encode_ordinary("This is a sample.")

from datasets import load_dataset

num_proc_load_dataset = 8  # 使用するマシンのCPUコア数に基づいて調整
dataset = load_dataset("openwebtext", num_proc=num_proc_load_dataset)

import pickle

# 確認: dataset が pickle でシリアライズ可能であること
with open("dataset.bin", "wb") as p:
    pickle.dump(dataset, p)

import pickle

# ファイルパスの確認と、データソースの信頼性の確認
with open("dataset.bin", "rb") as p:
    dataset = pickle.load(p)

#dataset分割
split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
split_dataset['val'] = split_dataset.pop('test') # rename the test split to val

def process(example):
        ids = tokenizer.encode_ordinary(example['text'])
        ids.append(tokenizer.eot_token) #文末に<endoftext>tokenを追加
        out = {'ids': ids, 'len': len(ids)}
        return out


tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc_load_dataset,
    )

# 確認: tokenized が pickle でシリアライズ可能であること
with open("tokenized_dataset.bin", "wb") as p:
    pickle.dump(tokenized, p)


# ファイルパスの確認と、データソースの信頼性の確認
with open("tokenized_dataset.bin", "rb") as p:
    tokenized = pickle.load(p)


from tqdm import tqdm
for split, dset in tokenized.items():
    #split: train or val, dset: train_dataset or val_dataset
    filename = split+".bin"
    length = np.sum(dset["len"], dtype=np.uint64) #データの長さ
    write_data = np.memmap(filename, dtype = np.uint16, mode = "w+", shape = (length,)) #Vocabが50257サイズなのでuint16で事足りる
    iteration = 1024
    index = 0
    for iter_index in tqdm(range(iteration)):
        add_data = dset.shard(num_shards=iteration, index = iter_index, contiguous=True).with_format('numpy')
        #dataset.shardはnum_shardsに指定した数だけデータを分割する
        add_data = np.concatenate(add_data['ids'])
        add_length = len(add_data)
        write_data[index:index+add_length] = add_data
        index += add_length
    write_data.flush()


train_data = np.memmap("/content/train.bin", dtype = np.uint16, mode = "r")
val_data = np.memmap("/content/val.bin", dtype = np.uint16, mode = "r")


# 文のサイズとバッチサイズの設定
sentence_size = 1024
batch_size = 6

# GPUが利用可能な場合はCUDAを使用し、そうでない場合はCPUを使用
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_batch(split: str, batch_size=batch_size, device=device) ->torch.Tensor:
    # splitに基づいて対応するデータセット（訓練または検証）を選択
    data = train_data if split == 'train' else val_data

    # ランダムな開始点をバッチサイズ分選択
    index = torch.randint(len(data) - sentence_size, (batch_size,))

    # 選択した開始点からsentence_size分のデータを抽出してTensorに変換
    x = torch.stack([torch.from_numpy((data[i:i+sentence_size]).astype(np.int64)) for i in index])

    # 同様に、各開始点の1つ後ろからデータを抽出（教師データとして使用）
    y = torch.stack([torch.from_numpy((data[i+1:i+1+sentence_size]).astype(np.int64)) for i in index])

    # データを適切なデバイス（GPUまたはCPU）に移動
    if device == "cuda":
        # CUDAデバイスの場合、非ブロッキング転送を使用
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)


device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_size = 768
num_heads = 6
tokenizer = tiktoken.get_encoding("gpt2")
#KarpathyのminGPTを参考に、パラメーターを設定した。
depth = 6
gpt = GPT(50257, embedding_size, embedding_size*4, num_heads, 0, batch_first=True, T = sentence_size, N = depth).to(device)
#事前学習のときはDropout無し、ファインチューニングのときはありが好ましい
warmup_iters = 2000

optimizer = torch.optim.Adam(gpt.parameters(), lr = 0.0001)

max_lr = 2.5e-5
min_lr = 2.5e-6
max_iters = 10000
def get_lr(cur_iter):
    #cur_iter現在のiteration
    if cur_iter < warmup_iters:
        return max_lr * cur_iter / warmup_iters
    return (max_lr * (np.cos(cur_iter / max_iters * np.pi) + 1)).clip(min_lr, max_lr)


import gc
from tqdm import tqdm
batch_iteration = 128
scaler = torch.cuda.amp.GradScaler(enabled=True)
best_loss = 1e9
begin = 0
val_iteration = 1


import gc
from tqdm import tqdm
import torch
import numpy as np
from time import sleep

# 以前の設定に基づく変数
batch_iteration = 128
best_loss = 1e9
begin = 0
max_iters = 10000
val_iteration = 1

# 学習率スケジューラーの設定（必要に応じて変更してください）
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda cur_iter: get_lr(cur_iter))

# トレーニングの前に不要なメモリを解放
gc.collect()
torch.cuda.empty_cache()
# sleep(5)  # 

gpt.train()
for cur_iter in tqdm(range(begin, max_iters)):
    # 学習率の更新
    scheduler.step()

    for batch_iter in range(batch_iteration):
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=True):
            x, y = get_batch("train", batch_size=batch_size, device=device)
            padding_mask, mask = gpt.create_mask(x, 0, device)
            loss, pred = gpt(x, y, padding_mask, mask)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        del x, y, padding_mask, mask, loss, pred
        # 
        # gc.collect()
        # torch.cuda.empty_cache()

    valid_loss = 0
    for val_iter in range(val_iteration):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True):
                x, y = get_batch("valid", batch_size=batch_size, device=device)
                padding_mask, mask = gpt.create_mask(x, 0, device)
                loss, pred = gpt(x, y, padding_mask, mask)
                valid_loss += loss.detach()

                del x, y, padding_mask, mask, loss, pred
                # 
                # gc.collect()
                # torch.cuda.empty_cache()

    avg_valid_loss = valid_loss.item() / val_iteration
    if best_loss > avg_valid_loss:
        best_loss = avg_valid_loss
        checkpoint = {
            "model": gpt.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "iter": cur_iter,
            "best_loss": best_loss,
        }
        torch.save(checkpoint, "best_checkpoint.bin")
        print("params updated. BestLoss: ", best_loss)
        print("Val all loss", avg_valid_loss)

    if torch.isnan(valid_loss):
        print("Loss is NaN!")
        break

    # 最新のチェックポイントの保存
    checkpoint = {
        "model": gpt.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "iter": cur_iter,
        "best_loss": best_loss,
        "loss": avg_valid_loss
    }
    torch.save(checkpoint, "latest_checkpoint.bin")

    # ファイル書き込みの最適化
    with open("learning_detail_latest.txt", "w") as f:
        f.write("学習状況\n")
        f.write(f"iter: {cur_iter}\n")
        f.write("hyper params: \n")
        f.write(f"vocab_size: 50257, embedding size: {embedding_size}, ffn: {embedding_size * 4}, num_heads: {num_heads}, Depth: {depth}, sentence_size: {sentence_size}\n")
        f.write(f"lr: {scheduler.get_last_lr()[0]}, best_loss: {best_loss}\n")
        f.write(f"val_loss: {avg_valid_loss}\n")

    # メモリのクリーンアップ
    del valid_loss
    # gc.collect()  
    # torch.cuda.empty_cache()


checkpoint = torch.load("best_checkpoint.bin", map_location="cuda") #or cpu


gpt.load_state_dict(checkpoint["model"])


print(gpt.generate_sentence("What is OpenAI?", \
                            sentence_size, 128, tokenizer,device,top_K=20,temperature = 2))
