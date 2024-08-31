import torch
from torch import nn
import torch.nn.functional as F

class PreLNGPTDecoderLayer(nn.Module):
    """
    Pre Layer Normalization GPT Decoder Layer（事前レイヤー正規化GPTデコーダ層）。
    
    この層は、Transformerアーキテクチャのデコーダ層を実装しています。
    事前レイヤー正規化を使用することで、勾配の流れを改善し、より深いネットワークの学習を可能にします。

    引数:
        embedding_dim (int): 各埋め込みベクトルのサイズ。
        ffn_dim (int): フィードフォワードネットワークモデルの次元数。
        num_heads (int): マルチヘッドアテンションモデルのヘッド数。
        drop_out_rate (float, optional): ドロップアウト率。デフォルトは0.0。
        layer_eps (float, optional): LayerNormのためのイプシロン値。デフォルトは1e-05。
        batch_first (bool, optional): Trueの場合、入力および出力テンソルは(batch, seq, feature)として提供されます。デフォルトはFalse。
    """

    def __init__(self, embedding_dim, ffn_dim, num_heads, drop_out_rate=0., layer_eps=1e-05, batch_first=False):
        super().__init__()
        
        # マルチヘッドセルフアテンション層
        self.masked_multihead_attention = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=batch_first)
        
        # セルフアテンション後のドロップアウト
        self.dropout_self_attn = nn.Dropout(p=drop_out_rate)
        
        # セルフアテンション用のレイヤー正規化
        self.layer_norm_self_attn = nn.LayerNorm(embedding_dim, eps=layer_eps)
        
        # フィードフォワードネットワーク
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, embedding_dim)
        )
        
        # フィードフォワードネットワーク用のレイヤー正規化
        self.layer_norm_ffn = nn.LayerNorm(embedding_dim, eps=layer_eps)
        
        # フィードフォワードネットワーク後のドロップアウト
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
        # セルフアテンション部分
        attention_input = self.layer_norm_self_attn(x)
        attention_output, _ = self.masked_multihead_attention(
            attention_input, attention_input, attention_input,
            key_padding_mask=pad_mask_self,
            attn_mask=mask_self
        )
        attention_output = self.dropout_self_attn(attention_output)
        x = x + attention_output

        # フィードフォワードネットワーク部分
        ffn_input = self.layer_norm_ffn(x)
        ffn_output = self.dropout_ffn(self.ffn(ffn_input))
        x = x + ffn_output

        return x

class GPT(nn.Module):
    """
    GPT (Generative Pre-trained Transformer) モデル。

    このクラスは、GPTモデルの全体的な構造を定義します。
    複数のPreLNGPTDecoderLayerを積み重ねて、
    入力シーケンスから次のトークンを予測するための言語モデルを構築します。

    引数:
        vocab_size (int): 語彙サイズ。
        embedding_dim (int): 埋め込みベクトルの次元数。
        ffn_dim (int): フィードフォワードネットワークの隠れ層の次元数。
        num_heads (int): マルチヘッドアテンションのヘッド数。
        drop_out_rate (float): ドロップアウト率。
        layer_eps (float): レイヤー正規化のイプシロン値。
        batch_first (bool): バッチが最初の次元かどうか。
        T (int): 最大シーケンス長。
        N (int): デコーダ層の数。
    """

    def __init__(self, vocab_size, embedding_dim, ffn_dim, num_heads, drop_out_rate=0.,
                layer_eps=1e-05, batch_first=False, T=10000, N=1):
        super().__init__()
        
        # トークン埋め込み層
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 位置埋め込み層
        self.positional_embedding = nn.Embedding(T, embedding_dim)
        
        # デコーダ層のリスト
        self.decoder = nn.ModuleList([
            PreLNGPTDecoderLayer(embedding_dim, ffn_dim, num_heads, drop_out_rate,
                                layer_eps, batch_first) for _ in range(N)
        ])
        
        # 出力層（語彙サイズに変換）
        self.linear = nn.Linear(embedding_dim, vocab_size, bias=False)
        
        self.vocab_size = vocab_size

    def forward(self, x, y=None, pad_mask_self=None, mask_self=None):
        """
        モデルのフォワードパス。

        引数:
            x: 入力シーケンス。
            y: ターゲットシーケンス（オプション）。
            pad_mask_self: パディングマスク。
            mask_self: アテンションマスク。

        戻り値:
            損失（学習時）または予測（推論時）。
        """
        # トークン埋め込みの適用
        x = self.embedding(x)
        
        # 位置埋め込みの生成と適用
        pos = torch.arange(0, x.size(1), dtype=torch.long).unsqueeze(0).to(x.device)
        pos = self.positional_embedding(pos)
        x = x + pos

        # デコーダ層を通過
        for layer in self.decoder:
            x = layer(x, pad_mask_self=pad_mask_self, mask_self=mask_self)

        # 出力層を通過
        x = self.linear(x)

        if y is not None:
            # 学習モード：損失を計算
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-1)
            pred = x.argmax(dim=-1).detach().cpu()
            return loss, pred
        else:
            # 推論モード：次のトークンを予測
            pred = x[:, [-1], :]
            return None, pred

    def create_mask(self, x: torch.tensor, x_pad: int, device: str):
        """
        マスクを生成する。

        引数:
            x: 入力テンソル (batch_size, sequence_length)
            x_pad: パディングトークンのID
            device: 使用するデバイス

        戻り値:
            padding_mask: パディングマスク
            mask: 因果的マスク（下三角行列）
        """
        seq_len = x.size(1)
        
        # パディングマスクの作成
        padding_mask = (x == x_pad).to(device)
        
        # 因果的マスクの作成（下三角行列）
        mask = torch.triu(torch.ones(size=(seq_len, seq_len), device=device) == 1).transpose(0, 1)
        
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        return padding_mask, mask

    @torch.no_grad()
    def generate(self, bos: str, sentence_size, tokenizer, device):
        """
        テキスト生成を行う。

        引数:
            bos: 生成開始トークン
            sentence_size: 生成するシーケンスの長さ
            tokenizer: トークナイザー
            device: 使用するデバイス

        戻り値:
            生成されたテキスト
        """
        self.eval()
        bos_tokenized = tokenizer.encode_ordinary(bos)
        bos_tokenized = bos_tokenized[-sentence_size:]
        bos_tokenized = torch.LongTensor([bos_tokenized])
        _, add_sentence = self(bos_tokenized.to(device))
        self.train()
        return add_sentence

    @torch.no_grad()
    def generate_sentence(self, bos: str, sentence_size, generate_tokens, tokenizer, device, top_K=None, temperature=1.0):
        """
        文章を生成する。

        引数:
            bos: 生成開始トークン
            sentence_size: 生成するシーケンスの長さ
            generate_tokens: 生成するトークン数
            tokenizer: トークナイザー
            device: 使用するデバイス
            top_K: Top-K サンプリングのK値
            temperature: 温度パラメータ

        戻り値:
            生成された文章
        """
        return_sentence = bos
        for i in range(generate_tokens):
            add_sentence = self.generate(return_sentence, sentence_size, tokenizer, device)
            add_sentence = add_sentence[:, -1, :] / temperature  # (1, vocab_size)
            
            if top_K is not None:
                v, _ = torch.topk(add_sentence, min(top_K, add_sentence.size(-1)))
                add_sentence[add_sentence < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(add_sentence, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            return_sentence += tokenizer.decode_batch(idx_next.tolist())[0]
        
        return return_sentence
