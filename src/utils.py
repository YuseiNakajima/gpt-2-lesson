import numpy as np

def get_lr(cur_iter, config):
    """
    現在のイテレーションに基づいて学習率を計算する関数。

    引数:
        cur_iter (int): 現在のイテレーション
        config (dict): 設定パラメータを含む辞書

    戻り値:
        float: 計算された学習率
    """
    if cur_iter < config['warmup_iters']:
        return config['max_lr'] * cur_iter / config['warmup_iters']
    return (config['max_lr'] * (np.cos(cur_iter / config['max_iters'] * np.pi) + 1)).clip(config['min_lr'], config['max_lr'])
