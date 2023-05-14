
Install
```
conda create -n iwvi python==3.8
conda activate iwvi
pip install notebook matplotlib torch 
pip install gym==0.25.1
```

Edit `config.py`


## メモ

* 倒立振子の方策最適化をきれいな結果にする。
  * モデルの規模を変える
  * リミットを変える
  * alphaを変える
  * predict_divergeあたりを調整する？



TODO:

* 学習モデルに対して最適化した方策の下でのモデル学習、をもう一度ちゃんとやる。
* ペナルティモデルの方策最適化のもとでの利用
* 係数の計算
* cartpoleの例題を進める
* 原稿を進める
* 3週間で見通しをつける


h_minは

hmin = minsa M
m=1 Es′ ∼Pm (·|sa) [− ln Pm (s |sa)]

より小さい数

開発：
* 訓練データデータの整理・
* cart poleを一旦最後までやる
* alpha=0.5は方策最適化が微妙？
* 係数の計算を考える

毎日：
* 論文：alpha正則化の記述
* 論文：英語化

* 挙動の可視化



