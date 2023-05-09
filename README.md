
Install
```
conda create -n iwvi python==3.8
conda activate iwvi
pip install notebook matplotlib torch 
pip install gym==0.25.1
```

Edit `config.py`


## メモ

* 最適方策を与えた場合の方策評価を条件とする際のモデル学習はOK



TODO:

* 学習モデルに対して最適化した方策の下でのモデル学習、をもう一度ちゃんとやる。
* ペナルティモデルの学習、および、方策最適化のもとでの利用
* 係数の計算
* cartpoleの例題を進める
* 原稿を進める
* 3週間で見通しをつける
* predict_divergeあたりを調整する？



開発：
* 訓練データデータの整理・
* cart poleを一旦最後までやる
* alpha=0.5は方策最適化が微妙？
* 学習の際のモデルの保存（方策評価vs方策最適化、反復別に保存）
* 係数の計算・ペナルティモデルの学習を考える

毎日：
* 論文：正則化の記述
* 論文：英語化


* 挙動の可視化



