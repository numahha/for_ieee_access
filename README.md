
Install
```
conda create -n iwvi python==3.8
conda activate iwvi
pip install notebook matplotlib torch 
pip install gym==0.25.1
```

iter 3とiter 5をすぐ切り替えられるように整備

# pendulum memo 
次の条件で一通り探索し切る
* コスト変更した今で固定
* alphaも0.1で固定?
* 1隠れ層は48 or 64で固定
## 結果
a3,b64は微妙（seed=0はお情けでOKしても良い）



# memo
* 時間がない時には文章整理
* 時間がある時には開発

alpha=0.2 / iter=5を試す


ベスト反復選択・のような仕組みを実装する必要

リミッター・初期分布・隠れ層・alphaの組み合わせ


* 最後、新しいreal bamdp・割引報酬・同じ初期分布で評価し直す必要がある（後付でOK）

* 隠れ層の数48, 32だとシードを変えていくとstandardvaeでそのまま成功するのがある
  * 隠れ層16→alpha=0.1, 0.2, 0.3だとうまく方策評価できない
  * 初期状態分布を変える
    * 0.8を0.5に変えたものに対して、隠れ層32は微妙、48も惜しい
  * 各種係数
  * 方策訓練ステップ数

* 押していけば、一応原稿は埋まるくらいにはなった
  * 倒立振子の設定を作る（シード0...5）
  * 変えるかも知れない箇所（隠れ層の数、）

* 2回反復やっても良いかも
* 重み付きのやつが局所解にトラップされない要注意


# cartpole memo
cartpoleのkouho2は、seed=1だとだいぶ悪くなる。




TODO:

* 原稿を進める


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



