TODO:

* 初期値候補探索の際の候補を減らす仕組みを実装
* 学習の際のモデルの保存（方策評価vs方策最適化、反復別に保存）
* 正則化
* 結果を入れて送る。
* 係数の計算・ペナルティモデルの学習を考える


* 挙動の可視化



```
conda create -n iwvi python==3.8
conda activate iwvi
pip install notebook matplotlib torch 
pip install gym==0.25.1
```