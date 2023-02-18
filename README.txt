TODO:
* シミュレーションサンプル固定で密度比推定と結果の可視化

0.2 1e-4/5e-4は更新無し
0.5 5e-4は更新無し


tanhは潜在変数がうまく推定できない
層が増えている状態なら、重みは強めにかけた方がいい？



* 家のHDDから、ポスターの元ファイルを探して来る。

conda create -n iwvi python==3.8
conda activate iwvi
pip install notebook matplotlib gym torch 

コードリファクタ


拡大状態版SACを実装


# 開発メモ
* frontiersに出す？
* enc_beliefは、方策最適化で拡大状態を扱うために使う
* priorは学習しなくても良い？
* 最終的にはjoint-optimization vs two-stage optimizationで議論をするため、two-stageのモデル推定は一応何でもいいはず。
  * ただし、同じにそろえておけば、重み付きvs重み無しの話もやりやすくなる、かも？
* メタ訓練時の事後分布を、BAMDPプランニングの事前分布として用いる、ももう少し考える。


# プライベート研究
* deep generative modelの教科書を実装する
* kappa
* ペナルティのオフセット
* 初期信念について、サンプルいっぱいとって正規近似という手もある。


# 論文メモ
* 問題設定が違うので、wmopoとの比較は要らない
