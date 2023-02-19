TODO:
* rollout_simenv refactor
  * goal is to capture z for real/simulation transition data
* 拡大状態版SACを実装


conda create -n iwvi python==3.8
conda activate iwvi
pip install notebook matplotlib torch 
pip install gym==0.25.1




# 開発メモ
* 最終的にはjoint-optimization vs two-stage optimizationで議論をするため、two-stageのモデル推定は一応何でもいいはず。
  * ただし、同じにそろえておけば、重み付きvs重み無しの話もやりやすくなる、かも？

# プライベート研究
* deep generative modelの教科書を実装する
* kappa
* ペナルティのオフセット
* 初期信念について、サンプルいっぱいとって正規近似という手もある。


# 論文メモ
* 問題設定が違うので、wmopoとの比較は要らない
