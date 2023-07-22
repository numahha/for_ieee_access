
Install
```
conda create -n iwvi python==3.8
conda activate iwvi
pip install notebook matplotlib torch 
pip install gym==0.25.1
```

# pendulum memo 
* コストの係数を変えてみる(1.0はダメ、0.1は今の所一番マシ、0.5もあり)
* ペナルティ係数を上げる
* 今の所一番マシなのは、初期状態0.8,1でhidden=32でalpha=0.2

係数0.5もやってみる


次の条件で一通り探索し切る
* コスト変更した今で固定
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


```
(iwvi) hishinuma@hishinuma:~$ pip freeze
anyio==3.6.2
argon2-cffi==21.3.0
argon2-cffi-bindings==21.2.0
arrow==1.2.3
asttokens==2.2.1
attrs==22.2.0
backcall==0.2.0
beautifulsoup4==4.11.2
bleach==6.0.0
certifi @ file:///croot/certifi_1671487769961/work/certifi
cffi==1.15.1
cloudpickle==2.2.1
comm==0.1.2
contourpy==1.0.7
cycler==0.11.0
debugpy==1.6.6
decorator==5.1.1
defusedxml==0.7.1
executing==1.2.0
fastjsonschema==2.16.2
fonttools==4.38.0
fqdn==1.5.1
gym==0.25.1
gym-notices==0.0.8
idna==3.4
importlib-metadata==6.0.0
importlib-resources==5.12.0
ipykernel==6.21.2
ipython==8.10.0
ipython-genutils==0.2.0
isoduration==20.11.0
jedi==0.18.2
Jinja2==3.1.2
joblib==1.2.0
jsonpointer==2.3
jsonschema==4.17.3
jupyter-events==0.6.3
jupyter_client==8.0.3
jupyter_core==5.2.0
jupyter_server==2.3.0
jupyter_server_terminals==0.4.4
jupyterlab-pygments==0.2.2
kiwisolver==1.4.4
MarkupSafe==2.1.2
matplotlib==3.7.0
matplotlib-inline==0.1.6
mistune==2.0.5
nbclassic==0.5.2
nbclient==0.7.2
nbconvert==7.2.9
nbformat==5.7.3
nest-asyncio==1.5.6
notebook==6.5.2
notebook_shim==0.2.2
numpy==1.24.2
nvidia-cublas-cu11==11.10.3.66
nvidia-cuda-nvrtc-cu11==11.7.99
nvidia-cuda-runtime-cu11==11.7.99
nvidia-cudnn-cu11==8.5.0.96
packaging==23.0
pandas==1.5.3
pandocfilters==1.5.0
parso==0.8.3
pexpect==4.8.0
pickleshare==0.7.5
Pillow==9.4.0
pkgutil_resolve_name==1.3.10
platformdirs==3.0.0
prometheus-client==0.16.0
prompt-toolkit==3.0.36
psutil==5.9.4
ptyprocess==0.7.0
pure-eval==0.2.2
pycparser==2.21
Pygments==2.14.0
pyparsing==3.0.9
pyrsistent==0.19.3
python-dateutil==2.8.2
python-json-logger==2.0.6
pytz==2022.7.1
PyYAML==6.0
pyzmq==25.0.0
rfc3339-validator==0.1.4
rfc3986-validator==0.1.1
scikit-learn==1.2.1
scipy==1.10.1
Send2Trash==1.8.0
six==1.16.0
sniffio==1.3.0
soupsieve==2.4
stack-data==0.6.2
terminado==0.17.1
threadpoolctl==3.1.0
tinycss2==1.2.1
torch==1.13.1
tornado==6.2
traitlets==5.9.0
typing_extensions==4.5.0
uri-template==1.2.0
wcwidth==0.2.6
webcolors==1.12
webencodings==0.5.1
websocket-client==1.5.1
zipp==3.14.0
(iwvi) hishinuma@hishinuma:~$ 
```