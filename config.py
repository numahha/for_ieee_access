# cfg_env = 'pendulum'
cfg_env = 'cartpole'

if cfg_env == "pendulum":
    cfg_z_dim = 1
    cfg_alpha = 0.1 # 0.3だと学習が発散?　0.01だと発散しない
    cfg_sac_num_steps = 80000
    cfg_dec_hidden=48 # pendulum: 1layer32unitは惜しい, 2layer32unitは重み無しで学習できてしまう, 2layer16unitも惜しい, 3layer16unitはアリ

if cfg_env == "cartpole":
    cfg_z_dim = 2
    cfg_alpha = 0.1
    cfg_sac_num_steps = 200000
    cfg_dec_hidden=48
cfg_seed = 0
cfg_default_lr=5e-4
cfg_default_early=100

