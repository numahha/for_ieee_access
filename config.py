cfg_env = 'pendulum'
# cfg_env = 'cartpole'

if cfg_env == "pendulum":
    cfg_z_dim = 1
    cfg_alpha = 0.1 # 0.3だと学習が発散?　0.01だと発散しない
    cfg_sac_num_steps = 100000
    cfg_dec_hidden=32 

if cfg_env == "cartpole":
    cfg_z_dim = 2
    cfg_alpha = 0.1
    cfg_sac_num_steps = 200000
    cfg_dec_hidden=64

cfg_seed = 0
# cfg_seed = 1
cfg_default_lr=5e-4
cfg_default_early=100

