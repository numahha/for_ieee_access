cfg_env = 'pendulum'
# cfg_env = 'cartpole'

if cfg_env == "pendulum":
    cfg_z_dim = 1
    cfg_alpha = 0.1 # 0.3だと学習が発散?　0.01だと発散しない

if cfg_env == "cartpole":
    cfg_z_dim = 1
    cfg_alpha = 0.1 # 0.3だと学習が発散?　0.01だと発散しない

cfg_seed = 0
