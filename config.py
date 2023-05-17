cfg_env = 'pendulum'
# cfg_env = 'cartpole'

if cfg_env == "pendulum":
    cfg_z_dim = 1
    cfg_alpha = 0.1 # 0.3だと学習が発散?　0.01だと発散しない

if cfg_env == "cartpole":
    cfg_z_dim = 1
    cfg_alpha = 0.1 # 0.3だと学習が発散?　0.01だと発散しない

cfg_seed = 0
cfg_sac_num_steps = 80000
cfg_default_lr=5e-4
cfg_default_early=100
cfg_penalty_lam_coef = 0.