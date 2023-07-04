cfg_env = 'pendulum'
# cfg_env = 'cartpole'

if cfg_env == "pendulum":
    cfg_z_dim = 1
    cfg_alpha = 0.1
    cfg_sac_num_steps = 100000
    cfg_dec_hidden=48

if cfg_env == "cartpole":
    cfg_z_dim = 2
    cfg_alpha = 0.2
    cfg_sac_num_steps = 200000
    cfg_dec_hidden=64

# cfg_seed = 0
# cfg_seed = 1
# cfg_seed = 2
cfg_seed = 3
# cfg_seed = 4
cfg_default_lr=5e-4
cfg_default_early=100
cfg_policy_evaluation_maxiter=5

