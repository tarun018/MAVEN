from run_experiment import extend_param_dicts

server_list = [
    ("saruman", [0,1,2,3], 1)
]

label = "qmix_exp_280218"
config = "qmix_journal"
env_config = "sc2"

n_repeat = 5 # Just incase some die

parallel_repeat = 1

param_dicts = []

shared_params = {
    "t_max": 10 * 1000 * 1000 + 50 * 1000,
    "test_interval": 2000,
    "test_nepisode": 32,
    "test_greedy": True,
    "env_args.obs_own_health": True,
    "save_model": True,
    "save_model_interval": 25 * 1000,
    "test_interval": 10000,
    "log_interval": 10000,
    "runner_log_interval": 10000,
    "learner_log_interval": 10000,
    "buffer_cpu_only": True, # 5k buffer is too big for VRAM!
}


maps = []

maps += ["27m_vs_30m"]
# maps += ["MMM2"]
# maps += ["bane_vs_bane"]
# maps += ["3s5z"]
# maps += ["25m"]
# maps += ["6h_vs_8z"]
# maps += ["3s5z_vs_3s6z"]
# maps += ["5m_vs_6m"]
# maps += ["3s_vs_5z"]
# maps += ["corridor"]
# maps += ["3s_vs_4z"]

for map_name in maps:
    name = "qmix_exp_2802_{}".format(map_name)
    extend_param_dicts(param_dicts, shared_params,
        {
            "name": name,
            "env_args.map_name": map_name,
            "skip_connections": [False],
            "gated": False,
            "mixing_embed_dim": [32],
            "hypernet_layers": [2],
            "epsilon_anneal_time" : [500000, 750000, 1000000]
        },
        repeats=parallel_repeat)

