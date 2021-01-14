from run_experiment import extend_param_dicts

server_list = [
    ("sauron", [2,3,4,5,6,7], 2),
]

label = "maven_maps_original_maven_v1"

config = "noisemix_smac"
env_config = "sc2"

n_repeat = 6 # Just incase some die

parallel_repeat =  1

param_dicts = []

shared_params = {
    "local_results_path": "/data/{}/tarpta/results/".format(server_list[0][0]),
    "t_max": 10 * 1000 * 1000 + 50 * 1000,
    "env_args.reward_only_positive": False,
    # "env_args.good_branches": 2,

    # "batch_size_run": 1,

    "test_interval": 30000,
    "test_nepisode": 8,
    "test_greedy": True,
    "log_interval": 30000,
    "runner_log_interval": 30000,
    "learner_log_interval": 30000,
    "buffer_cpu_only": True, # 5k buffer is too big for VRAM!
    "buffer_size": 3000,
    "epsilon_finish": 0.05,
    "epsilon_anneal_time": 250000,
    #"discrim_size": 32,
}

name = "noisemix"
extend_param_dicts(param_dicts, shared_params,
    {
        "env_args.map_name": ["4step"],
        "name": name,
        "noise_dim": [32],
        #"bandit_iters": 100,
        "noise_bandit": [True],
        "rnn_discrim": [True],
        "mi_loss": [0.01],
        "entropy_scaling": [0.01]
    },
    repeats=parallel_repeat)
