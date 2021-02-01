from run_experiment import extend_param_dicts
import random

server_list = [
    ("gimli",
     [
         0,
         1,
         2,
         3,
         4,
         5,
         6,
         7
     ],
    2),
]

label = "sc2_v4_homo_maven"
config = "noisemix_smac"
env_config = "sc2"

n_repeat = 16  # Just incase some die

parallel_repeat = 2

param_dicts = []

shared_params = {
    #"local_results_path": "/data/dgx1/tarpta/results/",
    "t_max": 6 * 1000 * 1000 + 50 * 1000,
    "runner": "parallel",
    "batch_size_run": 8,
    "batch_size": 32,
    "test_interval": 30000,
    "test_nepisode": 8,
    "test_greedy": True,
    "save_model": False,
    #"save_model_interval": 250 * 1000,
    "log_interval": 30000,
    "runner_log_interval": 30000,
    "learner_log_interval": 30000,
    "buffer_cpu_only": True,  # 5k buffer is too big for VRAM!
    "training_iters": 1,
    "buffer_size": 3000
}
name = label + config + "_" + env_config
extend_param_dicts(param_dicts, shared_params,
                   {
                       "lr": [0.0005],
                       "epsilon_anneal_time": [250000],
                       "env_args.reward_only_positive": [False],
                       "env_args.reward_negative_scale": [1.0],
                       "env_args.map_name": ["8m", "MMM2", "10m_vs_11m"],
                       "name": name,
                       "rnn_hidden_dim": [128],
                       "obs_agent_id": [False],
                       "grad_norm_clip": [10],
                       "target_update_mode": ["soft"],
                       "target_update_tau": [0.05],
                       "mi_loss": [0.001, 0.01],
                       "agent": ["noise_rnn_deep"]
                   },
                   repeats=parallel_repeat)
