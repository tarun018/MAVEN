from run_experiment import extend_param_dicts
import random

server_list = [
    ("sauron", [2, 3, 4, 5, 6, 7], 2)
]

label = "apex_sc2_vdn_qmix_vAA"
config = "qmix_journal"
env_config = "sc2_sf"

n_repeat = 2  # Just incase some die

parallel_repeat = 3

param_dicts = []

shared_params = {
    "local_results_path": "/data/dgx1/tarpta/results/",
    "t_max": 10 * 1000 * 1000 + 50 * 1000,
    "runner": "parallel",
    "batch_size_run": 4,
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
    #"env_args.episode_limit": 800,
    "training_iters": 1,
    "buffer_size": 3000,
    "test_epsilon": 0.0
}
name = label + config + "_" + env_config
extend_param_dicts(param_dicts, shared_params,
                   {
                       "lr": [0.0005],
                       "max_related_policies_task": [10],
                       "variance_multiplier": [0.1],
                       "num_related_task_policies": [1],
                       "epsilon_anneal_time": [250000],
                       "env_args.reward_only_positive": [False],
                       "env_args.reward_negative_scale": [0.5],
                       "env_args.map_name": ["8m", "10m_vs_11m"],
                       "name": name,
                       "rnn_hidden_dim": [128],
                       "obs_agent_id": [False],
                       "nn_activation": ["relu"],
                       "optimiser_to_use": ["rms"],
                       "grad_norm_clip": [10],
                       "target_update_mode": ["soft"],
                       "target_update_tau": [0.05],
                       #"target_update_interval": [400],
                       "agent": ["rnn_WTI"],
                       "reshape_task_input": [True],
                       #"agent": ["rnn_sf_gru_WTI_SEPOUT"],
                       #"learner": ["sf_q_learner_nogpi"],
                       #"use_detach": [False]
                   },
                   repeats=parallel_repeat)
