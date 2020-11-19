from run_experiment import extend_param_dicts
import random

server_list = [
    ("dgx1",
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

label = "apex_sc2_nogpi_vAA"
config = "vdn_journal_sf_nogpi"
env_config = "sc2_sf"

n_repeat = 16  # Just incase some die

parallel_repeat = 1

param_dicts = []

shared_params = {
    "local_results_path": "/data/dgx1/tarpta/results/",
    "t_max": 5 * 1000 * 1000 + 50 * 1000,
    "runner": "parallel",
    "batch_size_run": 4,
    "batch_size": 8,
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
    "training_iters": 4,
    "second_device": "cuda:0",
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
                       "sample_zero_penalty": [False],
                       "env_args.reward_only_positive": [False],
                       "env_args.reward_negative_scale": [0.5],
                       "env_args.map_name": ["5m_vs_6m", "3s5z_vs_3s6z", "6h_vs_8z", "MMM2", "3s_vs_5z", "10m_vs_11m", "3s5z", "1c3s5z"],
                       "name": name,
                       "rnn_hidden_dim": [1024],
                       "obs_agent_id": [False],
                       "nn_activation": ["relu"],
                       "optimiser_to_use": ["rms"],
                       "grad_norm_clip": [10],
                       "positive_sf": [False],
                       "target_update_mode": ["soft"],
                       "target_update_tau": [0.05],
                       #"target_update_interval": [600],
                       "agent": ["rnn_sf_gru_WTI_less"],
                       #"agent": ["rnn_sf_gru_WTI_SEPOUT"],
                       "learner": ["sf_q_learner_nogpi"],
                       "reshape_task_input": [True],
                       #"use_detach": [False]
                   },
                   repeats=parallel_repeat)
