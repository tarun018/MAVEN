from run_experiment import extend_param_dicts
import random

server_list = [
    ("dgx1",
     [
         # 0,
         1,
         # 2,
         # 3,
         4,
         # 5,
         6,
         # 7
     ],
    2),
]

label = "nmatrix_uniform_gpi_v4"
config = "vdn_journal_sf"
env_config = "nmatrix"

n_repeat = 6  # Just incase some die

parallel_repeat = 3

param_dicts = []

shared_params = {
    "local_results_path": "/data/{}/tarpta/results/".format(server_list[0][0]),
    "t_max": 3 * 1000 * 10 + 50 * 100,
    "runner": "episode",
    "batch_size_run": 1,
    "batch_size": 32,
    "test_interval": 1000,
    "test_nepisode": 32,
    "test_greedy": True,
    "save_model": True,
    "save_model_interval": 250 * 10,
    "log_interval": 1000,
    "runner_log_interval": 1000,
    "learner_log_interval": 1000,
    "buffer_cpu_only": True,  # 5k buffer is too big for VRAM!
    #"env_args.episode_limit": 800,
    "training_iters": 1,
    "second_device": "cuda:0",
    "buffer_size": 1000,
    "test_epsilon": 0.0
}
name = label + config + "_" + env_config
extend_param_dicts(param_dicts, shared_params,
                   {
                       "lr": [0.0005],
                       "max_related_policies_task": [200000],
                       "variance_multiplier": [0.1],
                       "num_related_task_policies": [6],
                       "sample_zero_penalty": [False],
                       "epsilon_anneal_time": [500],
                       "primary_task_selection_priority_start": [0.3],
                       "primary_task_selection_priority_finish": [1.0],
                       "env_args.steps": [10],
                       "task_selection_anneal_time": [500],
                       "name": name,
                       "rnn_hidden_dim": [32],
                       "obs_agent_id": [False],
                       "nn_activation": ["relu"],
                       "optimiser_to_use": ["rms"],
                       "grad_norm_clip": [10],
                       "positive_sf": [False],
                       "target_update_mode": ["soft"],
                       "target_update_tau": [0.005],
                       #"target_update_interval": [100, 200, 400],
                       "agent": ["rnn_sf_gru_WTI"],
                   },
                   repeats=parallel_repeat)
