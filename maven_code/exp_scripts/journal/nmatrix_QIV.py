from run_experiment import extend_param_dicts
import random

server_list = [
    ("sauron", [2, 3, 4, 5, 6, 7], 2)
]

label = "nmatrix_qiv_v5"
config = "iql"
env_config = "nmatrix"

n_repeat = 6  # Just incase some die

parallel_repeat = 2

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
                       "max_related_policies_task": [10],
                       "variance_multiplier": [0.1],
                       "num_related_task_policies": [1],
                       "epsilon_anneal_time": [500],
                       "name": name,
                       "rnn_hidden_dim": [32],
                       "obs_agent_id": [False],
                       "nn_activation": ["relu"],
                       "optimiser_to_use": ["rms"],
                       "grad_norm_clip": [10],
                       "target_update_mode": ["soft"],
                       "target_update_tau": [0.005],
                       "env_args.steps": [10],
                       #"target_update_interval": [400],
                       #"agent": ["rnn_WTI_even_less"],
                       #"env_args.state_type": ["grid"],
                       "agent": ["rnn_WTI"],
                       # "hypernet_layers": [1]
                       #"seed": [652231044]
                       #"learner": ["sf_q_learner_nogpi"],
                       #"use_detach": [False]
                   },
                   repeats=parallel_repeat)
