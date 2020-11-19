from run_experiment import extend_param_dicts
import random

server_list = [
    # ("brown",
    #  [
    #      '0,1',
    #      '2,3',
    #      '4,5',
    #      '6,7'
    #  ],
    #  1),
    # ("leo",
    #  [
    #      '4,5',
    #      '6,7'
    #  ],
    #  1),
    # ("gimli",
    #  [
    #      # 1,
    #      # 2,
    #      # 3,
    #      # 4,
    #      0,
    #      7
    #  ],
    #  1),
    ("brown",
     [
         # 0,
         # 1,
         # 2,
         # 3,
         4,
         5,
         6,
         7
     ],
    1),
]

label = "_pred_prey_sf_homogeneous_v30_"
config = "vdn_journal"
env_config = "pred_prey_tf_tasks_prey_2_pen_3"

n_repeat = 4  # Just incase some die

parallel_repeat = 1

param_dicts = []

shared_params = {
    "local_results_path": "/data/brown/tarpta/results/",
    "t_max": 5 * 1000 * 1000 + 50 * 1000,
    "runner": "parallel",
    "batch_size_run": 6,
    "batch_size": 6,
    "test_interval": 10000,
    "test_nepisode": 12,
    "test_greedy": True,
    "save_model": True,
    "save_model_interval": 50 * 1000,
    "log_interval": 10000,
    "runner_log_interval": 10000,
    "learner_log_interval": 10000,
    "buffer_cpu_only": True,  # 5k buffer is too big for VRAM!
    "env_args.episode_limit": 800,
    "target_update_interval": 200,
    "training_iters": 5,
    "second_device": "cuda:0"
}
name = label + config + "_" + env_config
extend_param_dicts(param_dicts, shared_params,
                   {
                       "env_args.max_related_policies_task": [500000],
                       "env_args.use_gpi_during_test": [True],
                       "env_args.variance_multiplier": [0.5],
                       "env_args.num_related_task_policies": [10],
                       "env_args.step_cost": [-0.0005],
                       "epsilon_anneal_time": [250000],
                       "name": name,
                       "rnn_hidden_dim": [64],
                       "obs_agent_id": [False],
                       "nn_activation": ["relu"],
                       "lr": [0.0005],
                       "optimiser_to_use": ["rms"],
                       "grad_norm_clip": [10],
                       "positive_sf": [False],
                       "target_update_interval": [400],
                       "agent": ["rnn_sf_gru_v2"]
                   },
                   repeats=parallel_repeat)
