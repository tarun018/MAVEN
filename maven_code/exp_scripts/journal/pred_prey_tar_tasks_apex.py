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
    # ("brown",
    #  [
    #      0,
    #      1,
    #      2,
    #      3,
    #      4,
    #      5,
    #      6,
    #      7
    #  ],
    #  1),
    ("crux",
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
    2),
]

label = "apex_sf_2807_v4_homo"
#label = "tst"
config = "vdn_journal_sf"
env_config = "pred_prey_tf_tasks_prey_p333p_pen_1"

n_repeat = 16  # Just incase some die

parallel_repeat = 1

param_dicts = []

shared_params = {
    "local_results_path": "/data/crux/tarpta/results/",
    "t_max": 10 * 1000 * 1000 + 50 * 1000,
    "runner": "parallel",
    "batch_size_run": 6,
    "batch_size": 6,
    "test_interval": 30000,
    "test_nepisode": 6,
    "test_greedy": True,
    "save_model": True,
    "save_model_interval": 250 * 1000,
    "log_interval": 30000,
    "runner_log_interval": 30000,
    "learner_log_interval": 30000,
    "buffer_cpu_only": True,  # 5k buffer is too big for VRAM!
    #"env_args.episode_limit": 800,
    "training_iters": 5,
    "second_device": "cuda:0",
    "buffer_size": 1000
}
name = label + config + "_" + env_config
extend_param_dicts(param_dicts, shared_params,
                   {
                       #"checkpoint_path": ["/data/dgx1/tarpta/results/models/97417/"],
                       #"load_step": [3200],
                       "opt_level": ["O0"],
                       "lr": [0.0005],
                       "max_related_policies_task": [200000],
                       "variance_multiplier_rew": [0.5],
                       "variance_multiplier_pen": [0.1],
                       "num_related_task_policies_rew": [3],
                       "num_related_task_policies_pen": [3],
                       "sample_zero_penalty": [True],
                       "env_args.step_cost": [-0.0001],
                       "epsilon_anneal_time": [250000, 1000000],
                       "env_args.observe_prey_ID": [True],
                       "primary_task_selection_priority_start": [0.3, 1.0],
                       "primary_task_selection_priority_finish": [1.0],
                       "task_selection_anneal_time": [1000000],
                       #"env_args.episode_limit": [1200],
                       #"env_args.grid_shape": ["[7,7]"],
                       #"env_args.max_n_preys": [4],
                       #"env_args.end_episode_on_1_capture": [False],
                       "name": name,
                       "rnn_hidden_dim": [128],
                       "obs_agent_id": [False],
                       "nn_activation": ["relu"],
                       "optimiser_to_use": ["rms"],
                       "grad_norm_clip": [10],
                       "positive_sf": [False],
                       "target_update_mode": ["soft"],
                       "target_update_tau": [0.005],
                       #"target_update_interval": [100, 200, 400],
                       "agent": ["rnn_sf_gru_WTI"],
                       #"seed": [548773837],
                       #"agent": ["rnn_sf_gru_WTI_SEPOUT"],
                       #"learner": ["sf_q_learner_nogpi"],
                       #"use_detach": [False]
                   },
                   repeats=parallel_repeat)
