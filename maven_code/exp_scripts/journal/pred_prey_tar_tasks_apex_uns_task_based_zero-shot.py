from run_experiment import extend_param_dicts
import random

server_list = [
    ("dgx1",
     [
         # 4,
         # 5,
         # 6,
         7
     ],
    2),
]

label = "apex_sf_task_based_2807_v4_homo_zero_shot_variBAD_seeded"
#label = "tst"
#label = "apex_sf_task_based_1508_v4_homo"
#label = "apex_sf_nogpi_uniform_v10_homo"
config = "vdn_journal_sf"
env_config = "pred_prey_tf_tasks_prey_p333p_pen_1"

n_repeat = 1  # Just incase some die

parallel_repeat = 1
exp_id = 102911
seed = 255609851

param_dicts = []

shared_params = {
    "local_results_path": "/data/{}/tarpta/results/".format(server_list[0][0]),
    "runner": "parallel",
    "batch_size_run": 20,
    "batch_size": 6,
    "test_nepisode": 20,
    "test_greedy": True,
    "save_model": False,
    "buffer_cpu_only": True,  # 5k buffer is too big for VRAM!
    # "env_args.episode_limit": 800,
    "buffer_size": 100,
    "evaluate": True,
    "checkpoint_path": "/data/{}/tarpta/results/models/{}".format(server_list[0][0], exp_id),
    # "use_cuda": False
    "second_device": "cuda:0",
}
name = label + config + "_" + env_config
extend_param_dicts(param_dicts, shared_params,
                   {
                       "opt_level": ["O0"],
                       "lr": [0.0005],
                       "max_related_policies_task": [200000],
                       "variance_multiplier": [0.1],
                       "num_related_task_policies": [6],
                       "sample_zero_penalty": [False],
                       "env_args.step_cost": [-0.0001],
                       "epsilon_anneal_time": [250000],
                       "env_args.observe_prey_ID": [True],
                       "primary_task_selection_priority_start": [1.0],
                       "primary_task_selection_priority_finish": [1.0],
                       "task_selection_anneal_time": [250000],
                       "name": name,
                       "rnn_hidden_dim": [128],
                       "obs_agent_id": [False],
                       "nn_activation": ["relu"],
                       "optimiser_to_use": ["rms"],
                       "grad_norm_clip": [10],
                       "positive_sf": [False],
                       "target_update_mode": ["soft"],
                       "target_update_tau": [0.005],
                       "agent": ["rnn_sf_gru_WTI"],
                       "seed": seed
                   },
                   repeats=parallel_repeat)
