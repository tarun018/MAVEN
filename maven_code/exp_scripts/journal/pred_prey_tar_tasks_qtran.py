from run_experiment import extend_param_dicts
import random

server_list = [
    ("dgx1",
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

label = "pred_prey_qtran_v1_"
config = "qtran"
# config = "ow_qmix"
env_config = "pred_prey_tf_tasks_prey_p333p_pen_1_0"

n_repeat = 16  # Just incase some die

parallel_repeat = 2

param_dicts = []

shared_params = {
    "local_results_path": "/data/{}/tarpta/results/".format(server_list[0][0]),
    "t_max": 10 * 1000 * 1000 + 50 * 1000,
    "runner": "parallel",
    "batch_size_run": 8,
    "batch_size": 8,
    "test_interval": 30000,
    "test_nepisode": 8,
    "test_greedy": True,
    "save_model": True,
    "save_model_interval": 250 * 1000,
    "log_interval": 30000,
    "runner_log_interval": 30000,
    "learner_log_interval": 30000,
    "buffer_cpu_only": True,  # 5k buffer is too big for VRAM!
    #"env_args.episode_limit": 800,
    "training_iters": 4,
    "buffer_size": 1000
}
name = label + config + "_" + env_config
extend_param_dicts(param_dicts, shared_params,
                   {
                       #"checkpoint_path": ["/data/dgx1/tarpta/results/models/97417/"],
                       #"load_step": [3200],
                       "lr": [0.0005],
                       "max_related_policies_task": [10],
                       "variance_multiplier": [0.5],
                       "num_related_task_policies": [1],
                       "sample_zero_penalty": [False],
                       "env_args.step_cost": [-0.0001],
                       "epsilon_anneal_time": [250000],
                       "env_args.observe_prey_ID": [True],
                       "name": name,
                       "rnn_hidden_dim": [128],
                       "obs_agent_id": [False],
                       "nn_activation": ["relu"],
                       "grad_norm_clip": [10],
                       "target_update_mode": ["soft"],
                       "target_update_tau": [0.005],
                       #"target_update_interval": [100, 200, 400],
                       "agent": ["rnn_WTI"],
                   },
                   repeats=parallel_repeat)
