from run_experiment import extend_param_dicts
import random

server_list = [
    ("orion", [0, 1, 2, 3], 1)
]

label = "_pred_prey_sf_set_3_"
config = "iql"
env_config = "pred_prey_tf_single"

n_repeat = 2  # Just incase some die

parallel_repeat = 2

param_dicts = []

shared_params = {
    "t_max": 5 * 1000 * 1000 + 50 * 1000,
    "runner": "parallel",
    "batch_size_run": 2,
    "test_interval": 10000,
    "test_nepisode": 32,
    "test_greedy": True,
    "save_model": True,
    "save_model_interval": 50 * 1000,
    "log_interval": 10000,
    "runner_log_interval": 10000,
    "learner_log_interval": 10000,
    "buffer_cpu_only": True,  # 5k buffer is too big for VRAM!
    "env_args.episode_limit": 1000
}
name = label + config + "_" + env_config
extend_param_dicts(param_dicts, shared_params,
                   {
                       "env_args.prey_stay_prob": [0.3],
                       # "env_args.state_type": ["all_obs", "grid"],
                       "env_args.penalty": [-0.02],
                       "env_args.obs_include_boundary_four_directions": [True],
                       "env_args.obs_include_x_y": [False],
                       # "env_args.obs_include_step_count": [True, False],
                       "epsilon_anneal_time": [250000],
                       # "epsilon_finish": [0.0, 0.05],
                       "name": name,
                       "env_args.prey_types": ["[1]"],
                       "rnn_hidden_dim": [128],
                       "obs_agent_id": [True, False]
                   },
                   repeats=parallel_repeat)
