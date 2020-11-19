from run_experiment import extend_param_dicts
import random

server_list = [
    ("saruman",
     [
         # 3,
         # 4,
         # # 5,
         # 6,
         7
     ],
    2),
]

label = "pred_prey_wqmix_v1_zero_shot_variBAD_v2_"
config = "ow_qmix"
# config = "ow_qmix"
env_config = "pred_prey_tf_tasks_prey_p333p_pen_1_0"

n_repeat = 1  # Just incase some die

parallel_repeat = 1

param_dicts = []
exp_id = 106332
seed = 230187121

shared_params = {
    "local_results_path": "/data/{}/tarpta/results/".format(server_list[0][0]),
    "runner": "parallel",
    "batch_size_run": 20,
    "batch_size": 6,
    "test_nepisode": 60,
    "test_greedy": True,
    "save_model": False,
    "buffer_cpu_only": True,  # 5k buffer is too big for VRAM!
    "buffer_size": 100,
    "evaluate": True,
    "checkpoint_path": "/data/{}/tarpta/results/models/{}".format(server_list[0][0], exp_id),
}
name = label + config + "_" + env_config
extend_param_dicts(param_dicts, shared_params,
                   {
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
                       "agent": ["rnn_even_less"],
                       "seed": seed
                   },
                   repeats=parallel_repeat)
