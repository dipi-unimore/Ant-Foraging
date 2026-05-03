import os
import argparse

import numpy as np
import random

from environments.ants_env.ants import Ants
from agents.utils.utils import read_params
from agents.utils.logger import create_loggers
from agents.IQLearning.iql import IQL


def main(args: argparse.Namespace) -> None:
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    curdir = os.path.dirname(os.path.abspath(__file__))
    
    env_params, learning_params, visual_params, log_params = read_params(
        args.env_params_path,
        args.learning_params_path,
        args.visualizer_params_path,
        args.logger_params_path
    )

    if args.train:
        env = Ants(seed=args.random_seed, **env_params)
    else:
        env = Ants(args.random_seed, True, args.fixed_foods, **env_params)

    if args.render:
        from environments.ants_env.ants import AntsVisualizer
        env_vis = AntsVisualizer(env, **visual_params)
    else:
        env_vis = None

    base_logger, other_logger = create_loggers(
        curdir,
        env.FOOD_SOURCES,
        env_params["learners"],
        env_params["actions"],
        args.train, 
        args.qtable_path,
        log_params 
    )
    
    iql = IQL(
        env,
        env_vis,
        args.qtable_path,
        base_logger,
        other_logger,
        **learning_params
    )

    if args.train:
        computation_time = iql.train(args.print_metrics)
    else:
        computation_time = iql.eval()
    
    base_logger.write_params(env_params, learning_params, log_params, computation_time, args.train)
   
def check_args(args: argparse.Namespace) -> bool:
    assert (
        args.env_params_path != ""
        and os.path.isfile(args.env_params_path)
        and args.env_params_path.endswith(".json")
    ), "[ERROR] params path is empty or is not a file or is not a json file"

    assert (
        args.learning_params_path != ""
        and os.path.isfile(args.learning_params_path)
        and args.learning_params_path.endswith(".json")
    ), "[ERROR] learning params path is empty or is not a file or is not a json file"
    
    assert (
        args.visualizer_params_path != ""
        and os.path.isfile(args.visualizer_params_path)
        and args.visualizer_params_path.endswith(".json")
    ), "[ERROR] visualizer params path is empty or is not a file or is not a json file"
    
    assert (
        args.logger_params_path != ""
        and os.path.isfile(args.logger_params_path)
        and args.logger_params_path.endswith(".json")
    ), "[ERROR] logger params path is empty or is not a file or is not a json file"

    if len(args.qtable_path) != 0:
        assert(args.qtable_path.endswith(".npy")), "[ERROR] qtable weights file must be a npy file" 

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--env_params_path",
        type=str,
        default="environments/ants_env/config/env-params.json",
        required=False
    )

    parser.add_argument(
        "--visualizer_params_path",
        type=str,
        default="environments/ants_env/config/env_visualizer-params.json",
        required=False
    )
    
    parser.add_argument(
        "--learning_params_path",
        type=str,
        default="agents/IQLearning/config/learning-params.json",
        required=False
    )
    
    parser.add_argument(
        "--logger_params_path",
        type=str,
        default="agents/IQLearning/config/logger-params.json",
        required=False
    )

    parser.add_argument(
        "--qtable_path",
        type=str,
        default="",
        required=False
    )
    
    parser.add_argument("--random_seed", type=int, default=42, required=False)
    parser.add_argument("--train", type=bool, default=False, required=False)
    parser.add_argument("--render", type=bool, default=False, required=False)
    parser.add_argument("--fixed_foods", type=bool, default=False, required=False)
    parser.add_argument("--print_metrics", type=int, default=30, required=False)
    
    args = parser.parse_args()
    if check_args(args):
        print("Current args: ", args)
        main(args)
