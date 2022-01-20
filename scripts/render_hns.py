#!/usr/bin/env python3
import sys
sys.path.append("..")
import numpy as np
from os.path import abspath, dirname, join
from gym.spaces import Tuple
from envs.hns.viewer.env_viewer import EnvViewer
from envs.hns.wrappers.multi_agent import JoinMultiAgentActions
from mujoco_worldgen.util.envs import examine_env, load_env
from mujoco_worldgen.util.types import extract_matching_arguments
from mujoco_worldgen.util.parse_arguments import parse_arguments
import torch
from config import get_config
import pdb

def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str,
                        default='quadrant', help="Which scenario to run on")
    parser.add_argument('--floor_size', type=float,
                        default=6.0, help="size of floor")
    parser.add_argument('--grid_size', type=int,
                        default=30, help="size of floor")
    parser.add_argument('--door_size', type=int,
                        default=2, help="size of floor")
    parser.add_argument('--prep_fraction', type=float, default=0.4)

    # transfer task
    parser.add_argument('--num_agents', type=int,
                        default=2, help="number of players")
    parser.add_argument('--num_boxes', type=int,
                        default=4, help="number of boxes")
    parser.add_argument("--task_type", type=str, default='all')
    parser.add_argument("--objective_placement", type=str, default='center')

    # hide and seek task
    parser.add_argument("--num_seekers", type=int,
                        default=1, help="number of seekers")
    parser.add_argument("--num_hiders", type=int,
                        default=1, help="number of hiders")
    parser.add_argument("--num_ramps", type=int,
                        default=1, help="number of ramps")
    parser.add_argument("--num_food", type=int,
                        default=0, help="number of food")

    parser.add_argument("--training_role", type=int,
                        default=0, help="number of food")
    
    all_args = parser.parse_known_args(args)[0]

    return all_args

def main(args):
    '''
    examine.py is used to display environments and run policies.

    For an example environment jsonnet, see
        mujoco-worldgen/examples/example_env_examine.jsonnet
    You can find saved policies and the in the 'examples' together with the environment they were
    trained in and the hyperparameters used. The naming used is 'examples/<env_name>.jsonnet' for
    the environment jsonnet file and 'examples/<env_name>.npz' for the policy weights file.
    Example uses:
        bin/examine.py hide_and_seek
        bin/examine.py mae_envs/envs/base.py
        bin/examine.py base n_boxes=6 n_ramps=2 n_agents=3
        bin/examine.py my_env_jsonnet.jsonnet
        bin/examine.py my_env_jsonnet.jsonnet my_policy.npz
        bin/examine.py hide_and_seek my_policy.npz n_hiders=3 n_seekers=2 n_boxes=8 n_ramps=1
        bin/examine.py examples/hide_and_seek_quadrant.jsonnet examples/hide_and_seek_quadrant.npz
    '''
    #names, kwargs = parse_arguments(argv)
    args = get_config()
    # args = parse_args(args, parser)
    kwargs={'args': args}

    env_name = args.env_name
    num_hiders = args.num_hiders
    num_seekers = args.num_seekers
    num_agents = num_hiders + num_seekers
    core_dir = abspath(join(dirname(__file__)))
    envs_dir = '/Users/chenjy/Desktop/VACL/envs/hns/envs/'  # where hide_and_seek.py is.
    xmls_dir = 'xmls'

    if args.use_render:  # run policies on the environment
        # importing PolicyViewer and load_policy here because they depend on several
        # packages which are only needed for playing policies, not for any of the
        # environments code.
        from envs.hns.viewer.policy_viewer import PolicyViewer_hs_single 
        # from onpolicy.envs.hns.ma_policy.load_policy import load_policy
        env, args_remaining_env = load_env(env_name, core_dir=core_dir,
                                           envs_dir=envs_dir, xmls_dir=xmls_dir,
                                           return_args_remaining=True, **kwargs)
        
        if isinstance(env.action_space, Tuple):
            env = JoinMultiAgentActions(env)
        if env is None:
            raise Exception(f'Could not find environment based on pattern {env_name}')
        
        env.reset()  # generate action and observation spaces
        

        role = {'hider','seeker'}
        model_dir = {}
        policies = []
        for agent_id in range(num_agents):
            if args.share_policy:
                actor_critic = torch.load(str(args.model_dir) + "/agent_model_884.pt", map_location=torch.device('cpu'))['model']
            else:
                actor_critic = torch.load(str(args.model_dir) + "/agent" + str(agent_id) + "_model.pt", map_location=torch.device('cpu'))['model']
            actor_critic.device = torch.device('cpu')
            policies.append(actor_critic)

        args_remaining_policy = args_remaining_env
        
        if env is not None and policies is not None:
            args_to_pass, args_remaining_viewer = extract_matching_arguments(PolicyViewer_hs_single, kwargs)
            args_remaining = set(args_remaining_env)
            args_remaining = args_remaining.intersection(set(args_remaining_policy))
            args_remaining = args_remaining.intersection(set(args_remaining_viewer))
            assert len(args_remaining) == 0, (
                f"There left unused arguments: {args_remaining}. There shouldn't be any.")
            viewer = PolicyViewer_hs_single(args, env, policies, **args_to_pass)
            viewer.run()
    else:
        # examine the environment
        examine_env(env_name, kwargs,
                    core_dir=core_dir, envs_dir=envs_dir, xmls_dir=xmls_dir,
                    env_viewer=EnvViewer)

if __name__ == '__main__':
    main(sys.argv)
