import argparse

def get_config():
    # get the parameters
    parser = argparse.ArgumentParser(description='MAPPO-sc.')

    # prepare
    parser.add_argument("--algorithm_name", type=str, default='mappo')
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cuda", action='store_false', default=True)
    parser.add_argument("--cuda_deterministic", action='store_false', default=True)
    parser.add_argument("--n_training_threads", type=int, default=12)
    parser.add_argument("--n_rollout_threads", type=int, default=32)
    parser.add_argument("--num_env_steps", type=int, default=10e6, help='number of environment steps to train (default: 10e6)') 
    
    # env
    parser.add_argument("--env_name", type=str, default='batch1')
    parser.add_argument("--num_agents", type=int, default=3)
    parser.add_argument("--share_reward", action='store_false', default=True)
    
    # hanabi
    parser.add_argument("--hanabi_name", type=str, default='Hanabi-Full-Minimal')
    # mpe
    parser.add_argument("--scenario_name", type=str, default='simple_spread')
    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument("--num_good_agents", type=int, default=3)
    parser.add_argument("--num_adversaries", type=int, default=1)
    
    # do not need num_agents
    # starcraft2
    parser.add_argument("--map_name", type=str, default='3m')
    # hide and seek
    parser.add_argument("--num_seekers", type=int, default=1)
    parser.add_argument("--num_hiders", type=int, default=1)
    parser.add_argument("--num_boxes", type=int, default=1)
    parser.add_argument("--num_ramps", type=int, default=1)
    parser.add_argument("--num_food", type=int, default=0)
    
    # network
    parser.add_argument("--share_policy", action='store_false', default=True, help='agent share the same policy')
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--layer_N", type=int, default=1)
    parser.add_argument("--use_ReLU", action='store_true', default=False)
    parser.add_argument("--use_common_layer", action='store_true', default=False)
    parser.add_argument("--use_popart", action='store_false', default=True)
    parser.add_argument("--use_feature_popart", action='store_true', default=False)
    parser.add_argument("--use_feature_normlization", action='store_false', default=True)   
    parser.add_argument("--use_orthogonal", action='store_false', default=True) 
    parser.add_argument("--multi_critic", action='store_true', default=False) 
    
    # lstm
    parser.add_argument("--naive_recurrent_policy", action='store_true', default=False, help='use a naive recurrent policy')
    parser.add_argument("--recurrent_policy", action='store_false', default=True, help='use a recurrent policy')
    parser.add_argument("--data_chunk_length", type=int, default=10)
    parser.add_argument("--critic_full_obs", action='store_true', default=False)
    
    # attn
    parser.add_argument("--attn", action='store_true', default=False)    
    parser.add_argument("--attn_N", type=int, default=1)
    parser.add_argument("--attn_size", type=int, default=64)
    parser.add_argument("--attn_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--use_average_pool", action='store_false', default=True)      
    
    # ppo
    parser.add_argument("--ppo_epoch", type=int, default=15, help='number of ppo epochs (default: 4)')    
    parser.add_argument("--use_clipped_value_loss", action='store_false', default=True)
    parser.add_argument("--clip_param", type=float, default=0.2, help='ppo clip parameter (default: 0.2)')
    parser.add_argument("--num_mini_batch", type=int, default=1, help='number of batches for ppo (default: 32)')   
    parser.add_argument("--entropy_coef", type=float, default=0.01, help='entropy term coefficient (default: 0.01)')
    parser.add_argument("--value_loss_coef", type=float, default=1, help='value loss coefficient (default: 0.5)')
    parser.add_argument("--lr", type=float, default=5e-4, help='learning rate (default: 7e-4)')
    parser.add_argument("--eps", type=float, default=1e-5, help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--use-max-grad-norm", action='store_false', default=True)
    parser.add_argument("--max-grad-norm", type=float, default=20.0, help='max norm of gradients (default: 0.5)')
    parser.add_argument("--use-gae", action='store_false', default=True, help='use generalized advantage estimation')
    parser.add_argument("--gamma", type=float, default=0.99, help='discount factor for rewards (default: 0.99)')
    parser.add_argument("--gae-lambda", type=float, default=0.95, help='gae lambda parameter (default: 0.95)')
    parser.add_argument("--use-proper-time-limits", action='store_true', default=False, help='compute returns taking into account time limits')
    parser.add_argument("--use_huber_loss", action='store_false', default=True)
    parser.add_argument("--huber_delta", type=float, default=10.0)   
    
    # replay buffer
    parser.add_argument("--episode_length", type=int, default=200, help='number of forward steps in A2C (default: 5)')

    # run
    parser.add_argument("--use-linear-lr-decay", action='store_true', default=False, help='use a linear schedule on the learning rate')
    
    # save
    parser.add_argument("--save_interval", type=int, default=10)
    
    # log
    parser.add_argument("--log_interval", type=int, default=1)    
    
    # eval
    parser.add_argument("--eval", action='store_true', default=False)
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--eval_episodes", type=int, default=32)
    
    # render
    parser.add_argument("--save_gifs", action='store_true', default=False)
    parser.add_argument("--ifi", type=float, default=0.333333)    
    parser.add_argument("--model_dir", type=str, default=None)
    
    args = parser.parse_args()

    return args
