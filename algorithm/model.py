import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.distributions import Bernoulli, Categorical, DiagGaussian
from utils.util import init
import copy
import math
import pdb

from .ppo import PopArt

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

# [L,[1,2],[1,2],[1,2]]   
# [33]
def split_obs(obs, split_shape):
    start_idx = 0
    split_obs = []
    for i in range(len(split_shape)):
        split_obs.append(obs[:,start_idx:(start_idx+split_shape[i][0]*split_shape[i][1])])
        start_idx += split_shape[i][0]*split_shape[i][1]
    return split_obs
    
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Policy(nn.Module):
    def __init__(self, obs_space, action_space, num_agents, base = None, actor_base=None, critic_base=None, base_kwargs=None, device=torch.device("cpu")):
        super(Policy, self).__init__()
        self.mixed_obs = False
        self.mixed_action = False
        self.multi_discrete = False
        self.device = device
        self.num_agents = num_agents
        if base_kwargs is None:
            base_kwargs = {}
        
        if obs_space.__class__.__name__ == "Box":
            obs_shape = obs_space.shape
        elif obs_space.__class__.__name__ == "list":
            if obs_space[-1].__class__.__name__ != "Box":
                obs_shape = obs_space
            else:# means all obs space is passed here
                # num_agents means agent_id
                # obs_space means all_obs_space
                agent_id = num_agents
                all_obs_space = obs_space
                if all_obs_space[agent_id].__class__.__name__ == "Box":
                    obs_shape = all_obs_space[agent_id].shape
                else:
                    obs_shape = all_obs_space[agent_id]
                self.mixed_obs = True                
        else:
            raise NotImplementedError
        
        if base is None:
            if self.mixed_obs:
                if len(obs_shape) == 3:
                    self.base = CNNBase(all_obs_space, agent_id, **base_kwargs)
                elif len(obs_shape) == 1:
                    self.base = MLPBase(all_obs_space, agent_id, **base_kwargs)
                else:
                    raise NotImplementedError
            else:
                if obs_shape[-1].__class__.__name__=='list':#attn
                    self.base = MLPBase(obs_shape, num_agents, **base_kwargs)
                else:
                    if len(obs_shape) == 3:
                        self.base = CNNBase(obs_shape, num_agents, **base_kwargs)
                    else:
                        self.base = MLPBase(obs_shape, num_agents, **base_kwargs)
        else:
            self.base = base
        self.actor_base = actor_base
        self.critic_base = critic_base

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def is_naive_recurrent(self):
        return self.base.is_naive_recurrent
        
    @property
    def is_attn(self):
        return self.base.is_attn

    @property
    def recurrent_hidden_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_size

    def forward(self, share_inputs, inputs, rnn_hxs_actor, rnn_hxs_critic, masks):
        raise NotImplementedError

    def act(self, agent_id, share_inputs, inputs, rnn_hxs_actor, rnn_hxs_critic, masks, available_actions=None, deterministic=False):
        share_inputs = share_inputs.to(self.device)
        inputs = inputs.to(self.device)
        rnn_hxs_actor = rnn_hxs_actor.to(self.device)
        rnn_hxs_critic = rnn_hxs_critic.to(self.device)
        masks = masks.to(self.device)
        if available_actions is not None:
            available_actions = available_actions.to(self.device)
        
        dist = self.actor_base(inputs, self.num_agents)
        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()
        action_log_probs = dist.log_probs(action)
        action_out = action
        action_log_probs_out = action_log_probs 
        value, rnn_hxs_actor, rnn_hxs_critic = self.critic_base(share_inputs, inputs, self.num_agents, rnn_hxs_actor, masks)       
        
        return value, action_out, action_log_probs_out, rnn_hxs_actor, rnn_hxs_critic

    def get_value(self, agent_id, share_inputs, inputs, rnn_hxs_actor, rnn_hxs_critic, masks):
    
        share_inputs = share_inputs.to(self.device)
        inputs = inputs.to(self.device)
        rnn_hxs_actor = rnn_hxs_actor.to(self.device)
        rnn_hxs_critic = rnn_hxs_critic.to(self.device)
        masks = masks.to(self.device)

        value, rnn_hxs_actor, rnn_hxs_critic = self.critic_base(share_inputs, inputs, self.num_agents, rnn_hxs_actor, masks) 
        
        return value, rnn_hxs_actor, rnn_hxs_critic

    def evaluate_actions(self, agent_id, share_inputs, inputs, rnn_hxs_actor, rnn_hxs_critic, masks, high_masks, action):
    
        share_inputs = share_inputs.to(self.device)
        inputs = inputs.to(self.device)
        rnn_hxs_actor = rnn_hxs_actor.to(self.device)
        rnn_hxs_critic = rnn_hxs_critic.to(self.device)
        masks = masks.to(self.device)
        high_masks = high_masks.to(self.device)
        action = action.to(self.device)
        
        dist = self.actor_base(inputs, self.num_agents)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy()
        action_log_probs_out = action_log_probs
        dist_entropy_out = dist_entropy.mean()
        value, rnn_hxs_actor, rnn_hxs_critic = self.critic_base(share_inputs, inputs, self.num_agents, rnn_hxs_actor, masks)  

        return value, action_log_probs_out, dist_entropy_out, rnn_hxs_actor, rnn_hxs_critic

    # for simple speaker listener
    def act_role(self, agent_id, share_inputs, inputs, role, rnn_hxs_actor, rnn_hxs_critic, masks, available_actions=None, deterministic=False):
        share_inputs = share_inputs.to(self.device)
        inputs = inputs.to(self.device)
        rnn_hxs_actor = rnn_hxs_actor.to(self.device)
        rnn_hxs_critic = rnn_hxs_critic.to(self.device)
        masks = masks.to(self.device)
        if available_actions is not None:
            available_actions = available_actions.to(self.device)
        
        dist = self.actor_base(inputs, self.num_agents)
        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()
        action_log_probs = dist.log_probs(action)
        action_out = action
        action_log_probs_out = action_log_probs 
        value, rnn_hxs_actor, rnn_hxs_critic = self.critic_base(share_inputs, inputs, self.num_agents, rnn_hxs_actor, masks)        
        
        return value, action_out, action_log_probs_out, rnn_hxs_actor, rnn_hxs_critic

    def get_value_role(self, agent_id, share_inputs, inputs, role, rnn_hxs_actor, rnn_hxs_critic, masks):
    
        share_inputs = share_inputs.to(self.device)
        inputs = inputs.to(self.device)
        rnn_hxs_actor = rnn_hxs_actor.to(self.device)
        rnn_hxs_critic = rnn_hxs_critic.to(self.device)
        masks = masks.to(self.device)
        
        value, rnn_hxs_actor, rnn_hxs_critic = self.critic_base(share_inputs, inputs, self.num_agents, rnn_hxs_actor, masks)  
        
        return value, rnn_hxs_actor, rnn_hxs_critic

    def evaluate_actions_role(self, agent_id, share_inputs, inputs, role, rnn_hxs_actor, rnn_hxs_critic, masks, high_masks, action):
    
        share_inputs = share_inputs.to(self.device)
        inputs = inputs.to(self.device)
        rnn_hxs_actor = rnn_hxs_actor.to(self.device)
        rnn_hxs_critic = rnn_hxs_critic.to(self.device)
        masks = masks.to(self.device)
        high_masks = high_masks.to(self.device)
        action = action.to(self.device)
        
        dist = self.actor_base(inputs, self.num_agents)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy()
        action_log_probs_out = action_log_probs
        dist_entropy_out = dist_entropy.mean()

        value, rnn_hxs_actor, rnn_hxs_critic = self.critic_base(share_inputs, inputs, self.num_agents, rnn_hxs_actor, masks) 

        return value, action_log_probs_out, dist_entropy_out, rnn_hxs_actor, rnn_hxs_critic

#obs_shape, num_agents, naive_recurrent, recurrent, hidden_size, attn, attn_size, attn_N, attn_heads, dropout, use_average_pool, use_common_layer, use_orthogonal
class NNBase(nn.Module):
    def __init__(self, obs_shape, num_agents, naive_recurrent=False, recurrent=False, hidden_size=64,
                 attn=False, attn_size=512, attn_N=2, attn_heads=8, dropout=0.05, use_average_pool=True, 
                 use_common_layer=False, use_orthogonal=True, use_ReLU=False, use_same_dim=False):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent
        self._naive_recurrent = naive_recurrent
        self._attn = attn
        self._use_common_layer = use_common_layer
        self._use_same_dim = use_same_dim
        #input_size, split_shape=None, d_model=512, attn_N=2, heads=8, dropout=0.0, use_average_pool=True, use_orthogonal=True 
        if self._attn:
            if self._use_same_dim:
                self.encoder_actor = Encoder(obs_shape, attn_size, attn_N, attn_heads, dropout, use_average_pool, use_orthogonal, use_ReLU)
                self.encoder_critic = Encoder(obs_shape, attn_size, attn_N, attn_heads, dropout, use_average_pool, use_orthogonal, use_ReLU)   
            else:
                self.encoder_actor = Encoder(obs_shape, attn_size, attn_N, attn_heads, dropout, use_average_pool, use_orthogonal, use_ReLU)
                self.encoder_critic = Encoder([[1,obs_shape[0]]]*num_agents, attn_size, attn_N, attn_heads, dropout, use_average_pool, use_orthogonal, use_ReLU)
    
        if self._recurrent or self._naive_recurrent:
            self.gru = nn.GRU(hidden_size, hidden_size)         
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    if use_orthogonal:
                        nn.init.orthogonal_(param)
                    else:
                        nn.init.xavier_uniform_(param)
            if not self._use_common_layer:
                self.gru_critic = nn.GRU(hidden_size, hidden_size)
                for name, param in self.gru_critic.named_parameters():
                    if 'bias' in name:
                        nn.init.constant_(param, 0)
                    elif 'weight' in name:
                        if use_orthogonal:
                            nn.init.orthogonal_(param)
                        else:
                            nn.init.xavier_uniform_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def is_naive_recurrent(self):
        return self._naive_recurrent
                
    @property
    def is_attn(self):
        return self._attn
        
    @property
    def recurrent_hidden_size(self):
        if self._recurrent or self._naive_recurrent or self._lstm:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            #x= self.gru(x.unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)          
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = torch.transpose(x.view(N, T, x.size(1)),0,1)
            
            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)

            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]
                rnn_scores, hxs = self.gru( x[start_idx:end_idx], hxs * masks[start_idx].view(1, -1, 1))                  
                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            
            x = torch.cat(outputs, dim=0)
            x= torch.transpose(x,0,1)

            # flatten
            x = x.reshape(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs
        
    def _forward_gru_critic(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru_critic(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            #x = self.gru_critic(x.unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = torch.transpose(x.view(N, T, x.size(1)),0,1)

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru_critic(x[start_idx:end_idx], hxs * masks[start_idx].view(1, -1, 1))
                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            x= torch.transpose(x,0,1)
            # flatten
            x = x.reshape(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs
               
class CNNBase(NNBase):
    def __init__(self, obs_shape, num_agents, naive_recurrent = False, recurrent=False, hidden_size=64, attn=False, attn_size=512, attn_N=2, attn_heads=8, dropout=0.05, use_average_pool=True, use_common_layer=False, use_feature_normlization=False, use_feature_popart=False, use_orthogonal=True, layer_N=1, use_ReLU=False):
        super(CNNBase, self).__init__(obs_shape, num_agents, naive_recurrent, recurrent, hidden_size, attn, attn_size, attn_N, attn_heads, dropout, use_average_pool, use_common_layer, use_orthogonal)
        
        self._use_common_layer = use_common_layer
        self._use_orthogonal = use_orthogonal
        
        if self._use_orthogonal:
            init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('relu'))
        else:
            init_ = lambda m: init(m, nn.init.xavier_uniform_, lambda x: nn.init.constant_(x, 0), gain = nn.init.calculate_gain('relu'))
                                       
        num_inputs = obs_shape[0]
        num_image = obs_shape[1]

        self.actor = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 3, stride=1)), nn.ReLU(),
            #init_(nn.Conv2d(32, 64, 3, stride=1)), nn.ReLU(),
            #init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), 
            Flatten(),
            init_(nn.Linear(32 * (num_image-3+1) * (num_image-3+1), hidden_size)), nn.ReLU(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.ReLU())

        self.critic = nn.Sequential(
            init_(nn.Conv2d(num_inputs * num_agents, 32, 3, stride=1)), nn.ReLU(),
            #init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            #init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), 
            Flatten(),
            init_(nn.Linear(32 * (num_image-3+1) * (num_image-3+1), hidden_size)), nn.ReLU(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.ReLU(),
            )
            
        if self._use_common_layer:
            self.actor = nn.Sequential(init_(nn.Conv2d(num_inputs, 32, 3, stride=1)), nn.ReLU())
            self.critic = nn.Sequential(init_(nn.Conv2d(num_inputs * num_agents, 32, 3, stride=1)), nn.ReLU())
            self.common_linear = nn.Sequential(
                Flatten(),
                init_(nn.Linear(32 * (num_image-3+1) * (num_image-3+1), hidden_size)), nn.ReLU(),
                init_(nn.Linear(hidden_size, hidden_size)), nn.ReLU())

        if self._use_orthogonal:
            init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        else:
            init_ = lambda m: init(m, nn.init.xavier_uniform_, lambda x: nn.init.constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

    def forward(self, agent_id, share_inputs, inputs, rnn_hxs_actor, rnn_hxs_critic, rnn_c_actor, rnn_c_critic, masks):
        x = inputs / 255.0
        share_x = share_inputs / 255.0
            
        if self._use_common_layer:
            hidden_actor = self.actor(x)
            hidden_critic = self.critic(share_x)
            hidden_actor = self.common_linear(hidden_actor)
            hidden_critic = self.common_linear(hidden_critic)
            
            if self.is_recurrent or self.is_naive_recurrent:
                hidden_actor, rnn_hxs_actor = self._forward_gru(hidden_actor, rnn_hxs_actor, masks)
                hidden_critic, rnn_hxs_critic = self._forward_gru(hidden_critic, rnn_hxs_critic, masks)
            
        else:
            hidden_actor = self.actor(x)
            hidden_critic = self.critic(share_x)

            if self.is_recurrent or self.is_naive_recurrent:
                hidden_actor, rnn_hxs_actor = self._forward_gru(hidden_actor, rnn_hxs_actor, masks)
                hidden_critic, rnn_hxs_critic = self._forward_gru_critic(hidden_critic, rnn_hxs_critic, masks)
                        
        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs_actor, rnn_hxs_critic

class MLPBase(NNBase):
    def __init__(self, obs_shape, num_agents, naive_recurrent = False, recurrent=False, hidden_size=64, 
                attn=False, attn_size=512, attn_N=2, attn_heads=8, dropout=0.05, use_average_pool=True, 
                use_common_layer=False, use_feature_normlization=True, use_feature_popart=True, 
                use_orthogonal=True, layer_N=1, use_ReLU=False, use_same_dim=False):
        super(MLPBase, self).__init__(obs_shape, num_agents, naive_recurrent, recurrent, hidden_size, 
                                      attn, attn_size, attn_N, attn_heads, dropout, use_average_pool, 
                                      use_common_layer, use_orthogonal, use_ReLU, use_same_dim)

        self._use_common_layer = use_common_layer
        self._use_feature_normlization = use_feature_normlization
        self._use_feature_popart = use_feature_popart
        self._use_orthogonal = use_orthogonal
        self._layer_N = layer_N
        self._use_ReLU = use_ReLU
        self._use_same_dim = use_same_dim
        self._attn = attn
        
        assert (self._use_feature_normlization and self._use_feature_popart) == False, ("--use_feature_normlization and --use_feature_popart can not be set True simultaneously.")
        if 'int' not in obs_shape[0].__class__.__name__: # mixed obs
            all_obs_space = obs_shape
            agent_id = num_agents
            num_agents = len(all_obs_space)
            if all_obs_space[agent_id].__class__.__name__ == "Box":
                obs_shape = all_obs_space[agent_id].shape
            else:
                obs_shape = all_obs_space[agent_id]
            share_obs_dim = 0
            for obs_space in all_obs_space:
                share_obs_dim += obs_space.shape[0]
        else:
            if self._use_same_dim:
                share_obs_dim = obs_shape[0]
            else:
                share_obs_dim = obs_shape[0]*num_agents
        
        if self._use_feature_normlization:
            self.actor_norm = nn.LayerNorm(obs_shape[0])
            self.critic_norm = nn.LayerNorm(share_obs_dim)
            
        if self._use_feature_popart:
            self.actor_norm = PopArt(obs_shape[0])
            self.critic_norm = PopArt(share_obs_dim)
            
        if self._attn:           
            if use_average_pool == True:
                num_inputs_actor = attn_size
                if self._use_same_dim:            
                    num_inputs_critic = attn_size + obs_shape[-1][1]
                else:
                    num_inputs_critic = attn_size 
            else:
                num_inputs = 0
                split_shape = obs_shape[1:]
                for i in range(len(split_shape)):
                    num_inputs += split_shape[i][0]
                num_inputs_actor = num_inputs * attn_size
                if self._use_same_dim:
                    num_inputs_critic = num_inputs * attn_size
                else:
                    num_inputs_critic = num_agents * attn_size
        else:
            num_inputs_actor = obs_shape[0]
            num_inputs_critic = share_obs_dim
            
        if self._use_orthogonal:
            init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        else:
            init_ = lambda m: init(m, nn.init.xavier_uniform_, lambda x: nn.init.constant_(x, 0), gain = nn.init.calculate_gain('tanh'))
        
        if self._use_ReLU:
            active_func = nn.ReLU()
        else:
            active_func = nn.Tanh()
        
        if self._layer_N == 1:
            self.actor = nn.Sequential(
                init_(nn.Linear(num_inputs_actor, hidden_size)), active_func,
                init_(nn.Linear(hidden_size, hidden_size)), active_func)
    
            self.critic = nn.Sequential(
                init_(nn.Linear(num_inputs_critic, hidden_size)), active_func,
                init_(nn.Linear(hidden_size, hidden_size)), active_func)
        elif self._layer_N == 2:
            self.actor = nn.Sequential(
                init_(nn.Linear(num_inputs_actor, hidden_size)), active_func,
                init_(nn.Linear(hidden_size, hidden_size)), active_func,
                init_(nn.Linear(hidden_size, hidden_size)), active_func)
    
            self.critic = nn.Sequential(
                init_(nn.Linear(num_inputs_critic, hidden_size)), active_func,
                init_(nn.Linear(hidden_size, hidden_size)), active_func,
                init_(nn.Linear(hidden_size, hidden_size)), active_func)
        else:
            raise NotImplementedError
            
        if self._use_common_layer:
            self.actor = nn.Sequential(
                init_(nn.Linear(num_inputs_actor, hidden_size)), active_func)    
            self.critic = nn.Sequential(
                init_(nn.Linear(num_inputs_critic, hidden_size)), active_func)
            self.common_linear = nn.Sequential(
                init_(nn.Linear(hidden_size, hidden_size)), active_func,
                init_(nn.Linear(hidden_size, hidden_size)), active_func)

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

    def forward(self, agent_id, share_inputs, inputs, rnn_hxs_actor, rnn_hxs_critic, masks):
        x = inputs
        share_x = share_inputs
        
        if self._use_feature_normlization or self._use_feature_popart:
            x = self.actor_norm(x)
            share_x = self.critic_norm(share_x)
        if self.is_attn:
            x = self.encoder_actor(x)
            if self._use_same_dim:
                share_x = self.encoder_critic(share_x)
            else:
                share_x = self.encoder_critic(share_x, agent_id)                      
        if self._use_common_layer:
            hidden_actor = self.actor(x)
            hidden_critic = self.critic(share_x)
            hidden_actor = self.common_linear(hidden_actor)
            hidden_critic = self.common_linear(hidden_critic)
            
            if self.is_recurrent or self.is_naive_recurrent:
                hidden_actor, rnn_hxs_actor = self._forward_gru(hidden_actor, rnn_hxs_actor, masks)
                hidden_critic, rnn_hxs_critic = self._forward_gru(hidden_critic, rnn_hxs_critic, masks)
        else:
            hidden_actor = self.actor(x)
            hidden_critic = self.critic(share_x)

            if self.is_recurrent or self.is_naive_recurrent:
                hidden_actor, rnn_hxs_actor = self._forward_gru(hidden_actor, rnn_hxs_actor, masks)
                hidden_critic, rnn_hxs_critic = self._forward_gru_critic(hidden_critic, rnn_hxs_critic, masks)  
                
        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs_actor, rnn_hxs_critic

# region amigo

class Policy_teacher(nn.Module):
    def __init__(self, base = None, actor_base=None, critic_base=None, base_kwargs=None, device=torch.device("cpu")):
        super(Policy_teacher, self).__init__()
        self.mixed_obs = False
        self.mixed_action = False
        self.multi_discrete = False
        self.device = device
        if base_kwargs is None:
            base_kwargs = {}
        
        self.actor_base = actor_base
        self.critic_base = critic_base

    def act(self, inputs, deterministic=False):
        inputs = inputs.to(self.device)
        
        dist = self.actor_base(inputs)
        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()
        action_log_probs = dist.log_probs(action)
        action_out = action
        action_log_probs_out = action_log_probs 
        value = self.critic_base(inputs)       
        
        return value, action_out, action_log_probs_out

    def get_value(self, inputs):
    
        inputs = inputs.to(self.device)
        
        value = self.critic_base(inputs)  
        
        return value

    def evaluate_actions(self, inputs, action):
    
        inputs = inputs.to(self.device)
        action = action.to(self.device)
        dist = self.actor_base(inputs)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy()
        action_log_probs_out = action_log_probs
        dist_entropy_out = dist_entropy.mean()
        value = self.critic_base(inputs) 

        return value, action_log_probs_out, dist_entropy_out

# sl
class ATTBase_actor_teacher_sl(NNBase):
    def __init__(self, num_inputs, action_space, num_agents, hidden_size=64):
        super(ATTBase_actor_teacher_sl, self).__init__(num_inputs, num_agents)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = ObsEncoder_teacher_sl(num_inputs=num_inputs, hidden_size=hidden_size)
        num_actions = action_space          
        self.dist = DiagGaussian(hidden_size, num_actions)

    def forward(self, inputs):
        hidden_actor = self.actor(inputs)
        dist = self.dist(hidden_actor)
        return dist
        # return action_out, action_log_probs_out, dist_entropy_out

class ATTBase_critic_teacher_sl(NNBase):
    def __init__(self, num_inputs, num_agents, hidden_size=64):
        super(ATTBase_critic_teacher_sl, self).__init__(num_inputs,num_agents)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.encoder = ObsEncoder_teacher_sl(num_inputs=num_inputs, hidden_size=hidden_size)

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

    def forward(self, inputs):
        vector_embedding = self.encoder(inputs)
        value = self.critic_linear(vector_embedding)

        return value

class ObsEncoder_teacher_sl(nn.Module):
    def __init__(self, num_inputs, hidden_size=100):
        super(ObsEncoder_teacher_sl, self).__init__()
        
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))
        self.encoder = nn.Sequential(
                            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(), nn.LayerNorm(hidden_size),
                            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(), nn.LayerNorm(hidden_size))


    def forward(self, inputs):
        vector_embedding = self.encoder(inputs)
        return vector_embedding

# sp
class ATTBase_actor_student(NNBase):
    def __init__(self, num_inputs, action_space, agent_num, recurrent=False, assign_id=False, hidden_size=64):
        super(ATTBase_actor_student, self).__init__(num_inputs, agent_num)
        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.agent_num = agent_num
        self.actor = ObsEncoder_student(hidden_size=hidden_size)
        num_actions = action_space.n            
        self.dist = Categorical(hidden_size, num_actions)

    def forward(self, inputs, agent_num):
        """
        inputs: [batch_size, obs_dim]
        """
        hidden_actor = self.actor(inputs, agent_num)
        dist = self.dist(hidden_actor, None)
        return dist
        # return action_out, action_log_probs_out, dist_entropy_out

class ATTBase_critic_student(NNBase):
    def __init__(self, num_inputs, agent_num, recurrent=False, assign_id=False, hidden_size=64):
        super(ATTBase_critic_student, self).__init__(num_inputs, agent_num)
        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.agent_num = agent_num
        # self.encoder = ObsEncoder(hidden_size=hidden_size)
        self.encoder = ObsEncoder_student(hidden_size=hidden_size)

        self.correlation_mat = nn.Parameter(torch.FloatTensor(hidden_size,hidden_size),requires_grad=True)
        nn.init.orthogonal_(self.correlation_mat.data, gain=1)

        self.critic_linear = nn.Sequential(
                init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
                nn.LayerNorm(hidden_size),
                init_(nn.Linear(hidden_size, 1)))

    def forward(self, share_inputs, inputs, agent_num, rnn_hxs, masks):
        """
        share_inputs: [batch_size, obs_dim*agent_num]
        inputs: [batch_size, obs_dim]
        """
        batch_size = inputs.shape[0]
        obs_dim = inputs.shape[-1]
        f_ii = self.encoder(inputs, agent_num)
        obs_beta_ij = torch.matmul(f_ii.view(batch_size,1,-1), self.correlation_mat) # (batch,1,hidden_size)
        
        # 矩阵f_ij
        f_ij = self.encoder(share_inputs.reshape(-1,obs_dim),agent_num)
        obs_encoder = f_ij.reshape(batch_size,agent_num,-1) # (batch_size, nagents, hidden_size)
        
        beta = torch.matmul(obs_beta_ij, obs_encoder.permute(0,2,1)).squeeze(1) # (batch_size,nagents)
        alpha = F.softmax(beta,dim = 1).unsqueeze(2) # (batch_size,nagents,1)
        vi = torch.mul(alpha,obs_encoder)
        vi = torch.sum(vi,dim = 1)
        value = self.critic_linear(vi)

        return value, rnn_hxs, rnn_hxs

class ObsEncoder_student(nn.Module):
    def __init__(self, hidden_size=100):
        super(ObsEncoder_student, self).__init__()
        
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))
        self.self_encoder = nn.Sequential(
                            init_(nn.Linear(4, hidden_size)), nn.Tanh(), nn.LayerNorm(hidden_size))
        self.other_agent_encoder = nn.Sequential(
                            init_(nn.Linear(2, hidden_size)), nn.Tanh(), nn.LayerNorm(hidden_size))
        self.landmark_encoder = nn.Sequential(
                            init_(nn.Linear(3, hidden_size)), nn.Tanh(), nn.LayerNorm(hidden_size))
        self.goal_landmark_encoder = nn.Sequential(
                            init_(nn.Linear(2, hidden_size)), nn.Tanh(), nn.LayerNorm(hidden_size))
        self.agent_correlation_mat = nn.Parameter(torch.FloatTensor(hidden_size,hidden_size),requires_grad=True)
        nn.init.orthogonal_(self.agent_correlation_mat.data, gain=1)
        self.landmark_correlation_mat = nn.Parameter(torch.FloatTensor(hidden_size,hidden_size),requires_grad=True)
        nn.init.orthogonal_(self.landmark_correlation_mat.data, gain=1)
        self.goal_landmark_correlation_mat = nn.Parameter(torch.FloatTensor(hidden_size,hidden_size),requires_grad=True)
        nn.init.orthogonal_(self.goal_landmark_correlation_mat.data, gain=1)
        self.fc = nn.Sequential(
                    init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
                    nn.LayerNorm(hidden_size)
                    )
        self.encoder_linear_goal = nn.Sequential(
                            init_(nn.Linear(hidden_size * 4, hidden_size)), nn.Tanh(),
                            nn.LayerNorm(hidden_size),
                            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
                            nn.LayerNorm(hidden_size)
                            )

    # agent_num需要手动设置一下
    def forward(self, inputs, agent_num):
        batch_size = inputs.shape[0]
        obs_dim = inputs.shape[-1]
        landmark_num = agent_num
        self_emb = self.self_encoder(inputs[:, :4])
        other_agent_emb = []
        beta_agent = []
        landmark_emb = []
        beta_landmark = []

        # goal obs
        goal_landmark_emb = []
        beta_goal_landmark = []

        agent_beta_ij = torch.matmul(self_emb.view(batch_size,1,-1), self.agent_correlation_mat)
        landmark_beta_ij = torch.matmul(self_emb.view(batch_size,1,-1), self.landmark_correlation_mat)
        goal_landmark_beta_ij = torch.matmul(self_emb.view(batch_size,1,-1), self.landmark_correlation_mat)

        for i in range(agent_num - 1):
            other_agent_emb.append(inputs[:, 4+3*landmark_num+2*i:4+3*landmark_num+2*(i+1)])
        for i in range(landmark_num):
            landmark_emb.append(inputs[:, 4+3*i:4+3*(i+1)])
        other_agent_emb = torch.stack(other_agent_emb,dim = 1)    #(batch_size,n_agents-1,eb_dim)
        other_agent_emb = self.other_agent_encoder(other_agent_emb)
        beta_agent = torch.matmul(agent_beta_ij, other_agent_emb.permute(0,2,1)).squeeze(1)
        landmark_emb = torch.stack(landmark_emb,dim = 1)    #(batch_size,n_agents-1,eb_dim)
        landmark_emb = self.landmark_encoder(landmark_emb)
        beta_landmark = torch.matmul(landmark_beta_ij, landmark_emb.permute(0,2,1)).squeeze(1)
        alpha_agent = F.softmax(beta_agent,dim = 1).unsqueeze(2)   
        alpha_landmark = F.softmax(beta_landmark,dim = 1).unsqueeze(2)
        other_agent_vi = torch.mul(alpha_agent,other_agent_emb)
        other_agent_vi = torch.sum(other_agent_vi,dim=1)
        landmark_vi = torch.mul(alpha_landmark,landmark_emb)
        landmark_vi = torch.sum(landmark_vi,dim=1)
        
        # goal obs, goal landmark 2*i, front 4+3*landmark+2*agent
        for i in range(landmark_num):
            goal_landmark_emb.append(inputs[:, 4+3*landmark_num+2*(agent_num-1)+2*i:4+3*landmark_num+2*(agent_num-1)+2*(i+1)])
        goal_landmark_emb = torch.stack(goal_landmark_emb,dim = 1)    #(batch_size,n_agents-1,eb_dim)
        goal_landmark_emb = self.goal_landmark_encoder(goal_landmark_emb)
        beta_goal_landmark = torch.matmul(goal_landmark_beta_ij, goal_landmark_emb.permute(0,2,1)).squeeze(1)
        alpha_goal_landmark = F.softmax(beta_goal_landmark,dim = 1).unsqueeze(2)
        goal_landmark_vi = torch.mul(alpha_goal_landmark,goal_landmark_emb)
        goal_landmark_vi = torch.sum(goal_landmark_vi,dim=1)

        gi = self.fc(self_emb)
        f = self.encoder_linear_goal(torch.cat([gi, other_agent_vi, landmark_vi, goal_landmark_vi], dim=1))
        return f

class ATTBase_actor_teacher(NNBase):
    def __init__(self, num_inputs, action_space, num_agents, hidden_size=64):
        super(ATTBase_actor_teacher, self).__init__(num_inputs, num_agents)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = ObsEncoder_teacher(num_agents=num_agents, hidden_size=hidden_size)
        num_actions = action_space          
        self.dist = DiagGaussian(hidden_size, num_actions)

    def forward(self, inputs):
        hidden_actor = self.actor(inputs)
        dist = self.dist(hidden_actor)
        return dist
        # return action_out, action_log_probs_out, dist_entropy_out

class ATTBase_critic_teacher(NNBase):
    def __init__(self, num_inputs, num_agents, hidden_size=64):
        super(ATTBase_critic_teacher, self).__init__(num_inputs,num_agents)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.encoder = ObsEncoder_teacher(num_agents=num_agents, hidden_size=hidden_size)

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

    def forward(self, inputs):
        vector_embedding = self.encoder(inputs)
        value = self.critic_linear(vector_embedding)

        return value

class ObsEncoder_teacher(nn.Module):
    def __init__(self, num_agents, hidden_size=100):
        super(ObsEncoder_teacher, self).__init__()
        
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))
        self.encoder = nn.Sequential(
                            init_(nn.Linear(num_agents * 4, hidden_size)), nn.Tanh(), nn.LayerNorm(hidden_size),
                            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(), nn.LayerNorm(hidden_size))


    def forward(self, inputs):
        vector_embedding = self.encoder(inputs)
        return vector_embedding

# pb 
class ATTBase_actor_student_pb(NNBase):
    def __init__(self, num_inputs, action_space, agent_num, recurrent=False, assign_id=False, hidden_size=64):
        super(ATTBase_actor_student_pb, self).__init__(num_inputs, agent_num)
        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.agent_num = agent_num
        self.actor = ObsEncoder_student_pb(hidden_size=hidden_size)
        num_actions = action_space.n            
        self.dist = Categorical(hidden_size, num_actions)

    def forward(self, inputs, agent_num, box_num):
        """
        share_inputs: [batch_size, obs_dim*agent_num]
        inputs: [batch_size, obs_dim]
        """
        hidden_actor = self.actor(inputs, agent_num, box_num, box_num)
        dist = self.dist(hidden_actor, None)
        return dist

class ATTBase_critic_student_pb(NNBase):
    def __init__(self, num_inputs, agent_num, recurrent=False, assign_id=False, hidden_size=64):
        super(ATTBase_critic_student_pb, self).__init__(num_inputs, agent_num)
        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.agent_num = agent_num
        self.box_num = agent_num
        self.encoder = ObsEncoder_student_pb(hidden_size=hidden_size)

        self.correlation_mat = nn.Parameter(torch.FloatTensor(hidden_size,hidden_size),requires_grad=True)
        nn.init.orthogonal_(self.correlation_mat.data, gain=1)

        self.critic_linear = nn.Sequential(
                init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
                nn.LayerNorm(hidden_size),
                init_(nn.Linear(hidden_size, 1)))

    def forward(self, share_inputs, inputs, agent_num, box_num, rnn_hxs, masks):
        """
        share_inputs: [batch_size, obs_dim*agent_num]
        inputs: [batch_size, obs_dim]
        """
        batch_size = inputs.shape[0]
        obs_dim = inputs.shape[-1]
        f_ii = self.encoder(inputs, agent_num, box_num, box_num)
        obs_beta_ij = torch.matmul(f_ii.view(batch_size,1,-1), self.correlation_mat) # (batch,1,hidden_size)
        
        # 矩阵f_ij
        f_ij = self.encoder(share_inputs.reshape(-1,obs_dim),agent_num, box_num,box_num)
        obs_encoder = f_ij.reshape(batch_size,agent_num,-1) # (batch_size, nagents, hidden_size)
              
        beta = torch.matmul(obs_beta_ij, obs_encoder.permute(0,2,1)).squeeze(1) # (batch_size,nagents)
        alpha = F.softmax(beta,dim = 1).unsqueeze(2) # (batch_size,nagents,1)
        vi = torch.mul(alpha,obs_encoder)
        vi = torch.sum(vi,dim = 1)
        value = self.critic_linear(vi)

        return value, rnn_hxs, rnn_hxs

class ObsEncoder_student_pb(nn.Module):
    def __init__(self, hidden_size=100):
        super(ObsEncoder_student_pb, self).__init__()
        
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))
        self.self_encoder = nn.Sequential(
                            init_(nn.Linear(4, hidden_size)), nn.Tanh(), nn.LayerNorm(hidden_size))
        self.landmark_encoder = nn.Sequential(
                            init_(nn.Linear(3, hidden_size)), nn.Tanh(), nn.LayerNorm(hidden_size))
        self.adv_encoder = nn.Sequential(
                            init_(nn.Linear(2, hidden_size)), nn.Tanh(), nn.LayerNorm(hidden_size))
        self.good_encoder = nn.Sequential(
                            init_(nn.Linear(2, hidden_size)), nn.Tanh(), nn.LayerNorm(hidden_size))
        self.goal_landmark_encoder = nn.Sequential(
                            init_(nn.Linear(2, hidden_size)), nn.Tanh(), nn.LayerNorm(hidden_size))

        self.adv_correlation_mat = nn.Parameter(torch.FloatTensor(hidden_size,hidden_size),requires_grad=True)
        nn.init.orthogonal_(self.adv_correlation_mat.data, gain=1)
        self.good_correlation_mat = nn.Parameter(torch.FloatTensor(hidden_size,hidden_size),requires_grad=True)
        nn.init.orthogonal_(self.good_correlation_mat.data, gain=1)
        self.landmark_correlation_mat = nn.Parameter(torch.FloatTensor(hidden_size,hidden_size),requires_grad=True)
        nn.init.orthogonal_(self.landmark_correlation_mat.data, gain=1)
        self.goal_landmark_correlation_mat = nn.Parameter(torch.FloatTensor(hidden_size,hidden_size),requires_grad=True)
        nn.init.orthogonal_(self.goal_landmark_correlation_mat.data, gain=1)
        self.fc = nn.Sequential(
                    init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(), nn.LayerNorm(hidden_size))
        self.encoder_linear_goal = nn.Sequential(
                            init_(nn.Linear(hidden_size * 5, hidden_size)), nn.Tanh(),
                            nn.LayerNorm(hidden_size),
                            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
                            nn.LayerNorm(hidden_size))

    # agent_num需要手动设置一下
    def forward(self, inputs, adv_num, good_num, landmark_num):
        batch_size = inputs.shape[0]
        obs_dim = inputs.shape[-1]
        emb_self = self.self_encoder(inputs[:, :4])
      
        emb_adv = []
        beta_adv = []
        emb_good = []
        beta_good = []
        emb_landmark = []
        beta_landmark = []
        # goal obs
        goal_landmark_emb = []
        beta_goal_landmark = []

        beta_adv_ij = torch.matmul(emb_self.view(batch_size,1,-1), self.adv_correlation_mat)
        beta_good_ij = torch.matmul(emb_self.view(batch_size,1,-1), self.good_correlation_mat)
        beta_landmark_ij = torch.matmul(emb_self.view(batch_size,1,-1), self.landmark_correlation_mat) 
        goal_landmark_beta_ij = torch.matmul(emb_self.view(batch_size,1,-1), self.landmark_correlation_mat)
        for i in range(adv_num-1):
            emb_adv.append(inputs[:, 4+2*i:4+2*(i+1)])
        good_offset = 4 + 2*(adv_num-1)
        for i in range(good_num):
            emb_good.append(inputs[:, good_offset+2*i:good_offset+2*(i+1)])
        landmark_offset = 4 + 2*(adv_num-1) + 2*good_num
        for i in range(landmark_num):
            emb_landmark.append(inputs[:, landmark_offset+3*i:landmark_offset+3*(i+1)])

        emb_adv = torch.stack(emb_adv,dim = 1)    #(batch_size,n_agents-1,eb_dim)
        emb_adv = self.adv_encoder(emb_adv)
        beta_adv = torch.matmul(beta_adv_ij, emb_adv.permute(0,2,1)).squeeze(1)

        emb_good = torch.stack(emb_good,dim = 1)    #(batch_size,n_agents-1,eb_dim)
        emb_good = self.good_encoder(emb_good)
        beta_good = torch.matmul(beta_good_ij, emb_good.permute(0,2,1)).squeeze(1)

        emb_landmark = torch.stack(emb_landmark,dim = 1)    #(batch_size,n_agents-1,eb_dim)
        emb_landmark = self.landmark_encoder(emb_landmark)
        beta_landmark = torch.matmul(beta_landmark_ij, emb_landmark.permute(0,2,1)).squeeze(1)

        alpha_adv = F.softmax(beta_adv,dim = 1).unsqueeze(2)   
        alpha_good = F.softmax(beta_good,dim = 1).unsqueeze(2)   
        alpha_landmark = F.softmax(beta_landmark,dim = 1).unsqueeze(2)
        adv_vi = torch.mul(alpha_adv,emb_adv)
        adv_vi = torch.sum(adv_vi,dim=1)
        good_vi = torch.mul(alpha_good,emb_good)
        good_vi = torch.sum(good_vi,dim=1)
        landmark_vi = torch.mul(alpha_landmark,emb_landmark)
        landmark_vi = torch.sum(landmark_vi,dim=1)

        # goal obs, goal landmark 2*i, front 4+3*landmark+2*(agent-1)+2*ball
        for i in range(landmark_num):
            goal_landmark_emb.append(inputs[:, 4+3*landmark_num+2*(adv_num-1)+2*good_num+2*i:4+3*landmark_num+2*(adv_num-1)+2*good_num+2*(i+1)])
        goal_landmark_emb = torch.stack(goal_landmark_emb,dim = 1)    #(batch_size,n_agents-1,eb_dim)
        goal_landmark_emb = self.goal_landmark_encoder(goal_landmark_emb)
        beta_goal_landmark = torch.matmul(goal_landmark_beta_ij, goal_landmark_emb.permute(0,2,1)).squeeze(1)
        alpha_goal_landmark = F.softmax(beta_goal_landmark,dim = 1).unsqueeze(2)
        goal_landmark_vi = torch.mul(alpha_goal_landmark,goal_landmark_emb)
        goal_landmark_vi = torch.sum(goal_landmark_vi,dim=1)

        gi = self.fc(emb_self)
        f = self.encoder_linear_goal(torch.cat([gi, adv_vi, good_vi, landmark_vi, goal_landmark_vi], dim=1))
        return f

class ATTBase_actor_teacher_pb(NNBase):
    def __init__(self, num_inputs, action_space, num_agents, hidden_size=64):
        super(ATTBase_actor_teacher_pb, self).__init__(num_inputs, num_agents)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = ObsEncoder_teacher_pb(num_agents=num_agents,hidden_size=hidden_size)
        num_actions = action_space          
        self.dist = DiagGaussian(hidden_size, num_actions)

    def forward(self, inputs):
        hidden_actor = self.actor(inputs)
        dist = self.dist(hidden_actor)
        return dist
        # return action_out, action_log_probs_out, dist_entropy_out

class ATTBase_critic_teacher_pb(NNBase):
    def __init__(self, num_inputs, num_agents, hidden_size=64):
        super(ATTBase_critic_teacher_pb, self).__init__(num_inputs,num_agents)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.encoder = ObsEncoder_teacher_pb(num_agents=num_agents,hidden_size=hidden_size)

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

    def forward(self, inputs):
        vector_embedding = self.encoder(inputs)
        value = self.critic_linear(vector_embedding)

        return value

class ObsEncoder_teacher_pb(nn.Module):
    def __init__(self, num_agents, hidden_size=100):
        super(ObsEncoder_teacher_pb, self).__init__()
        
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))
        self.encoder = nn.Sequential(
                            init_(nn.Linear(num_agents * 6, hidden_size)), nn.Tanh(), nn.LayerNorm(hidden_size),
                            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(), nn.LayerNorm(hidden_size))


    # agent_num需要手动设置一下
    def forward(self, inputs):
        vector_embedding = self.encoder(inputs)
        return vector_embedding

# end region

class ATTBase_actor_dist_pb_add(NNBase):
    def __init__(self, num_inputs, action_space, agent_num, recurrent=False, hidden_size=64):
        super(ATTBase_actor_dist_pb_add, self).__init__(num_inputs, agent_num)
        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.agent_num = agent_num
        self.actor = ObsEncoder_pb(hidden_size=hidden_size)
        num_actions = action_space.n            
        self.dist = Categorical(hidden_size, num_actions)
    
    def forward(self, inputs, agent_num):
        """
        share_inputs: [batch_size, obs_dim*agent_num]
        inputs: [batch_size, obs_dim]
        """
        hidden_actor = self.actor(inputs, agent_num)
        dist = self.dist(hidden_actor, None)
        return dist

class ATTBase_critic_pb_add(NNBase):
    def __init__(self, num_inputs, agent_num, recurrent=False, hidden_size=64):
        super(ATTBase_critic_pb_add, self).__init__(num_inputs, agent_num)
        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.agent_num = agent_num
        self.encoder = ObsEncoder_pb(hidden_size=hidden_size)

        self.correlation_mat = nn.Parameter(torch.FloatTensor(hidden_size,hidden_size),requires_grad=True)
        nn.init.orthogonal_(self.correlation_mat.data, gain=1)

        self.critic_linear = nn.Sequential(
                init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
                nn.LayerNorm(hidden_size),
                init_(nn.Linear(hidden_size, 1)))

    def forward(self, share_inputs, inputs, agent_num, rnn_hxs, masks):
        """
        share_inputs: [batch_size, obs_dim*agent_num]
        inputs: [batch_size, obs_dim]
        """
        batch_size = inputs.shape[0]
        obs_dim = inputs.shape[-1]
        f_ii = self.encoder(inputs, agent_num)
        obs_beta_ij = torch.matmul(f_ii.view(batch_size,1,-1), self.correlation_mat) # (batch,1,hidden_size)
        
        # 矩阵f_ij
        f_ij = self.encoder(share_inputs.reshape(-1,obs_dim),agent_num)
        obs_encoder = f_ij.reshape(batch_size,agent_num,-1) # (batch_size, nagents, hidden_size)
              
        beta = torch.matmul(obs_beta_ij, obs_encoder.permute(0,2,1)).squeeze(1) # (batch_size,nagents)
        alpha = F.softmax(beta,dim = 1).unsqueeze(2) # (batch_size,nagents,1)
        vi = torch.mul(alpha,obs_encoder)
        vi = torch.sum(vi,dim = 1)
        value = self.critic_linear(vi)

        return value, rnn_hxs, rnn_hxs

class ATTBase_actor(NNBase):
    def __init__(self, num_inputs, action_space, agent_num, model_name, recurrent=False, hidden_size=64):
        super(ATTBase_actor, self).__init__(num_inputs, agent_num)
        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.agent_num = agent_num
        if model_name == 'simple_spread' or 'hard_spread':
            self.actor = ObsEncoder_sp(hidden_size=hidden_size)
        elif model_name == 'push_ball':
            self.actor = ObsEncoder_pb(hidden_size=hidden_size)

        num_actions = action_space.n            
        self.dist = Categorical(hidden_size, num_actions)

    def forward(self, inputs, agent_num):
        """
        inputs: [batch_size, obs_dim]
        """
        hidden_actor = self.actor(inputs, agent_num)
        dist = self.dist(hidden_actor, None)
        return dist

class ATTBase_critic(NNBase):
    def __init__(self, num_inputs, agent_num, model_name, recurrent=False, hidden_size=64):
        super(ATTBase_critic, self).__init__(num_inputs, agent_num)
        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.agent_num = agent_num
        if model_name == 'simple_spread' or 'hard_spread':
            self.encoder = ObsEncoder_sp(hidden_size=hidden_size)
        elif model_name == 'push_ball':
            self.encoder = ObsEncoder_pb(hidden_size=hidden_size)

        self.correlation_mat = nn.Parameter(torch.FloatTensor(hidden_size,hidden_size),requires_grad=True)
        nn.init.orthogonal_(self.correlation_mat.data, gain=1)

        self.critic_linear = nn.Sequential(
                init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
                nn.LayerNorm(hidden_size),
                init_(nn.Linear(hidden_size, 1)))

    def forward(self, share_inputs, inputs, agent_num, rnn_hxs, masks):
        """
        share_inputs: [batch_size, obs_dim*agent_num]
        inputs: [batch_size, obs_dim]
        """
        batch_size = inputs.shape[0]
        obs_dim = inputs.shape[-1]
        f_ii = self.encoder(inputs, agent_num)
        obs_beta_ij = torch.matmul(f_ii.view(batch_size,1,-1), self.correlation_mat) # (batch,1,hidden_size)
        
        # 矩阵f_ij
        f_ij = self.encoder(share_inputs.reshape(-1,obs_dim),agent_num)
        obs_encoder = f_ij.reshape(batch_size,agent_num,-1) # (batch_size, nagents, hidden_size)
        
        beta = torch.matmul(obs_beta_ij, obs_encoder.permute(0,2,1)).squeeze(1) # (batch_size,nagents)
        alpha = F.softmax(beta,dim = 1).unsqueeze(2) # (batch_size,nagents,1)
        vi = torch.mul(alpha,obs_encoder)
        vi = torch.sum(vi,dim = 1)
        value = self.critic_linear(vi)

        return value, rnn_hxs, rnn_hxs

class ATTBase_actor_sl(NNBase):
    def __init__(self, num_inputs, action_space, agent_num, role, recurrent=False, assign_id=False, hidden_size=64):
        super(ATTBase_actor_sl, self).__init__(num_inputs, agent_num)
        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        if role == 'listener':
            self.actor = ObsEncoder_listener(num_inputs=num_inputs, hidden_size=hidden_size)
        elif role == 'speaker':
            self.actor = ObsEncoder_speaker(hidden_size=hidden_size)
        num_actions = action_space.n            
        self.dist = Categorical(hidden_size, num_actions)

    def forward(self, inputs, agent_num, landmark_num):
        """
        inputs: [batch_size, obs_dim]
        """
        hidden_actor = self.actor(inputs, agent_num, landmark_num)
        dist = self.dist(hidden_actor, None)
        return dist

class ATTBase_critic_sl(nn.Module):
    def __init__(self, agent_num, obs_role_dim, recurrent=False, assign_id=False, hidden_size=64):
        super(ATTBase_critic_sl, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.agent_num = agent_num
        self.encoder = {}
        self.encoder_speaker = ObsEncoder_speaker(hidden_size=hidden_size)
        self.encoder_listener = ObsEncoder_listener(obs_role_dim['listener'], hidden_size=hidden_size)
        self.obs_role_dim = obs_role_dim

        self.correlation_mat = nn.Parameter(torch.FloatTensor(hidden_size,hidden_size),requires_grad=True)
        nn.init.orthogonal_(self.correlation_mat.data, gain=1)

        self.critic_linear = nn.Sequential(
                init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
                nn.LayerNorm(hidden_size),
                init_(nn.Linear(hidden_size, 1)))

    def forward(self, share_inputs, inputs, role, agent_num, landmark_num, rnn_hxs, masks):
        """
        share_inputs: [batch_size, obs_dim*agent_num]
        inputs: [batch_size, obs_dim]
        """
        batch_size = inputs.shape[0]
        obs_dim = inputs.shape[-1]
        if role == 'speaker':
            f_ii = self.encoder_speaker(inputs, agent_num, landmark_num)
        elif role == 'listener':
            f_ii = self.encoder_listener(inputs, agent_num, landmark_num)
        obs_beta_ij = torch.matmul(f_ii.view(batch_size,1,-1), self.correlation_mat) # (batch,1,hidden_size)
        
        # 矩阵f_ij
        f_ij = []
        f_ij.append(self.encoder_speaker(share_inputs[:,:self.obs_role_dim['speaker']],agent_num, landmark_num))
        f_ij.append(self.encoder_listener(share_inputs[:,self.obs_role_dim['speaker']:],agent_num, landmark_num))
        obs_encoder = torch.stack(f_ij,dim = 1) # (batch_size, nagents, hidden_size)
        
        beta = torch.matmul(obs_beta_ij, obs_encoder.permute(0,2,1)).squeeze(1) # (batch_size,nagents)
        alpha = F.softmax(beta,dim = 1).unsqueeze(2) # (batch_size,nagents,1)
        vi = torch.mul(alpha,obs_encoder)
        vi = torch.sum(vi,dim = 1)
        value = self.critic_linear(vi)

        return value, rnn_hxs, rnn_hxs

class ObsEncoder_sp(nn.Module):
    def __init__(self, hidden_size=100):
        super(ObsEncoder_sp, self).__init__()
        
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))
        self.self_encoder = nn.Sequential(
                            init_(nn.Linear(4, hidden_size)), nn.Tanh(), nn.LayerNorm(hidden_size))
        self.other_agent_encoder = nn.Sequential(
                            init_(nn.Linear(2, hidden_size)), nn.Tanh(), nn.LayerNorm(hidden_size))
        self.landmark_encoder = nn.Sequential(
                            init_(nn.Linear(3, hidden_size)), nn.Tanh(), nn.LayerNorm(hidden_size))
        self.agent_correlation_mat = nn.Parameter(torch.FloatTensor(hidden_size,hidden_size),requires_grad=True)
        nn.init.orthogonal_(self.agent_correlation_mat.data, gain=1)
        self.landmark_correlation_mat = nn.Parameter(torch.FloatTensor(hidden_size,hidden_size),requires_grad=True)
        nn.init.orthogonal_(self.landmark_correlation_mat.data, gain=1)
        self.fc = nn.Sequential(
                    init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
                    nn.LayerNorm(hidden_size)
                    )
        self.encoder_linear = nn.Sequential(
                            init_(nn.Linear(hidden_size * 3, hidden_size)), nn.Tanh(),
                            nn.LayerNorm(hidden_size),
                            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
                            nn.LayerNorm(hidden_size)
                            )

    # agent_num需要手动设置一下
    def forward(self, inputs, agent_num):
        batch_size = inputs.shape[0]
        obs_dim = inputs.shape[-1]
        landmark_num = agent_num
        # landmark_num = int((obs_dim-4)/2)-2*(agent_num-1)
        #landmark_num = int((obs_dim-4-4*(agent_num-1))/3)
        self_emb = self.self_encoder(inputs[:, :4])
        other_agent_emb = []
        beta_agent = []
        landmark_emb = []
        beta_landmark = []
        #start = time.time()

        agent_beta_ij = torch.matmul(self_emb.view(batch_size,1,-1), self.agent_correlation_mat)
        landmark_beta_ij = torch.matmul(self_emb.view(batch_size,1,-1), self.landmark_correlation_mat) 

        for i in range(agent_num - 1):
            other_agent_emb.append(inputs[:, 4+3*landmark_num+2*i:4+3*landmark_num+2*(i+1)])
        for i in range(landmark_num):
            landmark_emb.append(inputs[:, 4+3*i:4+3*(i+1)])
        other_agent_emb = torch.stack(other_agent_emb,dim = 1)    #(batch_size,n_agents-1,eb_dim)
        other_agent_emb = self.other_agent_encoder(other_agent_emb)
        beta_agent = torch.matmul(agent_beta_ij, other_agent_emb.permute(0,2,1)).squeeze(1)
        landmark_emb = torch.stack(landmark_emb,dim = 1)    #(batch_size,n_agents-1,eb_dim)
        landmark_emb = self.landmark_encoder(landmark_emb)
        beta_landmark = torch.matmul(landmark_beta_ij, landmark_emb.permute(0,2,1)).squeeze(1)
        alpha_agent = F.softmax(beta_agent,dim = 1).unsqueeze(2)   
        alpha_landmark = F.softmax(beta_landmark,dim = 1).unsqueeze(2)
        other_agent_vi = torch.mul(alpha_agent,other_agent_emb)
        other_agent_vi = torch.sum(other_agent_vi,dim=1)
        landmark_vi = torch.mul(alpha_landmark,landmark_emb)
        landmark_vi = torch.sum(landmark_vi,dim=1)
        gi = self.fc(self_emb)
        f = self.encoder_linear(torch.cat([gi, other_agent_vi, landmark_vi], dim=1))
        return f

class ObsEncoder_listener(nn.Module):
    def __init__(self, num_inputs , hidden_size=100):
        super(ObsEncoder_listener, self).__init__()
        
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))
        self.encoder_linear = nn.Sequential(
                            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
                            nn.LayerNorm(hidden_size)
                            )

    def forward(self, inputs, agent_num, landmark_num):
        f = self.encoder_linear(inputs)
        return f

class ObsEncoder_speaker(nn.Module):
    def __init__(self, hidden_size=100):
        super(ObsEncoder_speaker, self).__init__()
        
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))
        self.encoder_linear = nn.Sequential(
                            init_(nn.Linear(3, hidden_size)), nn.Tanh(),
                            nn.LayerNorm(hidden_size)
                            )

    def forward(self, inputs, agent_num, landmark_num):
        f = self.encoder_linear(inputs)
        return f

class ObsEncoder_pb(nn.Module): # push ball
    def __init__(self, hidden_size=100):
        super(ObsEncoder_pb, self).__init__()
        
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))
        self.self_encoder = nn.Sequential(
                            init_(nn.Linear(4, hidden_size)), nn.Tanh(), nn.LayerNorm(hidden_size))
        self.landmark_encoder = nn.Sequential(
                            init_(nn.Linear(3, hidden_size)), nn.Tanh(), nn.LayerNorm(hidden_size))
        self.adv_encoder = nn.Sequential(
                            init_(nn.Linear(2, hidden_size)), nn.Tanh(), nn.LayerNorm(hidden_size))
        self.good_encoder = nn.Sequential(
                            init_(nn.Linear(2, hidden_size)), nn.Tanh(), nn.LayerNorm(hidden_size))

        self.adv_correlation_mat = nn.Parameter(torch.FloatTensor(hidden_size,hidden_size),requires_grad=True)
        nn.init.orthogonal_(self.adv_correlation_mat.data, gain=1)
        self.good_correlation_mat = nn.Parameter(torch.FloatTensor(hidden_size,hidden_size),requires_grad=True)
        nn.init.orthogonal_(self.good_correlation_mat.data, gain=1)
        self.landmark_correlation_mat = nn.Parameter(torch.FloatTensor(hidden_size,hidden_size),requires_grad=True)
        nn.init.orthogonal_(self.landmark_correlation_mat.data, gain=1)
        self.fc = nn.Sequential(
                    init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(), nn.LayerNorm(hidden_size))
        self.encoder_linear = nn.Sequential(
                            init_(nn.Linear(hidden_size * 4, hidden_size)), nn.Tanh(),
                            nn.LayerNorm(hidden_size),
                            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
                            nn.LayerNorm(hidden_size))

    def forward(self, inputs, agent_num):
        batch_size = inputs.shape[0]
        obs_dim = inputs.shape[-1]
        emb_self = self.self_encoder(inputs[:, :4])
        adv_num = agent_num
        good_num = agent_num
        landmark_num = agent_num
      
        emb_adv = []
        beta_adv = []
        emb_good = []
        beta_good = []
        emb_landmark = []
        beta_landmark = []

        beta_adv_ij = torch.matmul(emb_self.view(batch_size,1,-1), self.adv_correlation_mat)
        beta_good_ij = torch.matmul(emb_self.view(batch_size,1,-1), self.good_correlation_mat)
        beta_landmark_ij = torch.matmul(emb_self.view(batch_size,1,-1), self.landmark_correlation_mat) 
        for i in range(adv_num-1):
            emb_adv.append(inputs[:, 4+2*i:4+2*(i+1)])
        good_offset = 4 + 2*(adv_num-1)
        for i in range(good_num):
            emb_good.append(inputs[:, good_offset+2*i:good_offset+2*(i+1)])
        landmark_offset = 4 + 2*(adv_num-1) + 2*good_num
        for i in range(landmark_num):
            emb_landmark.append(inputs[:, landmark_offset+3*i:landmark_offset+3*(i+1)])

        emb_adv = torch.stack(emb_adv,dim = 1)    #(batch_size,n_agents-1,eb_dim)
        emb_adv = self.adv_encoder(emb_adv)
        beta_adv = torch.matmul(beta_adv_ij, emb_adv.permute(0,2,1)).squeeze(1)

        emb_good = torch.stack(emb_good,dim = 1)    #(batch_size,n_agents-1,eb_dim)
        emb_good = self.good_encoder(emb_good)
        beta_good = torch.matmul(beta_good_ij, emb_good.permute(0,2,1)).squeeze(1)

        emb_landmark = torch.stack(emb_landmark,dim = 1)    #(batch_size,n_agents-1,eb_dim)
        emb_landmark = self.landmark_encoder(emb_landmark)
        beta_landmark = torch.matmul(beta_landmark_ij, emb_landmark.permute(0,2,1)).squeeze(1)

        alpha_adv = F.softmax(beta_adv,dim = 1).unsqueeze(2)   
        alpha_good = F.softmax(beta_good,dim = 1).unsqueeze(2)   
        alpha_landmark = F.softmax(beta_landmark,dim = 1).unsqueeze(2)
        adv_vi = torch.mul(alpha_adv,emb_adv)
        adv_vi = torch.sum(adv_vi,dim=1)
        good_vi = torch.mul(alpha_good,emb_good)
        good_vi = torch.sum(good_vi,dim=1)
        landmark_vi = torch.mul(alpha_landmark,emb_landmark)
        landmark_vi = torch.sum(landmark_vi,dim=1)

        gi = self.fc(emb_self)
        f = self.encoder_linear(torch.cat([gi, adv_vi, good_vi, landmark_vi], dim=1))
        return f

# transformer

class FeedForward(nn.Module):

    def __init__(self, d_model, d_ff=512, dropout = 0.0, use_orthogonal=True, use_ReLU=False):

        super(FeedForward, self).__init__() 
        # We set d_ff as a default to 2048
        if use_orthogonal:
            init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        else:
            init_ = lambda m: init(m, nn.init.xavier_uniform_, lambda x: nn.init.constant_(x, 0))
        
        if use_ReLU:
            active_func = nn.ReLU()
        else:
            active_func = nn.Tanh()
           
        self.linear_1 = nn.Sequential(init_(nn.Linear(d_model, d_ff)), active_func)

        self.dropout = nn.Dropout(dropout)
        self.linear_2 = init_(nn.Linear(d_ff, d_model))
    def forward(self, x):
        x = self.dropout(self.linear_1(x))
        x = self.linear_2(x)
        return x

def ScaledDotProductAttention(q, k, v, d_k, mask=None, dropout=None):
    
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)
        
    output = torch.matmul(scores, v)
    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.0, use_orthogonal=True):
        super(MultiHeadAttention, self).__init__()
        if use_orthogonal:
            init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        else:        
            init_ = lambda m: init(m, nn.init.xavier_uniform_, lambda x: nn.init.constant_(x, 0))
         
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = init_(nn.Linear(d_model, d_model))
        self.v_linear = init_(nn.Linear(d_model, d_model))
        self.k_linear = init_(nn.Linear(d_model, d_model))
        self.dropout = nn.Dropout(dropout)
        self.out = init_(nn.Linear(d_model, d_model))
    
    def forward(self, q, k, v, mask=None):
        
        bs = q.size(0)
        
        # perform linear operation and split into h heads
        
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * h * sl * d_model
       
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        # calculate attention
        scores = ScaledDotProductAttention(q, k, v, self.d_k, mask, self.dropout)
        
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous()\
        .view(bs, -1, self.d_model)
        
        output = self.out(concat)
    
        return output
        
class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout = 0.0, use_orthogonal=True, use_ReLU=False, d_ff = 512, use_FF=False):
        super(EncoderLayer, self).__init__()
        self._use_FF = use_FF
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout, use_orthogonal)
        self.ff = FeedForward(d_model, d_ff, dropout, use_orthogonal, use_ReLU)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2,x2,x2,mask))
        if self._use_FF:
            x2 = self.norm_2(x)
            x = x + self.dropout_2(self.ff(x2))
        return x
        
class SelfEmbedding(nn.Module):
    def __init__(self, split_shape, d_model, use_orthogonal=True, use_ReLU=False):
        super(SelfEmbedding, self).__init__()
        self.split_shape = split_shape
        if use_orthogonal:        
            init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        else:
            init_ = lambda m: init(m, nn.init.xavier_uniform_, lambda x: nn.init.constant_(x, 0), gain = nn.init.calculate_gain('tanh'))
        
        if use_ReLU:
            active_func = nn.ReLU()
        else:
            active_func = nn.Tanh()
            
        for i in range(len(split_shape)):
            if i==(len(split_shape)-1):            
                setattr(self,'fc_'+str(i), nn.Sequential(init_(nn.Linear(split_shape[i][1], d_model)), active_func))
            else:
                setattr(self,'fc_'+str(i), nn.Sequential(init_(nn.Linear(split_shape[i][1], d_model)), active_func))
                  
        
    def forward(self, x, self_idx=-1):
        x = split_obs(x, self.split_shape)
        N = len(x)
        
        x1 = []  
        self_x = x[self_idx]      
        for i in range(N-1):
            K = self.split_shape[i][0]
            L = self.split_shape[i][1]
            for j in range(K):
                #temp = torch.cat((x[i][:, (L*j):(L*j+L)], self_x), dim=-1)
                temp = x[i][:, (L*j):(L*j+L)]
                exec('x1.append(self.fc_{}(temp))'.format(i))
        temp = x[self_idx]
        exec('x1.append(self.fc_{}(temp))'.format(N-1))
        
        out = torch.stack(x1,1)        
                 
        return out, self_x
  
class Embedding(nn.Module):
    def __init__(self, split_shape, d_model, use_orthogonal=True, use_ReLU=False):
        super(Embedding, self).__init__()
        self.split_shape = split_shape
        
        if use_orthogonal:        
            init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        else:
            init_ = lambda m: init(m, nn.init.xavier_uniform_, lambda x: nn.init.constant_(x, 0), gain = nn.init.calculate_gain('tanh'))
        
        if use_ReLU:
            active_func = nn.ReLU()
        else:
            active_func = nn.Tanh()
            
        for i in range(len(split_shape)):
            setattr(self,'fc_'+str(i), nn.Sequential(init_(nn.Linear(split_shape[i][1], d_model)), active_func))
                  
        
    def forward(self, x, self_idx):
        x = split_obs(x, self.split_shape)
        N = len(x)
        
        x1 = []   
        self_x = x[self_idx]     
        for i in range(N):
            K = self.split_shape[i][0]
            L = self.split_shape[i][1]
            for j in range(K):
                temp = x[i][:, (L*j):(L*j+L)]
                exec('x1.append(self.fc_{}(temp))'.format(i))

        out = torch.stack(x1, 1)        
                            
        return out, self_x
   
class Encoder(nn.Module):
    def __init__(self, split_shape, d_model=512, attn_N=2, heads=8, dropout=0.0, use_average_pool=True, use_orthogonal=True, use_ReLU=False):
        super(Encoder, self).__init__()
                                       
        self._attn_N = attn_N
        self._use_average_pool = use_average_pool
        self.catself=False
        if split_shape[0].__class__ == list:           
            self.embedding = Embedding(split_shape, d_model, use_orthogonal, use_ReLU)
        else:
            self.embedding = SelfEmbedding(split_shape[1:], d_model, use_orthogonal, use_ReLU)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout, use_orthogonal, use_ReLU), self._attn_N)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, src, self_idx=-1, mask=None):
        x, self_x = self.embedding(src, self_idx)
        for i in range(self._attn_N):
            x = self.layers[i](x, mask)
        x = self.norm(x)
        if self._use_average_pool:
            x = torch.transpose(x, 1, 2) 
            x = F.avg_pool1d(x, kernel_size=x.size(-1)).view(x.size(0), -1)
            if self.catself:
                x = torch.cat((x, self_x), dim=-1)
        x = x.view(x.size(0), -1)
        return x    
    
