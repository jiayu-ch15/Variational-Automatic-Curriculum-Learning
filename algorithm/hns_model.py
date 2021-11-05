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
    def __init__(self, obs_space, action_space, num_agents, with_PC=True, base=None, base_kwargs=None, device=torch.device("cpu")):
        super(Policy, self).__init__()
        self.mixed_obs = False
        self.mixed_action = False
        if action_space.__class__.__name__ == "MultiDiscrete":
            self.multi_discrete = True
            self.discrete_N = action_space.shape
        else:
            self.multi_discrete = False
        self.with_PC = with_PC
        self.device = device
        self.num_agents = num_agents
        self.is_recurrent = base_kwargs['recurrent']
        self.is_naive_recurrent = base_kwargs['naive_recurrent']
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
        
        self.actor_base = Actor(obs_shape, action_space, num_agents, **base_kwargs)
        self.critic_base = Critic(obs_shape, num_agents, **base_kwargs)

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
        
        # value, actor_features, rnn_hxs_actor, rnn_hxs_critic = self.base(agent_id, share_inputs, inputs, rnn_hxs_actor, rnn_hxs_critic, masks)        
        # dist = self.dist(actor_features, available_actions)

        dist, rnn_hxs_actor = self.actor_base(inputs, rnn_hxs_actor, masks, available_actions)
        value, rnn_hxs_critic  = self.critic_base(agent_id, share_inputs, rnn_hxs_critic, masks)

            
        if self.multi_discrete:
            action_out = []
            action_log_probs_out = []
            for i in range(self.discrete_N):
                
                if deterministic:
                    action = dist[i].mode()
                else:
                    action = dist[i].sample()
                    
                action_log_probs = dist[i].log_probs(action)
                
                action_out.append(action)
                action_log_probs_out.append(action_log_probs)
                
            action_out = torch.cat(action_out,-1)
            action_log_probs_out = torch.sum(torch.cat(action_log_probs_out, -1), -1, keepdim = True)
            
        else:
            if deterministic:
                action = dist.mode()
            else:
                action = dist.sample()
  
            action_log_probs = dist.log_probs(action)
            
            action_out = action
            action_log_probs_out = action_log_probs
        
        return value, action_out, action_log_probs_out, rnn_hxs_actor, rnn_hxs_critic

    def get_value(self, agent_id, share_inputs, inputs, rnn_hxs_actor, rnn_hxs_critic, masks):
    
        share_inputs = share_inputs.to(self.device)
        inputs = inputs.to(self.device)
        rnn_hxs_actor = rnn_hxs_actor.to(self.device)
        rnn_hxs_critic = rnn_hxs_critic.to(self.device)
        masks = masks.to(self.device)
        
        # value, _, rnn_hxs_actor, rnn_hxs_critic = self.base(agent_id, share_inputs, inputs, rnn_hxs_actor, rnn_hxs_critic, masks)
        _, rnn_hxs_actor = self.actor_base(inputs, rnn_hxs_actor, masks, available_actions=None)
        value, rnn_hxs_critic  = self.critic_base(agent_id, share_inputs, rnn_hxs_critic, masks)
        
        return value, rnn_hxs_actor, rnn_hxs_critic

    def evaluate_actions(self, agent_id, share_inputs, inputs, rnn_hxs_actor, rnn_hxs_critic, masks, high_masks, action):
    
        share_inputs = share_inputs.to(self.device)
        inputs = inputs.to(self.device)
        rnn_hxs_actor = rnn_hxs_actor.to(self.device)
        rnn_hxs_critic = rnn_hxs_critic.to(self.device)
        masks = masks.to(self.device)
        high_masks = high_masks.to(self.device)
        action = action.to(self.device)
        # value, actor_features, rnn_hxs_actor, rnn_hxs_critic = self.base(agent_id, share_inputs, inputs, rnn_hxs_actor, rnn_hxs_critic, masks)
        # dist = self.dist(actor_features)
        
        dist, rnn_hxs_actor = self.actor_base(inputs, rnn_hxs_actor, masks, available_actions=None)
        value, rnn_hxs_critic  = self.critic_base(agent_id, share_inputs, rnn_hxs_critic, masks)

        if self.multi_discrete:           
            action = torch.transpose(action,0,1)
            action_log_probs = []
            dist_entropy = []
            for i in range(self.discrete_N):
                action_log_probs.append(dist[i].log_probs(action[i]))
                if high_masks is not None:
                    dist_entropy.append( (dist[i].entropy()*high_masks.squeeze(-1)).sum()/high_masks.sum() )
                else:
                    dist_entropy.append(dist[i].entropy().mean())
                    
            action_log_probs_out = torch.sum(torch.cat(action_log_probs, -1), -1, keepdim = True)
            dist_entropy_out = torch.tensor(dist_entropy).mean()
        else:
            action_log_probs = dist.log_probs(action)
            dist_entropy = (dist.entropy()*high_masks.squeeze(-1)).sum()/high_masks.sum()
            action_log_probs_out = action_log_probs
            dist_entropy_out = dist_entropy

        return value, action_log_probs_out, dist_entropy_out, rnn_hxs_actor, rnn_hxs_critic

#obs_shape, num_agents, naive_recurrent, recurrent, hidden_size, attn, attn_size, attn_N, attn_heads, dropout, use_average_pool, use_common_layer, use_orthogonal
class NNBase(nn.Module):
    def __init__(self, obs_shape, num_agents, naive_recurrent=False, recurrent=False, hidden_size=64,
                 attn=False, attn_only_critic=False, attn_size=512, attn_N=2, attn_heads=8, dropout=0.05, use_average_pool=True, 
                 use_common_layer=False, use_orthogonal=True, use_ReLU=False, use_same_dim=False):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent
        self._naive_recurrent = naive_recurrent
        self._attn = attn
        self._attn_only_critic = attn_only_critic
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
        elif self._attn_only_critic:
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

class Actor(nn.Module):
    def __init__(self, obs_shape, action_space, num_agents, naive_recurrent = False, recurrent=False, hidden_size=64, 
                attn=False, attn_only_critic=False, attn_size=512, attn_N=2, attn_heads=8, dropout=0.05, use_average_pool=True, 
                use_common_layer=False, use_feature_normlization=True, use_feature_popart=True, 
                use_orthogonal=True, layer_N=1, use_ReLU=False, use_same_dim=False):
        super(Actor, self).__init__()

        self._use_common_layer = use_common_layer
        self._use_feature_normlization = use_feature_normlization
        self._use_feature_popart = use_feature_popart
        self._use_orthogonal = use_orthogonal
        self._layer_N = layer_N
        self._use_ReLU = use_ReLU
        self._use_same_dim = use_same_dim
        self._attn = attn
        self._attn_only_critic = attn_only_critic
        self._hidden_size = hidden_size
        self._output_size = hidden_size
        self._recurrent = recurrent
        self._naive_recurrent = naive_recurrent
        self.multi_discrete = False
        self.mixed_action = False
        
        assert (self._use_feature_normlization and self._use_feature_popart) == False, ("--use_feature_normlization and --use_feature_popart can not be set True simultaneously.")

        if self._use_feature_normlization:
            self.actor_norm = nn.LayerNorm(obs_shape[0])
            
        if self._use_feature_popart:
            self.actor_norm = PopArt(obs_shape[0])
            
        if self._attn:           
            if use_average_pool == True:
                num_inputs_actor = attn_size + obs_shape[-1][1]
            else:
                num_inputs = 0
                split_shape = obs_shape[1:]
                for i in range(len(split_shape)):
                    num_inputs += split_shape[i][0]
                num_inputs_actor = num_inputs * attn_size
        else:
            num_inputs_actor = obs_shape[0]
            
        if self._use_orthogonal:
            init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        else:
            init_ = lambda m: init(m, nn.init.xavier_uniform_, lambda x: nn.init.constant_(x, 0), gain = nn.init.calculate_gain('tanh'))
        
        if self._use_ReLU:
            active_func = nn.ReLU()
        else:
            active_func = nn.Tanh()

        # attn embedding
        if self._attn:
            self.encoder_actor = Encoder(obs_shape, attn_size, attn_N, attn_heads, dropout, use_average_pool, use_orthogonal, use_ReLU)
        self.rnn = RNNlayer(inputs_dim=hidden_size, outputs_dim=hidden_size, use_orthogonal=use_orthogonal)

        self.actor = MLPLayer(num_inputs_actor, hidden_size, self._layer_N, self._use_orthogonal, self._use_ReLU)
   
        if self._use_common_layer:
            self.actor = nn.Sequential(
                init_(nn.Linear(num_inputs_actor, hidden_size)), active_func)    
            self.fc_h = nn.Sequential(init_(nn.Linear(hidden_size, hidden_size)), active_func)
            self.common_linear = get_clones(self.fc_h, self._layer_N)
        
        # select dist        
        if action_space.__class__.__name__ == "Discrete":
            num_actions = action_space.n            
            self.dist = Categorical(self._output_size, num_actions)
        elif action_space.__class__.__name__ == "Box":
            num_actions = action_space.shape[0]
            self.dist = DiagGaussian(self._output_size, num_actions)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_actions = action_space.shape[0]
            self.dist = Bernoulli(self._output_size, num_actions)
        elif action_space.__class__.__name__ == "MultiDiscrete":
            self.multi_discrete = True
            self.discrete_N = action_space.shape
            action_size = action_space.high-action_space.low+1
            self.dists = []
            for num_actions in action_size:
                self.dists.append(Categorical(self._output_size, num_actions))
            self.dists = nn.ModuleList(self.dists)
        else:# discrete+continous
            self.mixed_action = True
            continous = action_space[0].shape[0]
            discrete = action_space[1].n
            self.dist = nn.ModuleList([DiagGaussian(self._output_size, continous), Categorical(self.actor._output_size, discrete)])
                
    def forward(self, inputs, rnn_hxs_actor, masks, available_actions=None):
        x = inputs
        
        if self._use_feature_normlization or self._use_feature_popart:
            x = self.actor_norm(x)

        if self._attn:
            x = self.encoder_actor(x)
                            
        if self._use_common_layer:
            hidden_actor = self.actor(x)
            for i in range(self._layer_N):
                hidden_actor = self.common_linear[i](hidden_actor)         
            if self._recurrent or self._naive_recurrent:
                hidden_actor, rnn_hxs_actor = self.rnn(hidden_actor, rnn_hxs_actor, masks)
        else:
            hidden_actor = self.actor(x)
            if self._recurrent or self._naive_recurrent:
                hidden_actor, rnn_hxs_actor = self.rnn(hidden_actor, rnn_hxs_actor, masks) 

        if self.mixed_action:
            dist, action, action_log_probs = [None, None], [None, None], [None, None]
            for i in range(2):
                dist[i] = self.dist[i](hidden_actor, available_actions)
            
        elif self.multi_discrete:
            dist = []
            for i in range(self.discrete_N):
                dist.append(self.dists[i](hidden_actor))
            
        else:
            dist = self.dist(hidden_actor, available_actions)
                
        return dist, rnn_hxs_actor

class Critic(nn.Module):
    def __init__(self, obs_shape, num_agents, naive_recurrent = False, recurrent=False, hidden_size=64, 
                attn=False, attn_only_critic=False, attn_size=512, attn_N=2, attn_heads=8, dropout=0.05, use_average_pool=True, 
                use_common_layer=False, use_feature_normlization=True, use_feature_popart=True, 
                use_orthogonal=True, layer_N=1, use_ReLU=False, use_same_dim=False):
        super(Critic, self).__init__()

        self._use_common_layer = use_common_layer
        self._use_feature_normlization = use_feature_normlization
        self._use_feature_popart = use_feature_popart
        self._use_orthogonal = use_orthogonal
        self._layer_N = layer_N
        self._use_ReLU = use_ReLU
        self._use_same_dim = use_same_dim
        self._attn = attn
        self._attn_only_critic = attn_only_critic
        self._recurrent = recurrent
        self._naive_recurrent = naive_recurrent
        
        assert (self._use_feature_normlization and self._use_feature_popart) == False, ("--use_feature_normlization and --use_feature_popart can not be set True simultaneously.")

        if self._use_same_dim:
            share_obs_dim = obs_shape[0]
        else:
            share_obs_dim = obs_shape[0]*num_agents
        
        if self._use_feature_normlization:
            self.critic_norm = nn.LayerNorm(share_obs_dim)
            
        if self._use_feature_popart:
            self.critic_norm = PopArt(share_obs_dim)
            
        if self._attn:           
            if use_average_pool == True:
                if self._use_same_dim:            
                    num_inputs_critic = attn_size + obs_shape[-1][1]
                else:
                    num_inputs_critic = attn_size 
            else:
                num_inputs = 0
                split_shape = obs_shape[1:]
                for i in range(len(split_shape)):
                    num_inputs += split_shape[i][0]
                if self._use_same_dim:
                    num_inputs_critic = num_inputs * attn_size
                else:
                    num_inputs_critic = num_agents * attn_size
        elif self._attn_only_critic:
            if use_average_pool == True:
                num_inputs_critic = attn_size
            else:
                num_inputs_critic = num_agents * attn_size
        else:
            num_inputs_critic = share_obs_dim
            
        if self._use_orthogonal:
            init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        else:
            init_ = lambda m: init(m, nn.init.xavier_uniform_, lambda x: nn.init.constant_(x, 0), gain = nn.init.calculate_gain('tanh'))
        
        if self._use_ReLU:
            active_func = nn.ReLU()
        else:
            active_func = nn.Tanh()

        if self._attn:
            if self._use_same_dim:
                self.encoder_critic = Encoder(obs_shape, attn_size, attn_N, attn_heads, dropout, use_average_pool, use_orthogonal, use_ReLU)   
            else:
                self.encoder_critic = Encoder([[1,obs_shape[0]]]*num_agents, attn_size, attn_N, attn_heads, dropout, use_average_pool, use_orthogonal, use_ReLU)
        self.rnn = RNNlayer(inputs_dim=hidden_size, outputs_dim=hidden_size, use_orthogonal=use_orthogonal)

        self.critic = MLPLayer(num_inputs_critic, hidden_size, self._layer_N, self._use_orthogonal, self._use_ReLU)
   
        if self._use_common_layer:  
            self.critic = nn.Sequential(
                init_(nn.Linear(num_inputs_critic, hidden_size)), active_func)
            self.fc_h = nn.Sequential(init_(nn.Linear(hidden_size, hidden_size)), active_func)
            self.common_linear = get_clones(self.fc_h, self._layer_N)

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

    def forward(self, agent_id, share_inputs, rnn_hxs_critic, masks):
        share_x = share_inputs
        
        if self._use_feature_normlization or self._use_feature_popart:
            share_x = self.critic_norm(share_x)

        if self._attn:
            if self._use_same_dim:
                share_x = self.encoder_critic(share_x)
            else:
                share_x = self.encoder_critic(share_x, agent_id)
        elif self._attn_only_critic:
            share_x = self.encoder_critic(share_x, agent_id)
                            
        if self._use_common_layer:
            hidden_critic = self.critic(share_x)
            for i in range(self._layer_N):
                hidden_critic = self.common_linear[i](hidden_critic)            
            if self._recurrent or self._naive_recurrent:
                hidden_critic, rnn_hxs_critic = self.rnn(hidden_critic, rnn_hxs_critic, masks)
        else:
            hidden_critic = self.critic(share_x)
            if self._recurrent or self._naive_recurrent:
                hidden_critic, rnn_hxs_critic = self.rnn(hidden_critic, rnn_hxs_critic, masks)  
                
        return self.critic_linear(hidden_critic), rnn_hxs_critic

class MLPBase(NNBase):
    def __init__(self, obs_shape, num_agents, naive_recurrent = False, recurrent=False, hidden_size=64, 
                attn=False, attn_only_critic=False, attn_size=512, attn_N=2, attn_heads=8, dropout=0.05, use_average_pool=True, 
                use_common_layer=False, use_feature_normlization=True, use_feature_popart=True, 
                use_orthogonal=True, layer_N=1, use_ReLU=False, use_same_dim=False):
        super(MLPBase, self).__init__(obs_shape, num_agents, naive_recurrent, recurrent, hidden_size, 
                                      attn, attn_only_critic, attn_size, attn_N, attn_heads, dropout, use_average_pool, 
                                      use_common_layer, use_orthogonal, use_ReLU, use_same_dim)

        self._use_common_layer = use_common_layer
        self._use_feature_normlization = use_feature_normlization
        self._use_feature_popart = use_feature_popart
        self._use_orthogonal = use_orthogonal
        self._layer_N = layer_N
        self._use_ReLU = use_ReLU
        self._use_same_dim = use_same_dim
        self._attn = attn
        self._attn_only_critic = attn_only_critic
        
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
                num_inputs_actor = attn_size + obs_shape[-1][1]
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
        elif self._attn_only_critic:
            num_inputs_actor = obs_shape[0]
            if use_average_pool == True:
                num_inputs_critic = attn_size
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

        self.actor = MLPLayer(num_inputs_actor, hidden_size, self._layer_N, self._use_orthogonal, self._use_ReLU)
        self.critic = MLPLayer(num_inputs_critic, hidden_size, self._layer_N, self._use_orthogonal, self._use_ReLU)
   
        if self._use_common_layer:
            self.actor = nn.Sequential(
                init_(nn.Linear(num_inputs_actor, hidden_size)), active_func)    
            self.critic = nn.Sequential(
                init_(nn.Linear(num_inputs_critic, hidden_size)), active_func)
            self.fc_h = nn.Sequential(init_(nn.Linear(hidden_size, hidden_size)), active_func)
            self.common_linear = get_clones(self.fc_h, self._layer_N)

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
        elif self._attn_only_critic:
            share_x = self.encoder_critic(share_x, agent_id)
                            
        if self._use_common_layer:
            hidden_actor = self.actor(x)
            hidden_critic = self.critic(share_x)
            for i in range(self._layer_N):
                hidden_actor = self.common_linear[i](hidden_actor)
                hidden_critic = self.common_linear[i](hidden_critic)            
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

class MLPBase_PC(NNBase): # for box locking
    def __init__(self, obs_shape, num_agents, naive_recurrent = False, recurrent=False, hidden_size=64, 
                attn=False, attn_only_critic=False, attn_size=512, attn_N=2, attn_heads=8, dropout=0.05, use_average_pool=True, 
                use_common_layer=False, use_feature_normlization=False, use_feature_popart=False, 
                use_orthogonal=True, layer_N=1, use_ReLU=False, use_same_dim=False):
        super(MLPBase_PC, self).__init__(obs_shape, num_agents, naive_recurrent, recurrent, hidden_size, 
                                      attn, attn_only_critic, attn_size, attn_N, attn_heads, dropout, use_average_pool, 
                                      use_common_layer, use_orthogonal, use_ReLU, use_same_dim)

        self._use_common_layer = use_common_layer
        self._use_feature_normlization = use_feature_normlization
        self._use_feature_popart = use_feature_popart
        self._use_orthogonal = use_orthogonal
        self._layer_N = layer_N
        self._use_ReLU = use_ReLU
        self._use_same_dim = use_same_dim
        self._attn = attn
        self._attn_only_critic = attn_only_critic
        # add for PC
        self.num_agents = num_agents
        self.obs_shape = obs_shape
        
        assert (self._use_feature_normlization and self._use_feature_popart) == False, ("--use_feature_normlization and --use_feature_popart can not be set True simultaneously.")
        if 'int' not in obs_shape[0].__class__.__name__: # mixed obs
            all_obs_space = obs_shape
            agent_id = self.num_agents
            self.num_agents = len(all_obs_space)
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
                share_obs_dim = obs_shape[0]*self.num_agents
        
        if self._use_feature_normlization:
            self.actor_norm = nn.LayerNorm(obs_shape[0])
            self.critic_norm = nn.LayerNorm(share_obs_dim)
            
        if self._use_feature_popart:
            self.actor_norm = PopArt(obs_shape[0])
            self.critic_norm = PopArt(share_obs_dim)
        if self._attn:           
            if use_average_pool == True:
                num_inputs_actor = attn_size + obs_shape[-1][1]
                if self._use_same_dim:            
                    num_inputs_critic = attn_size + obs_shape[-1][1]    #this one
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
                    num_inputs_critic = self.num_agents * attn_size
        elif self._attn_only_critic:
            num_inputs_actor = obs_shape[0]
            if use_average_pool == True:
                num_inputs_critic = attn_size
            else:
                num_inputs_critic = self.num_agents * attn_size
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

        self.actor = MLPLayer(num_inputs_actor, hidden_size, self._layer_N, self._use_orthogonal, self._use_ReLU)
        self.critic = MLPLayer(num_inputs_critic, hidden_size, self._layer_N, self._use_orthogonal, self._use_ReLU)
   
        if self._use_common_layer:
            self.actor = nn.Sequential(
                init_(nn.Linear(num_inputs_actor, hidden_size)), active_func)    
            self.critic = nn.Sequential(
                init_(nn.Linear(num_inputs_critic, hidden_size)), active_func)
            self.fc_h = nn.Sequential(init_(nn.Linear(hidden_size, hidden_size)), active_func)
            self.common_linear = get_clones(self.fc_h, self._layer_N)

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

    def forward(self, agent_id, share_inputs, inputs, rnn_hxs_actor, rnn_hxs_critic, masks):
        x = inputs
        share_x = share_inputs
        x_actor = torch.tensor(x)
        x_actor[:,-3:-1] = 0   #actor with no spawn obs
        share_x_actor = torch.tensor(share_x)
        share_x_actor[:,-3:-1] = 0 #actor with no spawn obs
        if self._use_feature_normlization or self._use_feature_popart:
            x = self.actor_norm(x)
            share_x = self.critic_norm(share_x)
            x_actor = self.actor_norm(x_actor)
            share_x_actor = self.critic_norm(share_x_actor)

        if self.is_attn:
            x = self.encoder_actor(x)  # TO DO
            x_actor = self.encoder_actor(x_actor)
            if self._use_same_dim:
                share_x = self.encoder_critic(share_x)  # TO DO
                share_x_actor = self.encoder_critic(share_x_actor)
            else:
                share_x = self.encoder_critic(share_x, agent_id)
                share_x_actor = self.encoder_critic(share_x_actor, agent_id)
        elif self._attn_only_critic:
            share_x = self.encoder_critic(share_x, agent_id)
            share_x_actor = self.encoder_critic(share_x_actor, agent_id)
                            
        if self._use_common_layer:
            hidden_actor = self.actor(x_actor)
            hidden_critic = self.critic(share_x)
            for i in range(self._layer_N):
                hidden_actor = self.common_linear[i](hidden_actor)
                hidden_critic = self.common_linear[i](hidden_critic)            
            if self.is_recurrent or self.is_naive_recurrent:
                hidden_actor, rnn_hxs_actor = self._forward_gru(hidden_actor, rnn_hxs_actor, masks)
                hidden_critic, rnn_hxs_critic = self._forward_gru(hidden_critic, rnn_hxs_critic, masks)
        else:
            hidden_actor = self.actor(x_actor)   # TO DO
            hidden_critic = self.critic(share_x)   # TO DO
            if self.is_recurrent or self.is_naive_recurrent:
                hidden_actor, rnn_hxs_actor = self._forward_gru(hidden_actor, rnn_hxs_actor, masks)
                hidden_critic, rnn_hxs_critic = self._forward_gru_critic(hidden_critic, rnn_hxs_critic, masks)  
                
        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs_actor, rnn_hxs_critic

# region amigo
class Policy_teacher(nn.Module): 
    def __init__(self, num_inputs, action_space, num_agents, device=torch.device("cpu")):
        super(Policy_teacher, self).__init__()
        self.device = device
        self.actor_base = teacher_actor(num_inputs, action_space)
        self.critic_base = teacher_critic(num_inputs)

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

class teacher_actor(nn.Module):
    def __init__(self, obs_shape, action_space,hidden_size=64):
        super(teacher_actor, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))
        self.actor = ObsEncoder_teacher(obs_shape)
        num_actions = action_space          
        self.dist = DiagGaussian(hidden_size, num_actions)

    def forward(self, inputs):
        hidden_actor = self.actor(inputs)
        dist = self.dist(hidden_actor)
        return dist
        # return action_out, action_log_probs_out, dist_entropy_out

class teacher_critic(nn.Module):
    def __init__(self, obs_shape,hidden_size=64):
        super(teacher_critic, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.encoder = ObsEncoder_teacher(obs_shape)

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

    def forward(self, inputs):
        vector_embedding = self.encoder(inputs)
        value = self.critic_linear(vector_embedding)

        return value

class ObsEncoder_teacher(nn.Module):
    def __init__(self, num_inputs, hidden_size=64):
        super(ObsEncoder_teacher, self).__init__()
        
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))
        self.encoder = nn.Sequential(
                            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(), nn.LayerNorm(hidden_size),
                            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(), nn.LayerNorm(hidden_size))


    # agent_num需要手动设置一下
    def forward(self, inputs):
        vector_embedding = self.encoder(inputs)
        return vector_embedding

# end region

class RNNlayer(nn.Module):
    def __init__(self,inputs_dim, outputs_dim, use_orthogonal):
        super(RNNlayer, self).__init__()
        self._use_orthogonal = use_orthogonal
        self.gru = nn.GRU(inputs_dim, outputs_dim)         
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                if self._use_orthogonal:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)

    def forward(self, x, hxs, masks):
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

class MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_size, layer_N, use_orthogonal, use_ReLU):
        super(MLPLayer, self).__init__()
        self._layer_N = layer_N
        if use_orthogonal:
            init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        else:
            init_ = lambda m: init(m, nn.init.xavier_uniform_, lambda x: nn.init.constant_(x, 0))
        
        if use_ReLU:
            active_func = nn.ReLU()
        else:
            active_func = nn.Tanh()

        self.fc1 = nn.Sequential(init_(nn.Linear(input_dim, hidden_size)), active_func)
        self.fc_h = nn.Sequential(init_(nn.Linear(hidden_size, hidden_size)), active_func)
        self.fc2 = get_clones(self.fc_h, self._layer_N)
    
    def forward(self, x):
        x = self.fc1(x)
        for i in range(self._layer_N):
            x = self.fc2[i](x)
        return x

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
                setattr(self,'fc_'+str(i), nn.Sequential(init_(nn.Linear(split_shape[i][1]+split_shape[-1][1], d_model)), active_func))
        
                         
    def forward(self, x, self_idx=-1):
        x = split_obs(x, self.split_shape)
        N = len(x)
        
        x1 = []  
        self_x = x[self_idx]      
        for i in range(N-1):
            K = self.split_shape[i][0]
            L = self.split_shape[i][1]
            for j in range(K):
                temp = torch.cat((x[i][:, (L*j):(L*j+L)], self_x), dim=-1)
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
            self.catself=True
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
    
