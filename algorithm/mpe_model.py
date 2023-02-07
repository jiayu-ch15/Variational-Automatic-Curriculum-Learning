import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.distributions import Bernoulli, Categorical, DiagGaussian
from utils.util import init
import copy
import math

class Policy(nn.Module):
    def __init__(self, obs_space, action_space, num_agents, base = None, actor_base=None, critic_base=None, base_kwargs=None, device=torch.device("cpu")):
        super(Policy, self).__init__()
        self.mixed_obs = False
        self.mixed_action = False
        self.multi_discrete = False
        self.device = device
        self.num_agents = num_agents
        self.args = base_kwargs
        if base_kwargs is None:
            base_kwargs = {}
        self.actor_base = actor_base
        self.critic_base = critic_base

    @property
    def is_recurrent(self):
        return self.args['recurrent']

    @property
    def is_naive_recurrent(self):
        return self.args['naive_recurrent']

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

class ATTBase_actor(nn.Module):
    def __init__(self, num_inputs, action_space, agent_num, model_name, recurrent=False, hidden_size=64):
        super(ATTBase_actor, self).__init__()
        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.agent_num = agent_num
        if model_name == 'simple_spread' or model_name == 'hard_spread':
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

class ATTBase_critic(nn.Module):
    def __init__(self, num_inputs, agent_num, model_name, recurrent=False, hidden_size=64):
        super(ATTBase_critic, self).__init__()
        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.agent_num = agent_num
        if model_name == 'simple_spread' or model_name == 'hard_spread':
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
        
        f_ij = self.encoder(share_inputs.reshape(-1,obs_dim),agent_num)
        obs_encoder = f_ij.reshape(batch_size,agent_num,-1) # (batch_size, nagents, hidden_size)
        
        beta = torch.matmul(obs_beta_ij, obs_encoder.permute(0,2,1)).squeeze(1) # (batch_size,nagents)
        alpha = F.softmax(beta,dim = 1).unsqueeze(2) # (batch_size,nagents,1)
        vi = torch.mul(alpha,obs_encoder)
        vi = torch.sum(vi,dim = 1)
        value = self.critic_linear(vi)

        return value, rnn_hxs, rnn_hxs

class ObsEncoder_sp(nn.Module): # simple spread and hard spread
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

