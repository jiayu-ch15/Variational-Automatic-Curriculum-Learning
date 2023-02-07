import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import wandb

def huber_loss(e, d):
    a = (abs(e)<=d).float()
    b = (e>d).float()
    return a*e**2/2 + b*d*(abs(e)-d/2)
    
def get_p_and_g_mean_norm(it):

    size = 1e-8
    su_p = 0
    su_g = 0
    for x in it:
        if x.grad is None:continue
        size += 1.
        su_p += x.norm()
        su_g += x.grad.norm()
    return su_p / size, su_g / size

class PopArt(nn.Module):
    """ Normalize a vector of observations - across the first norm_axes dimensions"""

    def __init__(self, input_shape, norm_axes=1, beta=0.99999, per_element_update=False, epsilon=1e-5, device=torch.device("cpu")):
        super(PopArt, self).__init__()

        self.input_shape = input_shape
        self.norm_axes = norm_axes
        self.epsilon = epsilon
        self.beta = beta
        self.per_element_update = per_element_update
        self.train = True
        self.device = device

        self.running_mean = nn.Parameter(torch.zeros(input_shape, dtype=torch.float), requires_grad=False).to(self.device)
        self.running_mean_sq = nn.Parameter(torch.zeros(input_shape, dtype=torch.float), requires_grad=False).to(self.device)
        self.debiasing_term = nn.Parameter(torch.tensor(0.0, dtype=torch.float), requires_grad=False).to(self.device)

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_mean_sq.zero_()
        self.debiasing_term.zero_()

    def running_mean_var(self):
        debiased_mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
        debiased_mean_sq = self.running_mean_sq / self.debiasing_term.clamp(min=self.epsilon)
        debiased_var = (debiased_mean_sq - debiased_mean ** 2).clamp(min=1e-2)
        return debiased_mean, debiased_var

    def forward(self, input_vector):
        # Make sure input is float32
        input_vector = input_vector.to(torch.float).to(self.device)

        if self.train:
            # Detach input before adding it to running means to avoid backpropping through it on
            # subsequent batches.
            detached_input = input_vector.detach()            
            batch_mean = detached_input.mean(dim=tuple(range(self.norm_axes)))
            batch_sq_mean = (detached_input ** 2).mean(dim=tuple(range(self.norm_axes)))

            if self.per_element_update:
                batch_size = np.prod(detached_input.size()[:self.norm_axes])
                weight = self.beta ** batch_size
            else:
                weight = self.beta

            self.running_mean.mul_(weight).add_(batch_mean * (1.0 - weight))
            self.running_mean_sq.mul_(weight).add_(batch_sq_mean * (1.0 - weight))
            self.debiasing_term.mul_(weight).add_(1.0 * (1.0 - weight))

        mean, var = self.running_mean_var()
        out = (input_vector - mean[(None,) * self.norm_axes]) / torch.sqrt(var)[(None,) * self.norm_axes]
        return out

    def denormalize(self, input_vector):
        """ Transform normalized data back into original distribution """
        input_vector = input_vector.to(torch.float).to(self.device)

        mean, var = self.running_mean_var()
        out = input_vector * torch.sqrt(var)[(None,) * self.norm_axes] + mean[(None,) * self.norm_axes]
        return out

class PPO():
    def __init__(self,                 
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 data_chunk_length,
                 value_loss_coef,
                 entropy_coef,
                 logger = None,
                 lr=None,
                 eps=None,
                 weight_decay=None,
                 max_grad_norm=None,
                 use_max_grad_norm=True,
                 use_clipped_value_loss=True,
                 use_common_layer=False,
                 use_huber_loss = False,
                 use_accumulate_grad = True,
                 use_grad_average = False,
                 huber_delta=2,
                 use_popart = True,
                 device = torch.device("cpu")
                 ):

        self.step=0
        self.device = device
        self.logger = logger

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch
        self.use_accumulate_grad = use_accumulate_grad
        self.use_grad_average = use_grad_average
        self.data_chunk_length = data_chunk_length

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_max_grad_norm = use_max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.use_common_layer = use_common_layer
        self.use_huber_loss = use_huber_loss
        self.huber_delta = huber_delta
        self.lr = lr
        self.eps = eps
        self.weight_decay = weight_decay
        self.actor_critic = actor_critic
        self.optimizer_actor = optim.Adam(actor_critic.actor_base.parameters(), lr=lr, eps=eps, weight_decay=weight_decay)
        self.optimizer_critic = optim.Adam(actor_critic.critic_base.parameters(), lr=lr, eps=eps, weight_decay=weight_decay)

        self.use_popart = use_popart
        if self.use_popart:
            self.value_normalizer = PopArt(1, device=self.device)
        else:
            self.value_normalizer = None
    
    def load(self, load_model_path, initial_optimizer):
        # load model and optimizer
        checkpoints = torch.load(load_model_path, map_location=self.device)
        self.actor_critic = checkpoints['model']
        if initial_optimizer:
            self.optimizer_actor = optim.Adam(self.actor_critic.actor_base.parameters(), lr=self.lr, eps=self.eps, weight_decay=self.weight_decay)
            self.optimizer_critic = optim.Adam(self.actor_critic.critic_base.parameters(), lr=self.lr, eps=self.eps, weight_decay=self.weight_decay)
        else:
            self.optimizer_actor = checkpoints['optimizer_actor']
            self.optimizer_critic = checkpoints['optimizer_critic']
    
    def update_share(self, num_agents, rollouts, warm_up=False):
        advantages = []
        for agent_id in range(num_agents):
            if self.use_popart:
                advantage = rollouts.returns[:-1,:,agent_id] - self.value_normalizer.denormalize(torch.tensor(rollouts.value_preds[:-1,:,agent_id])).cpu().numpy()
            else:
                advantage = rollouts.returns[:-1,:,agent_id] - rollouts.value_preds[:-1,:,agent_id]           
            advantages.append(advantage)
        #agent ,step, parallel,1
        advantages = np.array(advantages).transpose(1,2,0,3)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)      

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        self.actor_critic.num_agents = num_agents

        for e in range(self.ppo_epoch):
            
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator_share(
                    advantages, self.num_mini_batch, self.data_chunk_length)
            elif self.actor_critic.is_naive_recurrent:
                data_generator = rollouts.naive_recurrent_generator_share(
                    advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator_share(
                    advantages, self.num_mini_batch)
            
            count_stop_step = 0
            for sample in data_generator: 
                share_obs_batch, obs_batch, recurrent_hidden_states_batch, recurrent_hidden_states_critic_batch, actions_batch, \
                   value_preds_batch, return_batch, masks_batch, high_masks_batch, old_action_log_probs_batch, \
                        adv_targ = sample               
                  
                old_action_log_probs_batch = old_action_log_probs_batch.to(self.device)
                
                adv_targ = adv_targ.to(self.device)
                value_preds_batch = value_preds_batch.to(self.device)
                return_batch = return_batch.to(self.device)
                high_masks_batch = high_masks_batch.to(self.device)
  
                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _, _ = self.actor_critic.evaluate_actions(agent_id, share_obs_batch, 
                obs_batch, recurrent_hidden_states_batch, recurrent_hidden_states_critic_batch, masks_batch, high_masks_batch, actions_batch)
                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                # KL_divloss = nn.KLDivLoss(reduction='batchmean')(old_action_log_probs_batch, torch.exp(action_log_probs))

                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                action_loss = (-torch.min(surr1, surr2)* high_masks_batch).sum() / high_masks_batch.sum()

                if self.use_huber_loss:
                    if self.use_popart:
                        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                        error_clipped = self.value_normalizer(return_batch) - value_pred_clipped
                        value_losses_clipped = huber_loss(error_clipped, self.huber_delta)
                        error = self.value_normalizer(return_batch) - values
                        value_losses = huber_loss(error,self.huber_delta)
                        value_loss = torch.max(value_losses, value_losses_clipped).mean()
                    else:
                        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                        error_clipped = (return_batch) - value_pred_clipped
                        value_losses_clipped = huber_loss(error_clipped, self.huber_delta)
                        error = (return_batch) - values
                        value_losses = huber_loss(error,self.huber_delta)
                        value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    if self.use_popart:
                        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                        value_losses = (values - self.value_normalizer(return_batch)).pow(2)
                        value_losses_clipped = (value_pred_clipped - self.value_normalizer(return_batch)).pow(2)
                        value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                    else:
                        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                        value_losses = (values - (return_batch)).pow(2)
                        value_losses_clipped = (value_pred_clipped - (return_batch)).pow(2)
                        value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                    

                if self.use_accumulate_grad:
                    if count_stop_step >= self.num_mini_batch or count_stop_step==0:
                        self.optimizer_actor.zero_grad()
                        self.optimizer_critic.zero_grad()
                        count_stop_step = 0

                    if self.use_grad_average:
                        value_loss = value_loss / self.num_mini_batch
                        action_loss = action_loss / self.num_mini_batch
                        dist_entropy = dist_entropy / self.num_mini_batch

                    (value_loss * self.value_loss_coef).backward()
                    if warm_up == False:
                        (action_loss - dist_entropy * self.entropy_coef).backward()
                    
                    actor_norm, actor_grad_norm = get_p_and_g_mean_norm(self.actor_critic.actor_base.parameters())
                    critic_norm, critic_grad_norm = get_p_and_g_mean_norm(self.actor_critic.critic_base.parameters())
                        
                    if self.use_max_grad_norm:
                        nn.utils.clip_grad_norm_(self.actor_critic.actor_base.parameters(), self.max_grad_norm)
                        nn.utils.clip_grad_norm_(self.actor_critic.critic_base.parameters(), self.max_grad_norm)
                    
                    if count_stop_step == self.num_mini_batch-1: 
                        self.optimizer_critic.step()
                        if warm_up == False:
                            self.optimizer_actor.step()
                    count_stop_step += 1
                else:
                    self.optimizer_actor.zero_grad()
                    self.optimizer_critic.zero_grad()
                    (value_loss * self.value_loss_coef).backward()
                    if warm_up == False:
                        (action_loss - dist_entropy * self.entropy_coef).backward()
                    if self.use_max_grad_norm:
                        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                    
                    self.optimizer_critic.step()
                    if warm_up == False:
                        self.optimizer_actor.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()  
       
        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch

    def update_double_share(self, last_agents, current_agents, rollouts_last, rollouts_current):
        advantages_last = []
        advantages_current = []
        for agent_id in range(last_agents):
            if self.use_popart:
                advantage_last = rollouts_last.returns[:-1,:,agent_id] - self.value_normalizer.denormalize(torch.tensor(rollouts_last.value_preds[:-1,:,agent_id])).cpu().numpy()
            else:
                advantage_last = rollouts_last.returns[:-1,:,agent_id] - rollouts_last.value_preds[:-1,:,agent_id]           
            advantages_last.append(advantage_last)
        #agent , step, parallel,1
        advantages_last = np.array(advantages_last).transpose(1,2,0,3) 
        # step, parallel, agent, 1
        for agent_id in range(current_agents):
            if self.use_popart:
                advantage_current = rollouts_current.returns[:-1,:,agent_id] - self.value_normalizer.denormalize(torch.tensor(rollouts_current.value_preds[:-1,:,agent_id])).cpu().numpy()
            else:
                advantage_current = rollouts_current.returns[:-1,:,agent_id] - rollouts_current.value_preds[:-1,:,agent_id]           
            advantages_current.append(advantage_current)
        #agent ,step, parallel,1
        advantages_current = np.array(advantages_current).transpose(1,2,0,3)
        tmp_advantages = np.concatenate((advantages_last.reshape(-1,1),advantages_current.reshape(-1,1)),axis=0)
        advantages_last = (advantages_last - tmp_advantages.mean()) / (
                tmp_advantages.std() + 1e-5)  
        advantages_current = (advantages_current - tmp_advantages.mean()) / (
                tmp_advantages.std() + 1e-5) 
        
        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            
            if self.actor_critic.is_recurrent:
                data_generator_last = rollouts_last.recurrent_generator_share(
                    advantages_last, self.num_mini_batch, self.data_chunk_length)
                data_generator_current = rollouts_current.recurrent_generator_share(
                    advantages_current, self.num_mini_batch, self.data_chunk_length)
            elif self.actor_critic.is_naive_recurrent:
                data_generator_last = rollouts_last.naive_recurrent_generator_share(
                    advantages_last, self.num_mini_batch)
                data_generator_current = rollouts_current.naive_recurrent_generator_share(
                    advantages_current, self.num_mini_batch)
            else:
                data_generator_last = rollouts_last.feed_forward_generator_share(
                    advantages_last, self.num_mini_batch)
                data_generator_current = rollouts_current.feed_forward_generator_share(
                    advantages_current, self.num_mini_batch)
            
            count_stop_step = 0
            for sample_last, sample_current in zip(data_generator_last, data_generator_current): 
                share_obs_batch_last, obs_batch_last, recurrent_hidden_states_batch_last, recurrent_hidden_states_critic_batch_last, actions_batch_last, \
                   value_preds_batch_last, return_batch_last, masks_batch_last, high_masks_batch_last, old_action_log_probs_batch_last, \
                        adv_targ_last = sample_last 
                share_obs_batch_current, obs_batch_current, recurrent_hidden_states_batch_current, recurrent_hidden_states_critic_batch_current, actions_batch_current, \
                   value_preds_batch_current, return_batch_current, masks_batch_current, high_masks_batch_current, old_action_log_probs_batch_current, \
                        adv_targ_current = sample_current             
                
                weight_last = obs_batch_last.shape[0]/(obs_batch_last.shape[0]+obs_batch_current.shape[0])
                old_action_log_probs_batch_last = old_action_log_probs_batch_last.to(self.device)
                adv_targ_last = adv_targ_last.to(self.device)
                value_preds_batch_last = value_preds_batch_last.to(self.device)
                return_batch_last = return_batch_last.to(self.device)
                high_masks_batch_last = high_masks_batch_last.to(self.device)

                weight_current = obs_batch_current.shape[0]/(obs_batch_last.shape[0]+obs_batch_current.shape[0])
                old_action_log_probs_batch_current = old_action_log_probs_batch_current.to(self.device)
                adv_targ_current = adv_targ_current.to(self.device)
                value_preds_batch_current = value_preds_batch_current.to(self.device)
                return_batch_current = return_batch_current.to(self.device)
                high_masks_batch_current = high_masks_batch_current.to(self.device)
  
                # Reshape to do in a single forward pass for all steps
                
                # last agent update
                self.actor_critic.num_agents = last_agents
                values, action_log_probs, dist_entropy, _, _ = self.actor_critic.evaluate_actions(agent_id, share_obs_batch_last, 
                obs_batch_last, recurrent_hidden_states_batch_last, recurrent_hidden_states_critic_batch_last, masks_batch_last, high_masks_batch_last, actions_batch_last)

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch_last)
                KL_divloss = nn.KLDivLoss(reduction='batchmean')(old_action_log_probs_batch_last, torch.exp(action_log_probs))
                surr1 = ratio * adv_targ_last
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ_last
                action_loss = (-torch.min(surr1, surr2)* high_masks_batch_last).sum() / high_masks_batch_last.sum()

                if self.use_clipped_value_loss:
                    if self.use_huber_loss:
                        if self.use_popart:
                            value_pred_clipped = value_preds_batch_last + (values - value_preds_batch_last).clamp(-self.clip_param, self.clip_param)
                            error_clipped = self.value_normalizer(return_batch_last) - value_pred_clipped
                            value_losses_clipped = huber_loss(error_clipped, self.huber_delta)
                            error = self.value_normalizer(return_batch_last) - values
                            value_losses = huber_loss(error,self.huber_delta)
                            value_loss = torch.max(value_losses, value_losses_clipped).mean()
                        else:
                            value_pred_clipped = value_preds_batch_last + (values - value_preds_batch_last).clamp(-self.clip_param, self.clip_param)
                            error_clipped = (return_batch_last) - value_pred_clipped
                            value_losses_clipped = huber_loss(error_clipped, self.huber_delta)
                            error = (return_batch_last) - values
                            value_losses = huber_loss(error,self.huber_delta)
                            value_loss = torch.max(value_losses, value_losses_clipped).mean()
                    else:
                        if self.use_popart:
                            value_pred_clipped = value_preds_batch_last + (values - value_preds_batch_last).clamp(-self.clip_param, self.clip_param)
                            value_losses = (values - self.value_normalizer(return_batch_last)).pow(2)
                            value_losses_clipped = (value_pred_clipped - self.value_normalizer(return_batch_last)).pow(2)
                            value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                        else:
                            value_pred_clipped = value_preds_batch_last + (values - value_preds_batch_last).clamp(-self.clip_param, self.clip_param)
                            value_losses = (values - (return_batch_last)).pow(2)
                            value_losses_clipped = (value_pred_clipped - (return_batch_last)).pow(2)
                            value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean() 
                else:
                    if self.use_huber_loss:
                        if self.use_popart:
                            error = self.value_normalizer(return_batch_last) - values
                        else:
                            error = return_batch_last - values
                        value_loss = huber_loss(error,self.huber_delta).mean()
                    else:
                        if self.use_popart:
                            value_loss = 0.5 * (self.value_normalizer(return_batch_last) - values).pow(2).mean()
                        else:
                            value_loss = 0.5 * (return_batch_last - values).pow(2).mean()               
                
                if count_stop_step > self.num_mini_batch or count_stop_step==0: 
                    self.optimizer_actor.zero_grad()
                    self.optimizer_critic.zero_grad()
                    count_stop_step = 0
                
                
                value_loss = value_loss * weight_last
                action_loss = action_loss * weight_last
                dist_entropy = dist_entropy * weight_last

                (value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef).backward()
                
                actor_norm, actor_grad_norm = get_p_and_g_mean_norm(self.actor_critic.actor_base.parameters())
                critic_norm, critic_grad_norm = get_p_and_g_mean_norm(self.actor_critic.critic_base.parameters())
                       
                if self.use_max_grad_norm:
                    nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                 
                # current agent backward
                self.actor_critic.num_agents = current_agents
                values, action_log_probs, dist_entropy, _, _ = self.actor_critic.evaluate_actions(agent_id, share_obs_batch_current, 
                obs_batch_current, recurrent_hidden_states_batch_current, recurrent_hidden_states_critic_batch_current, masks_batch_current, high_masks_batch_current, actions_batch_current)
  
                ratio = torch.exp(action_log_probs - old_action_log_probs_batch_current)
                KL_divloss = nn.KLDivLoss(reduction='batchmean')(old_action_log_probs_batch_current, torch.exp(action_log_probs))
                surr1 = ratio * adv_targ_current
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ_current
                action_loss = (-torch.min(surr1, surr2)* high_masks_batch_current).sum() / high_masks_batch_current.sum()

                if self.use_clipped_value_loss:
                    if self.use_huber_loss:
                        if self.use_popart:
                            value_pred_clipped = value_preds_batch_current + (values - value_preds_batch_current).clamp(-self.clip_param, self.clip_param)
                            error_clipped = self.value_normalizer(return_batch_current) - value_pred_clipped
                            value_losses_clipped = huber_loss(error_clipped, self.huber_delta)
                            error = self.value_normalizer(return_batch_current) - values
                            value_losses = huber_loss(error,self.huber_delta)
                            value_loss = torch.max(value_losses, value_losses_clipped).mean()
                        else:
                            value_pred_clipped = value_preds_batch_current + (values - value_preds_batch_current).clamp(-self.clip_param, self.clip_param)
                            error_clipped = (return_batch_current) - value_pred_clipped
                            value_losses_clipped = huber_loss(error_clipped, self.huber_delta)
                            error = (return_batch_current) - values
                            value_losses = huber_loss(error,self.huber_delta)
                            value_loss = torch.max(value_losses, value_losses_clipped).mean()
                    else:
                        if self.use_popart:
                            value_pred_clipped = value_preds_batch_current + (values - value_preds_batch_current).clamp(-self.clip_param, self.clip_param)
                            value_losses = (values - self.value_normalizer(return_batch_current)).pow(2)
                            value_losses_clipped = (value_pred_clipped - self.value_normalizer(return_batch_current)).pow(2)
                            value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                        else:
                            value_pred_clipped = value_preds_batch_current + (values - value_preds_batch_current).clamp(-self.clip_param, self.clip_param)
                            value_losses = (values - (return_batch_current)).pow(2)
                            value_losses_clipped = (value_pred_clipped - (return_batch_current)).pow(2)
                            value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean() 
                else:
                    if self.use_huber_loss:
                        if self.use_popart:
                            error = self.value_normalizer(return_batch_current) - values
                        else:
                            error = return_batch_current - values
                        value_loss = huber_loss(error,self.huber_delta).mean()
                    else:
                        if self.use_popart:
                            value_loss = 0.5 * (self.value_normalizer(return_batch_current) - values).pow(2).mean()
                        else:
                            value_loss = 0.5 * (return_batch_current - values).pow(2).mean()               
                        

                value_loss = value_loss * weight_current
                action_loss = action_loss * weight_current
                dist_entropy = dist_entropy * weight_current
                (value_loss * self.value_loss_coef).backward()
                (action_loss - dist_entropy * self.entropy_coef).backward()

                norm, grad_norm = get_p_and_g_mean_norm(self.actor_critic.parameters())
                       
                if self.use_max_grad_norm:
                    nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)

                if count_stop_step == self.num_mini_batch - 1: 
                    self.optimizer_critic.step()
                    self.optimizer_actor.step()
                count_stop_step += 1

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()  
       
        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch

    def update_single(self, agent_id, role, rollouts, timestep, warm_up=False):
        advantages = []
        if self.use_popart:
            advantages = rollouts.returns[:-1,:] - self.value_normalizer.denormalize(torch.tensor(rollouts.value_preds[:-1,:])).cpu().numpy()
        else:
            advantages = rollouts.returns[:-1,:] - rollouts.value_preds[:-1,:]
        #agent ,step, parallel,1
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)      

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator_share(
                    advantages, self.num_mini_batch, self.data_chunk_length)
            elif self.actor_critic.is_naive_recurrent:
                data_generator = rollouts.naive_recurrent_generator_share(
                    advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)
            
            count_stop_step = 0
            for sample in data_generator: 
                share_obs_batch, obs_batch, recurrent_hidden_states_batch, recurrent_hidden_states_critic_batch, actions_batch, \
                   value_preds_batch, return_batch, masks_batch, high_masks_batch, old_action_log_probs_batch, \
                        adv_targ = sample               
                  
                old_action_log_probs_batch = old_action_log_probs_batch.to(self.device)
                
                adv_targ = adv_targ.to(self.device)
                value_preds_batch = value_preds_batch.to(self.device)
                return_batch = return_batch.to(self.device)
                high_masks_batch = high_masks_batch.to(self.device)
  
                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _, _ = self.actor_critic.evaluate_actions_role(agent_id, share_obs_batch, 
                obs_batch, role, recurrent_hidden_states_batch, recurrent_hidden_states_critic_batch, masks_batch, high_masks_batch, actions_batch)
                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                # KL_divloss = nn.KLDivLoss(reduction='batchmean')(old_action_log_probs_batch, torch.exp(action_log_probs))

                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                action_loss = (-torch.min(surr1, surr2)* high_masks_batch).sum() / high_masks_batch.sum()

                if self.use_clipped_value_loss:
                    if self.use_huber_loss:
                        if self.use_popart:
                            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                            error_clipped = self.value_normalizer(return_batch) - value_pred_clipped
                            value_losses_clipped = huber_loss(error_clipped, self.huber_delta)
                            error = self.value_normalizer(return_batch) - values
                            value_losses = huber_loss(error,self.huber_delta)
                            value_loss = torch.max(value_losses, value_losses_clipped).mean()
                        else:
                            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                            error_clipped = (return_batch) - value_pred_clipped
                            value_losses_clipped = huber_loss(error_clipped, self.huber_delta)
                            error = (return_batch) - values
                            value_losses = huber_loss(error,self.huber_delta)
                            value_loss = torch.max(value_losses, value_losses_clipped).mean()
                    else:
                        if self.use_popart:
                            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                            value_losses = (values - self.value_normalizer(return_batch)).pow(2)
                            value_losses_clipped = (value_pred_clipped - self.value_normalizer(return_batch)).pow(2)
                            value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                        else:
                            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                            value_losses = (values - (return_batch)).pow(2)
                            value_losses_clipped = (value_pred_clipped - (return_batch)).pow(2)
                            value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                    
                else:
                    if self.use_huber_loss:
                        if self.use_popart:
                            error = self.value_normalizer(return_batch) - values
                        else:
                            error = return_batch - values
                        value_loss = huber_loss(error,self.huber_delta).mean()
                    else:
                        if self.use_popart:
                            value_loss = 0.5 * (self.value_normalizer(return_batch) - values).pow(2).mean()
                        else:
                            value_loss = 0.5 * (return_batch - values).pow(2).mean()               

                if self.use_accumulate_grad:
                    if count_stop_step >= self.num_mini_batch or count_stop_step==0:
                        self.optimizer_actor.zero_grad()
                        self.optimizer_critic.zero_grad()
                        count_stop_step = 0

                    if self.use_grad_average:
                        value_loss = value_loss / self.num_mini_batch
                        action_loss = action_loss / self.num_mini_batch
                        dist_entropy = dist_entropy / self.num_mini_batch

                    (value_loss * self.value_loss_coef).backward()
                    if warm_up == False:
                        (action_loss - dist_entropy * self.entropy_coef).backward()
                    
                    actor_norm, actor_grad_norm = get_p_and_g_mean_norm(self.actor_critic.actor_base.parameters())
                    critic_norm, critic_grad_norm = get_p_and_g_mean_norm(self.actor_critic.critic_base.parameters())
                        
                    if self.use_max_grad_norm:
                        nn.utils.clip_grad_norm_(self.actor_critic.actor_base.parameters(), self.max_grad_norm)
                        nn.utils.clip_grad_norm_(self.actor_critic.critic_base.parameters(), self.max_grad_norm)
                    
                    if count_stop_step == self.num_mini_batch-1: 
                        self.optimizer_critic.step()
                        if warm_up == False:
                            self.optimizer_actor.step()
                    count_stop_step += 1
                else:
                    self.optimizer_actor.zero_grad()
                    self.optimizer_critic.zero_grad()
                    (value_loss * self.value_loss_coef).backward()
                    if warm_up == False:
                        (action_loss - dist_entropy * self.entropy_coef).backward()
                    if self.use_max_grad_norm:
                        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                    
                    self.optimizer_critic.step()
                    if warm_up == False:
                        self.optimizer_actor.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()  
       
        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch