import torch
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import time

def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])

class SingleRolloutStorage(object):
    def __init__(self, agent_id, episode_length, n_rollout_threads, all_obs_space, all_action_space,
                 recurrent_hidden_state_size):
        
        if all_obs_space[agent_id].__class__.__name__ == 'Box':
            obs_shape = all_obs_space[agent_id].shape
            share_obs_dim = 0
            for obs_space in all_obs_space:
                share_obs_dim += obs_space.shape[0]
            if len(obs_shape) == 3:                
                self.share_obs = np.zeros((episode_length + 1, n_rollout_threads, share_obs_dim, obs_shape[1], obs_shape[2])).astype(np.float32)
                self.obs = np.zeros((episode_length + 1, n_rollout_threads, *obs_shape)).astype(np.float32)
            else:               
                self.share_obs = np.zeros((episode_length + 1, n_rollout_threads, share_obs_dim)).astype(np.float32)
                self.obs = np.zeros((episode_length + 1, n_rollout_threads, obs_shape[0])).astype(np.float32)
        elif all_obs_space[agent_id].__class__.__name__ == 'list':
            obs_shape = all_obs_space[agent_id]
            share_obs_dim = 0
            for obs_space in all_obs_space:
                share_obs_dim += obs_space[0]
            if len(obs_shape) == 3:
                self.share_obs = np.zeros((episode_length + 1, n_rollout_threads, share_obs_dim, obs_shape[1], obs_shape[2])).astype(np.float32)
                self.obs = np.zeros((episode_length + 1, n_rollout_threads, *obs_shape)).astype(np.float32)
            else:
                self.share_obs = np.zeros((episode_length + 1, n_rollout_threads, share_obs_dim)).astype(np.float32)
                self.obs = np.zeros((episode_length + 1, n_rollout_threads, obs_shape[0])).astype(np.float32)
        else:
            raise NotImplementedError
               
        self.recurrent_hidden_states = np.zeros((
            episode_length + 1, n_rollout_threads, recurrent_hidden_state_size)).astype(np.float32)
        self.recurrent_hidden_states_critic = np.zeros((
            episode_length + 1, n_rollout_threads, recurrent_hidden_state_size)).astype(np.float32)
                       
        self.rewards = np.zeros((episode_length, n_rollout_threads, 1)).astype(np.float32)
        self.value_preds = np.zeros((episode_length + 1, n_rollout_threads, 1)).astype(np.float32)
        self.returns = np.zeros((episode_length + 1, n_rollout_threads, 1)).astype(np.float32)
        self.action_log_probs = np.zeros((episode_length, n_rollout_threads, 1)).astype(np.float32)
        
        self.available_actions = None
        if all_action_space[agent_id].__class__.__name__ == 'Discrete':
            self.available_actions = np.ones((episode_length + 1, n_rollout_threads, all_action_space[agent_id].n)).astype(np.float32)
            action_shape = 1
        elif all_action_space[agent_id].__class__.__name__ == "MultiDiscrete":
            action_shape = all_action_space[agent_id].shape
        elif all_action_space[agent_id].__class__.__name__ == "Box":
            action_shape = all_action_space[agent_id].shape[0]
        elif all_action_space[agent_id].__class__.__name__ == "MultiBinary":
            action_shape = all_action_space[agent_id].shape[0]
        else:
            raise NotImplementedError
        self.actions = np.zeros((episode_length, n_rollout_threads, action_shape)).astype(np.float32)
        #if action_space.__class__.__name__ == 'Discrete':
            #self.actions = self.actions.long()
        self.masks = np.ones((episode_length + 1, n_rollout_threads, 1)).astype(np.float32)

        # Masks that indicate whether it's a true terminal state
        # or time limit end state
        self.bad_masks = np.ones((episode_length + 1, n_rollout_threads, 1)).astype(np.float32)
        
        self.high_masks = np.ones((episode_length + 1, n_rollout_threads, 1)).astype(np.float32)

        self.episode_length = episode_length
        self.step = 0

    def insert(self, share_obs, obs, recurrent_hidden_states, recurrent_hidden_states_critic, actions, action_log_probs,
               value_preds, rewards, masks, bad_masks=None, high_masks=None, available_actions=None):
        self.share_obs[self.step + 1] = share_obs.copy()
        self.obs[self.step + 1] = obs.copy()
        self.recurrent_hidden_states[self.step + 1] = recurrent_hidden_states.copy()
        self.recurrent_hidden_states_critic[self.step + 1] = recurrent_hidden_states_critic.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        if bad_masks is not None:
            self.bad_masks[self.step + 1] = bad_masks.copy()
        if high_masks is not None:
            self.high_masks[self.step + 1] = high_masks.copy()
        if available_actions is not None:
            self.available_actions[self.step + 1] = available_actions.copy()

        self.step = (self.step + 1) % self.episode_length
                
    def after_update(self):
        self.share_obs[0] = self.share_obs[-1].copy()
        self.obs[0] = self.obs[-1].copy()
        self.recurrent_hidden_states[0] = self.recurrent_hidden_states[-1].copy()
        self.recurrent_hidden_states_critic[0] = self.recurrent_hidden_states_critic[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()
        self.high_masks[0] = self.high_masks[-1].copy()
        if self.available_actions is not None:
            self.available_actions[0] = self.available_actions[-1].copy()       
        
    def compute_returns(self,
                        next_value,
                        use_gae,
                        gamma,
                        gae_lambda,
                        use_proper_time_limits=True,
                        use_popart=True,
                        value_normalizer=None):
        if use_proper_time_limits:
            if use_gae:
                self.value_preds[-1,:] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    if use_popart:
                        delta = self.rewards[step,:] + gamma * value_normalizer.denormalize(torch.tensor(self.value_preds[
                        step + 1,:])).cpu().numpy() * self.masks[step + 1,:] - value_normalizer.denormalize(torch.tensor(self.value_preds[step,:])).cpu().numpy()
                        gae = delta + gamma * gae_lambda * self.masks[step + 1,:] * gae
                        gae = gae * self.bad_masks[step + 1,:]
                        self.returns[step,:] = gae + value_normalizer.denormalize(torch.tensor(self.value_preds[step,:])).cpu().numpy()
                    else:
                        delta = self.rewards[step,:] + gamma * self.value_preds[
                            step + 1,:] * self.masks[step + 1,:] - self.value_preds[step,:]
                        gae = delta + gamma * gae_lambda * self.masks[step + 1,:] * gae
                        gae = gae * self.bad_masks[step + 1,:]
                        self.returns[step,:] = gae + self.value_preds[step,:]
            else:
                self.returns[-1,:] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    if use_popart:
                        self.returns[step,:] = (self.returns[step + 1,:] * \
                        gamma * self.masks[step + 1,:] + self.rewards[step,:]) * self.bad_masks[step + 1,:] \
                        + (1 - self.bad_masks[step + 1,:]) * value_normalizer.denormalize(torch.tensor(self.value_preds[step,:])).cpu().numpy()
                    else:
                        self.returns[step,:] = (self.returns[step + 1,:] * \
                            gamma * self.masks[step + 1,:] + self.rewards[step,:]) * self.bad_masks[step + 1,:] \
                            + (1 - self.bad_masks[step + 1,:]) * self.value_preds[step,:]
        else:
            if use_gae:
                self.value_preds[-1,:] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    if use_popart:
                        delta = self.rewards[step,:] + gamma * value_normalizer.denormalize(torch.tensor(self.value_preds[
                            step + 1,:])).cpu().numpy() * self.masks[step + 1,:] - value_normalizer.denormalize(torch.tensor(self.value_preds[step,:])).cpu().numpy()
                        gae = delta + gamma * gae_lambda * self.masks[step + 1,:] * gae                       
                        self.returns[step,:] = gae + value_normalizer.denormalize(torch.tensor(self.value_preds[step,:])).cpu().numpy()
                    else:
                        delta = self.rewards[step,:] + gamma * self.value_preds[step + 1,:] * self.masks[step + 1,:] - self.value_preds[step,:]
                        gae = delta + gamma * gae_lambda * self.masks[step + 1,:] * gae
                        self.returns[step,:] = gae + self.value_preds[step,:]
            else:
                self.returns[-1,:] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    self.returns[step,:] = self.returns[step + 1,:] * \
                            gamma * self.masks[step + 1,:] + self.rewards[step,:]

    def feed_forward_generator(self, advantages, num_mini_batch=None, mini_batch_size=None):
        episode_length, n_rollout_threads = self.rewards.shape[0:2]
        batch_size = n_rollout_threads * episode_length

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(n_rollout_threads, episode_length, n_rollout_threads * episode_length,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch
            
        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i*mini_batch_size:(i+1)*mini_batch_size] for i in range(num_mini_batch)]
        
        share_obs = self.share_obs[:-1,:].reshape(-1, *self.share_obs.shape[2:])
        obs = self.obs[:-1,:].reshape(-1, *self.obs.shape[2:])
        recurrent_hidden_states = self.recurrent_hidden_states[:-1,:].reshape(-1, self.recurrent_hidden_states.shape[-1])
        recurrent_hidden_states_critic = self.recurrent_hidden_states_critic[:-1,:].reshape(-1, self.recurrent_hidden_states_critic.shape[-1])
        actions = self.actions[:,:].reshape(-1, self.actions.shape[-1])
        value_preds = self.value_preds[:-1,:].reshape(-1, 1)
        returns = self.returns[:-1,:].reshape(-1, 1)
        masks = self.masks[:-1,:].reshape(-1, 1)
        high_masks = self.high_masks[:-1,:].reshape(-1, 1)
        action_log_probs = self.action_log_probs[:,:].reshape(-1, 1)
        advantages = advantages.reshape(-1, 1)
        
        for indices in sampler:
            # obs size [T+1 N Dim]-->[T N Dim]-->[T*N,Dim]-->[index,Dim]
            share_obs_batch = torch.tensor(share_obs[indices])
            obs_batch = torch.tensor(obs[indices])
            recurrent_hidden_states_batch = torch.tensor(recurrent_hidden_states[indices])
            recurrent_hidden_states_critic_batch = torch.tensor(recurrent_hidden_states_critic[indices])
            actions_batch = torch.tensor(actions[indices])
            value_preds_batch = torch.tensor(value_preds[indices])
            return_batch = torch.tensor(returns[indices])
            masks_batch = torch.tensor(masks[indices])
            high_masks_batch = torch.tensor(high_masks[indices])
            old_action_log_probs_batch = torch.tensor(action_log_probs[indices])
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = torch.tensor(advantages[indices])

            yield share_obs_batch, obs_batch, recurrent_hidden_states_batch, recurrent_hidden_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, high_masks_batch, old_action_log_probs_batch, adv_targ
            
    def naive_recurrent_generator(self, advantages, num_mini_batch):
        n_rollout_threads = self.rewards.shape[1]
        assert n_rollout_threads >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(n_rollout_threads, num_mini_batch))
        num_envs_per_batch = n_rollout_threads // num_mini_batch
        perm = torch.randperm(n_rollout_threads).numpy()
        for start_ind in range(0, n_rollout_threads, num_envs_per_batch):
            share_obs_batch = []
            obs_batch = []
            recurrent_hidden_states_batch = []
            recurrent_hidden_states_critic_batch = []
            actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            high_masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                share_obs_batch.append(torch.tensor(self.share_obs[:-1, ind]))
                obs_batch.append(torch.tensor(self.obs[:-1, ind]))
                recurrent_hidden_states_batch.append(
                    torch.tensor(self.recurrent_hidden_states[0:1, ind]))
                recurrent_hidden_states_critic_batch.append(
                    torch.tensor(self.recurrent_hidden_states_critic[0:1, ind]))
                actions_batch.append(torch.tensor(self.actions[:, ind]))
                value_preds_batch.append(torch.tensor(self.value_preds[:-1, ind]))
                return_batch.append(torch.tensor(self.returns[:-1, ind]))
                masks_batch.append(torch.tensor(self.masks[:-1, ind]))
                high_masks_batch.append(torch.tensor(self.high_masks[:-1, ind]))
                old_action_log_probs_batch.append(
                    torch.tensor(self.action_log_probs[:, ind]))
                adv_targ.append(torch.tensor(advantages[:, ind]))

            T, N = self.episode_length, num_envs_per_batch
            # These are all tensors of size (T, N, -1)
            share_obs_batch = torch.stack(share_obs_batch, 1)
            obs_batch = torch.stack(obs_batch, 1)
            actions_batch = torch.stack(actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            high_masks_batch = torch.stack(high_masks_batch, 1)
            old_action_log_probs_batch = torch.stack(
                old_action_log_probs_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)

            # States is just a (N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(
                recurrent_hidden_states_batch, 1).view(N, -1)
            recurrent_hidden_states_critic_batch = torch.stack(
                recurrent_hidden_states_critic_batch, 1).view(N, -1)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            share_obs_batch = _flatten_helper(T, N, share_obs_batch)
            obs_batch = _flatten_helper(T, N, obs_batch)
            actions_batch = _flatten_helper(T, N, actions_batch)
            value_preds_batch = _flatten_helper(T, N, value_preds_batch)
            return_batch = _flatten_helper(T, N, return_batch)
            masks_batch = _flatten_helper(T, N, masks_batch)
            high_masks_batch = _flatten_helper(T, N, high_masks_batch)
            old_action_log_probs_batch = _flatten_helper(T, N, \
                    old_action_log_probs_batch)
            adv_targ = _flatten_helper(T, N, adv_targ)
            

            yield share_obs_batch, obs_batch, recurrent_hidden_states_batch, recurrent_hidden_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, high_masks_batch, old_action_log_probs_batch, adv_targ 
                
    def recurrent_generator(self, advantages, num_mini_batch, data_chunk_length):
        episode_length, n_rollout_threads = self.rewards.shape[0:2]
        batch_size = n_rollout_threads * episode_length
        data_chunks = batch_size // data_chunk_length #[C=r*T/L]
        mini_batch_size = data_chunks // num_mini_batch
            
        rand = torch.randperm(data_chunks).numpy()
        sampler = [rand[i*mini_batch_size:(i+1)*mini_batch_size] for i in range(num_mini_batch)]
            
        if len(self.share_obs.shape) > 3:
            share_obs = self.share_obs[:-1,:].transpose(1,0,2,3,4).reshape(-1, *self.share_obs.shape[2:])
            obs = self.obs[:-1,:].transpose(1,0,2,3,4).reshape(-1, *self.obs.shape[2:])
        else:
            share_obs = self.share_obs[:-1,:].transpose(1,0,2).reshape(-1, *self.share_obs.shape[2:])
            obs = self.obs[:-1,:].transpose(1,0,2).reshape(-1, *self.obs.shape[2:])
            
        actions = self.actions[:,:].transpose(1,0,2).reshape(-1, self.actions.shape[-1])
        value_preds = self.value_preds[:-1,:].transpose(1,0,2).reshape(-1, 1)
        returns = self.returns[:-1,:].transpose(1,0,2).reshape(-1, 1)
        masks = self.masks[:-1,:].transpose(1,0,2).reshape(-1, 1)
        high_masks = self.high_masks[:-1,:].transpose(1,0,2).reshape(-1, 1)
        action_log_probs = self.action_log_probs[:,:].transpose(1,0,2).reshape(-1, 1)
        advantages = advantages.transpose(1,0,2).reshape(-1, 1)
        recurrent_hidden_states = self.recurrent_hidden_states[:-1,:].transpose(1,0,2).reshape(-1, self.recurrent_hidden_states.shape[-1])
        recurrent_hidden_states_critic = self.recurrent_hidden_states_critic[:-1,:].transpose(1,0,2).reshape(-1, self.recurrent_hidden_states_critic.shape[-1])

        for indices in sampler:
            share_obs_batch = []
            obs_batch = []
            recurrent_hidden_states_batch = []
            recurrent_hidden_states_critic_batch = []
            actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            high_masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []
            
            for index in indices:
                ind = index * data_chunk_length
                # size [T+1 N M Dim]-->[T N Dim]-->[N T Dim]-->[T*N,Dim]-->[L,Dim]
                share_obs_batch.append(torch.tensor(share_obs[ind:ind+data_chunk_length]))
                obs_batch.append(torch.tensor(obs[ind:ind+data_chunk_length]))
                actions_batch.append(torch.tensor(actions[ind:ind+data_chunk_length]))
                value_preds_batch.append(torch.tensor(value_preds[ind:ind+data_chunk_length]))
                return_batch.append(torch.tensor(returns[ind:ind+data_chunk_length]))
                masks_batch.append(torch.tensor(masks[ind:ind+data_chunk_length]))
                high_masks_batch.append(torch.tensor(high_masks[ind:ind+data_chunk_length]))
                old_action_log_probs_batch.append(torch.tensor(action_log_probs[ind:ind+data_chunk_length]))
                adv_targ.append(torch.tensor(advantages[ind:ind+data_chunk_length]))
                # size [T+1 N Dim]-->[T N Dim]-->[T*N,Dim]-->[1,Dim]
                recurrent_hidden_states_batch.append(torch.tensor(recurrent_hidden_states[ind]))
                recurrent_hidden_states_critic_batch.append(torch.tensor(recurrent_hidden_states_critic[ind]))
                      
            L, N =  data_chunk_length, mini_batch_size
                        
            # These are all tensors of size (L, N, Dim)
            share_obs_batch = torch.stack(share_obs_batch)         
            obs_batch = torch.stack(obs_batch)
            
            actions_batch = torch.stack(actions_batch)
            value_preds_batch = torch.stack(value_preds_batch)
            return_batch = torch.stack(return_batch)
            masks_batch = torch.stack(masks_batch)
            high_masks_batch = torch.stack(high_masks_batch)
            old_action_log_probs_batch = torch.stack(
                old_action_log_probs_batch)
            adv_targ = torch.stack(adv_targ)

            # States is just a (N, -1) tensor
            
            recurrent_hidden_states_batch = torch.stack(
                recurrent_hidden_states_batch).view(N, -1)
            recurrent_hidden_states_critic_batch = torch.stack(
                recurrent_hidden_states_critic_batch).view(N, -1)

            # Flatten the (L, N, ...) tensors to (L * N, ...)
            share_obs_batch = _flatten_helper(L, N, share_obs_batch)
            obs_batch = _flatten_helper(L, N, obs_batch)
            actions_batch = _flatten_helper(L, N, actions_batch)
            value_preds_batch = _flatten_helper(L, N, value_preds_batch)
            return_batch = _flatten_helper(L, N, return_batch)
            masks_batch = _flatten_helper(L, N, masks_batch)
            high_masks_batch = _flatten_helper(L, N, high_masks_batch)
            old_action_log_probs_batch = _flatten_helper(L, N, \
                    old_action_log_probs_batch)
            adv_targ = _flatten_helper(L, N, adv_targ)
            
            yield share_obs_batch, obs_batch, recurrent_hidden_states_batch, recurrent_hidden_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, high_masks_batch, old_action_log_probs_batch, adv_targ
 