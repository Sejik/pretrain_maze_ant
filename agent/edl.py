
import math
from collections import OrderedDict

import ipdb
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dm_env import specs

import utils
from agent.vae import VQVAEDiscriminator
from agent.ddpg import DDPGAgent
from agent.sac import SACAgent
from agent.smm import SMMAgent


class EDLAgent(SACAgent):
    def __init__(self, update_skill_every_step, skill_dim, diayn_scale,
                 update_encoder, max_skill_dim, **kwargs):
        self.skill_dim = skill_dim
        self.max_skill_dim = max_skill_dim
        self.update_skill_every_step = update_skill_every_step
        self.diayn_scale = diayn_scale
        self.update_encoder = update_encoder
        # increase obs shape to include skill dim
        kwargs["meta_dim"] = self.max_skill_dim

        # create actor and critic
        super().__init__(**kwargs)
        
        kwargs['smm_args']['update_encoder'] = update_encoder
        for key, value in kwargs.items():
            if key in ['name', 'reward_free', 'obs_type', 'obs_shape', 'action_shape', 'device', 'num_expl_steps', 'use_tb', 'use_wandb']:
                kwargs['smm_args'][key] = value
        self.smm = SMMAgent(**kwargs['smm_args'])
        
        # create vae : decoder, skill, (encoder)
        # self.obs_dim = kwargs['obs_sha[e'][0] + self.max_dkill_dim  # DIAYN에서 meta_dim은 skill_dim으로 init됨.
 
        # self.vae = VQVAEDiscriminator(state_size=kwargs['obs_shape'][0], **kwargs["vae_args"]).to(kwargs['device'])
        self.vae = VQVAEDiscriminator(state_size=2, **kwargs["vae_args"]).to(kwargs['device'])


        # optimizers
        self.vae_opt = torch.optim.Adam(self.vae.parameters(), lr=kwargs['vae_lr'])

        self.vae.train()

        # dtype
        if self.dtype == 'float64':
            self.vae.to(torch.float64)
        elif self.dtype == 'float32':
            self.vae.to(torch.float32)

    def get_meta_specs(self):
        return (specs.Array((self.max_skill_dim,), np.float32, 'skill'),)

    def init_meta(self):
        skill = np.zeros(self.max_skill_dim, dtype=np.float32)
        skill[np.random.choice(self.skill_dim)] = 1.0
        meta = OrderedDict()
        meta['skill'] = skill
        return meta
    
    def init_all_meta(self):
        skill = np.eye(self.max_skill_dim, dtype=np.float32)
        meta = OrderedDict()
        meta['skill'] = skill
        return meta

    def update_meta(self, meta, global_step, time_step):
        if global_step % self.update_skill_every_step == 0:
            return self.init_meta()
        return meta
    
    def update_vae(self, batch):
        metrics = dict()
        
        self.vae_opt.zero_grad()
        loss = self.vae(batch)
        loss.backward()
        self.vae_opt.step()
        
        if self.use_tb or self.use_wandb:
            metrics['pretrain_reward'] = loss.item()
        
        return metrics

    def compute_intr_reward(self, skill, obs):
        z_hat = torch.argmax(skill, dim=1)
        
        return self.vae.compute_intr_reward(z_hat, obs) 

    def compute_diayn_loss(self, next_state, skill):
        """
        DF Loss
        """
        z_hat = torch.argmax(skill, dim=1)
        d_pred = self.diayn(next_state)
        d_pred[:, self.skill_dim:] = float('-inf')  
        d_pred_log_softmax = F.log_softmax(d_pred, dim=1)
        _, pred_z = torch.max(d_pred_log_softmax, dim=1, keepdim=True)
        d_loss = self.diayn_criterion(d_pred, z_hat)
        df_accuracy = torch.sum(
            torch.eq(z_hat,
                     pred_z.reshape(1,
                                    list(
                                        pred_z.size())[0])[0])).float() / list(
                                            pred_z.size())[0]
        return d_loss, df_accuracy

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)

        obs, action, extr_reward, discount, next_obs, skill = utils.to_torch(
            batch, self.device)

        # augment and encode
        obs = self.aug_and_encode(obs)
        next_obs = self.aug_and_encode(next_obs)

        if self.reward_free:

            with torch.no_grad():
                # vae
                z_hat = torch.argmax(skill, dim=1)
                # intr_reward = self.vae.compute_intr_reward(z_hat, next_obs)
                intr_reward = self.vae.compute_intr_reward(z_hat, next_obs[:, :2]) 

            if self.use_tb or self.use_wandb:
                metrics['intr_reward'] = intr_reward.mean().item()
            reward = intr_reward + extr_reward
        else:
            reward = extr_reward

        if self.use_tb or self.use_wandb:
            metrics['extr_reward'] = extr_reward.mean().item()
            metrics['batch_reward'] = reward.mean().item()

        if not self.update_encoder:
            obs = obs.detach()
            next_obs = next_obs.detach()

        # extend observations with skill
        obs = torch.cat([obs, skill], dim=1)
        next_obs = torch.cat([next_obs, skill], dim=1)

        # update critic
        metrics.update(
            self.update_critic(obs.detach(), action, reward, discount,
                               next_obs.detach(), step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics
