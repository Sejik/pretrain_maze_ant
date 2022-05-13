from collections import OrderedDict
from torch import distributions as pyd
from torch.distributions import Independent

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb
import copy
import utils

# TODO: (BK) image based, MUJOCO에 대해서는 코드 실험해보지 않았음
# TODO: (BK) sas.yaml은 아예 쓰이지 않는 yaml 파일인가? 그럼 sas에 적용할 parameter도 diayn 밑 다른 yaml 파일에 
# 전부 복사해서 박아야하나?

# pixel-based env에서만 쓰임.
class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())
        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h

# Truncated Normal 대신 Squashed Normal로 action range를 제한한다.
class DiagGaussianActor(nn.Module):
    '''
    (s,z) --self.trunk--policy_layers--> mu, log_std
    '''
    def __init__(self, obs_type, obs_dim, action_dim, feature_dim, hidden_dim, 
                 action_range, log_std_bounds):
        super().__init__()

        self.action_dim = action_dim
        self.action_range = action_range
        self.log_std_bounds = log_std_bounds
        feature_dim = feature_dim if obs_type == 'pixels' else hidden_dim
        
        self.trunk = nn.Sequential(nn.Linear(obs_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())
        
        policy_layers = []
        policy_layers += [
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True)
        ]
        # add additional hidden layer for pixels
        if obs_type == 'pixels':
            policy_layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True)
            ]
        policy_layers += [nn.Linear(hidden_dim, action_dim * 2)]  # (BK) 각각 mu, std

        self.policy = nn.Sequential(*policy_layers)

        self.apply(utils.weight_init)

    def forward(self, obs):
        h = self.trunk(obs)
        mu, log_std = self.policy(h).chunk(2, dim=-1) 

        # log_std가 vanish/explore 하지 않게 하기 위한 디테일
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + (0.5 * (log_std_max-log_std_min) * (log_std + 1))
        std = log_std.exp()

        if (isinstance(mu, torch.Tensor)) and torch.isnan(mu).any():
            import pdb; pdb.set_trace()
        if (isinstance(std, torch.Tensor)) and torch.isnan(std).any(): 
            import pdb; pdb.set_trace()
        
        # (BK) action_range를 고려한 distribution
        base_dist = pyd.Normal(mu, std)
        dist = utils.custom_SquashedNormal(loc=mu, scale=std, dist = base_dist, 
                        action_range=self.action_range, action_dim=self.action_dim)

        base_dist = Independent(base_dist, 1)
        dist = Independent(dist, 1)
        
        return dist, base_dist
    
class Critic(nn.Module):
    '''
    (s,z,a) --self.trunk--q_layers--> q-value
    '''
    def __init__(self, obs_type, obs_dim, action_dim, feature_dim, hidden_dim):
        super().__init__()

        self.obs_type = obs_type

        if obs_type == 'pixels':
            # for pixels actions will be added after trunk
            self.trunk = nn.Sequential(nn.Linear(obs_dim, feature_dim),
                                       nn.LayerNorm(feature_dim), nn.Tanh())
            trunk_dim = feature_dim + action_dim
        else:
            # for states actions come in the beginning
            self.trunk = nn.Sequential(
                nn.Linear(obs_dim + action_dim, hidden_dim),
                nn.LayerNorm(hidden_dim), nn.Tanh())
            trunk_dim = hidden_dim

        def make_q():
            q_layers = []
            q_layers += [
                nn.Linear(trunk_dim, hidden_dim),
                nn.ReLU(inplace=True)
            ]
            if obs_type == 'pixels':
                q_layers += [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(inplace=True)
                ]
            q_layers += [nn.Linear(hidden_dim, 1)]
            return nn.Sequential(*q_layers)

        self.Q1 = make_q()
        self.Q2 = make_q()

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        inpt = obs if self.obs_type == 'pixels' else torch.cat([obs, action],
                                                               dim=-1)
        h = self.trunk(inpt)
        h = torch.cat([h, action], dim=-1) if self.obs_type == 'pixels' else h

        q1 = self.Q1(h)
        q2 = self.Q2(h)

        return q1, q2

class SACAgent:
    def __init__(self,
                 name,
                 dtype,
                 maze_type,
                 reward_free,
                 obs_type,
                 obs_shape,
                 action_shape,
                 action_range,
                 device,
                 encoder_lr, 
                 diayn_lr,
                 actor_lr,
                 critic_lr,
                 alpha_lr,
                 init_alpha,
                 feature_dim,
                 hidden_dim,
                 critic_target_tau,
                 num_expl_steps,
                 update_every_steps,
                 critic_target_update_frequency,
                 actor_update_frequency,
                 nstep,
                 batch_size,
                 init_critic,
                 use_tb,
                 use_wandb,
                 log_std_bounds,
                 meta_dim=0,
                 **kwargs):
        self.dtype = dtype
        self.maze_type = maze_type
        self.reward_free = reward_free
        self.obs_type = obs_type
        self.obs_shape = obs_shape
        self.action_dim = action_shape[0]
        self.action_range = action_range
        self.hidden_dim = hidden_dim
        self.encoder_lr = encoder_lr
        self.diayn_lr = diayn_lr
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.alpha_lr = alpha_lr
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.init_alpha = init_alpha
        self.update_every_steps = update_every_steps
        self.critic_target_update_frequency = critic_target_update_frequency
        self.actor_update_frequency = actor_update_frequency
        self.batch_size = batch_size
        self.use_tb = use_tb
        self.use_wandb = use_wandb
        self.log_std_bounds = log_std_bounds
        self.num_expl_steps = num_expl_steps
        self.init_critic = init_critic
        self.feature_dim = feature_dim
        # models
        if obs_type == 'pixels':
            self.aug = utils.RandomShiftsAug(pad=4)
            self.encoder = Encoder(obs_shape).to(device)
            self.obs_dim = self.encoder.repr_dim + meta_dim
        else:
            self.aug = nn.Identity()
            self.encoder = nn.Identity()
            self.obs_dim = obs_shape[0] + meta_dim  # DIAYN에서 meta_dim은 skill_dim으로 init됨.

        # Actor
        self.actor = DiagGaussianActor(obs_type, self.obs_dim, self.action_dim,
                                       feature_dim, hidden_dim, action_range, 
                                       log_std_bounds).to(device)
        
        # Critic, Critic target
        self.critic = Critic(obs_type, self.obs_dim, self.action_dim,
                             feature_dim, hidden_dim).to(device)
        
        self.critic_target = Critic(obs_type, self.obs_dim, self.action_dim,
                             feature_dim, hidden_dim).to(device)
        
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Alpha
        self.log_alpha = torch.tensor(np.log(init_alpha)).to(device)
        self.log_alpha.requires_grad = True
        self.target_entropy = -self.action_dim 
        
        # Optimizers
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.log_alpha_opt = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        
        if obs_type == 'pixels':
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(),
                                                lr=encoder_lr)
        else:
            self.encoder_opt = None

        self.train()
        self.critic_target.train()

        if self.dtype == 'float64':
            dtype = torch.float64
        elif self.dtype == 'float32':
            dtype = torch.float32
        self.actor.to(dtype)
        self.critic.to(dtype)
        self.critic_target.to(dtype)
        self.log_alpha.to(dtype)

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)
    
    @property
    def alpha(self):
        return self.log_alpha.exp()
    

    # TODO: fine-tuning 시에 쓰는 것 같아서 안건드렸음
    def init_from(self, other):
        ipdb.set_trace()
        # copy parameters over
        utils.hard_update_params(other.encoder, self.encoder)
        utils.hard_update_params(other.diayn, self.diayn)
        utils.hard_update_params(other.actor, self.actor)
        utils.hard_update_params(other.actor_target, self.actor_target)
        utils.hard_update_params(other.critic, self.critic)
        utils.hard_update_params(other.critic_target, self.critic_target)
        
        if self.init_critic:
            utils.soft_update_params(other.value.trunk, self.value.trunk)
    
    def get_meta_specs(self):
        return tuple()

    def init_meta(self):
        return OrderedDict()

    def update_meta(self, meta, global_step, time_step, finetune=False):
        return meta

    def act(self, obs, meta, step, eval_mode, eps=1e-2):
        # 1. Obs --encoder--> e(obs) --concat--> [e(obs), skill_index]
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        h = self.encoder(obs)  # maze_env에서는 encoder = Identity()
        inputs = [h]
        for value in meta.values():
            value = torch.as_tensor(value, device=self.device).unsqueeze(0)
            inputs.append(value)
        inpt = torch.cat(inputs, dim=-1)

        # 2. [e(obs), skill_index] --actor--> dist
        dist, _ = self.actor(inpt)  # distribution은 더이상 factored Gaussian이 아님

        # 3. action_range 범위 안에 들어가도록 함 + 초반에는 random action.
        if eval_mode:
            #action = dist.mean  
            action = dist.sample() # For Visualization TODO: why?
        else:
            action = dist.sample()
            if step < self.num_expl_steps:
                action.uniform_(-self.action_range, self.action_range)
        assert ((action>self.action_range) | (action<-self.action_range)).sum() == 0, "action range error"
        
        action[action == 1.0] = 1.0 - eps
        action[action == -1.0] = -1.0 + eps
        
        return action.cpu().numpy()[0]

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        # Q(S_(t+1), A_(t+1))을 계산할 때 A_(t+1)을 critic_target이 아니라 critic으로부터 뽑음
        # Different from DQN
        dist, _ = self.actor(next_obs)
        next_action = dist.rsample()  # reparameterized sampling for backprop
        log_prob = dist.log_prob(next_action).unsqueeze(1)  # continuous R.V -> log_prob>1 OK
        
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1, target_Q2) - (self.alpha.detach() * log_prob)
        target_Q = reward + (discount * target_V)
        target_Q = target_Q.detach()

        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) +\
                        F.mse_loss(current_Q2, target_Q)
        
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        
        if self.use_tb or self.use_wandb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = current_Q1.mean().item()
            metrics['critic_q2'] = current_Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize critic
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad()
            self.encoder_opt.step()

        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        dist, _ = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).unsqueeze(1)
        actor_Q1, actor_Q2 = self.critic(obs, action)
        
        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
        
        # optimize actor
        # self.actor_opt.zero_grad(set_to_none=True)  FIXME:
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # optimize alpha
        self.log_alpha_opt.zero_grad()
        alpha_loss.backward()
        self.log_alpha_opt.step()
            
        if self.use_tb or self.use_wandb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['alpha_loss'] = alpha_loss.item()
            metrics['alpha_value'] = self.alpha.item()

        return metrics

    # TODO: image based obs에만 쓰일거라 안건듬
    def aug_and_encode(self, obs):
        obs = self.aug(obs)
        return self.encoder(obs)

    # TODO: 어차피 안쓰일거기 때문에 수정하지 않음
    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(
            batch, self.device)

        # augment and encode
        obs = self.aug_and_encode(obs)
        with torch.no_grad():
            next_obs = self.aug_and_encode(next_obs)

        if self.use_tb or self.use_wandb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step))

        # update actor
        metrics.update(self.update_actor_and_alpha(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.value, self.value_target,
                                 self.critic_target_tau)

        return metrics