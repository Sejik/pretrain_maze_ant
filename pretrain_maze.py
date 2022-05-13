import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import ipdb
import hydra
import numpy as np
import torch
import wandb
import imageio
from collections import OrderedDict
from tqdm import tqdm

import utils
import omegaconf
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder
import envs.make_maze as make_maze
from dm_env import specs

torch.backends.cudnn.benchmark = True



def make_agent(obs_type, obs_spec, action_spec, action_range, num_expl_steps, cfg):
    
    cfg.obs_type = obs_type
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    cfg.action_range = action_range
    cfg.num_expl_steps = num_expl_steps
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        self.maze_type = cfg.maze_type
        self.dtype = cfg.dtype
        self.sibling_epsilon = cfg.sibling_epsilon
        
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        exp_name = '_'.join([
                cfg.agent.name, cfg.maze_type, 
                str(cfg.agent.skill_dim)
            ])
        self.exp_name = exp_name
        # create logger
        self.logger = Logger(self.work_dir,
                             use_tb=cfg.use_tb,
                             use_wandb=cfg.use_wandb)
        # create envs
        if cfg.maze_type in ['AntU','AntFb','AntMaze']:
            self.train_env0, _ = \
                        make_maze.make_antmaze(cfg.maze_type, cfg.maximum_timestep, 
                                                cfg.dtype, is_pretrain=True)
            self.train_env, self.eval_env = \
                        make_maze.make_antmaze(cfg.maze_type, cfg.maximum_timestep, 
                                                cfg.dtype, is_pretrain=True)
            if cfg.sibling_rivalry:
                self.train_env2, _ = \
                        make_maze.make_antmaze(cfg.maze_type, cfg.maximum_timestep, 
                                                cfg.dtype, is_pretrain=True)
                
        else:
            self.train_env0 = make_maze.make(cfg.maze_type, cfg.maximum_timestep)
            self.train_env = make_maze.make(cfg.maze_type, cfg.maximum_timestep)
            self.eval_env = make_maze.make(cfg.maze_type, cfg.maximum_timestep)
            if cfg.sibling_rivalry:
                self.train_env2 = make_maze.make(cfg.maze_type, cfg.maximum_timestep)

        # create agent
        self.agent = make_agent(cfg.obs_type,
                                self.train_env.observation_spec(),
                                self.train_env.action_spec(),
                                self.train_env.action_range(),
                                cfg.num_seed_frames // cfg.action_repeat,
                                cfg.agent)

        # create wandb
        if cfg.use_wandb:
            if cfg.wandb_name == 'None':
                name = self.exp_name
            else:
                name = cfg.wandb_name
            # hydra -> wandb config
            config = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
            wandb.init(project="urlb", group=cfg.agent.name, name=name, config=config)

        # get meta specs
        meta_specs_smm = self.agent.smm.get_meta_specs()
        meta_specs = self.agent.get_meta_specs()
        # create replay buffer
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))

        # create data storage
        self.replay_storage_smm = ReplayBufferStorage(data_specs, meta_specs_smm,
                                                      self.work_dir / 'buffer')
        self.replay_storage = ReplayBufferStorage(data_specs, meta_specs,
                                                  self.work_dir / 'buffer')

        # create replay buffer
        self.replay_loader_smm = make_replay_loader(self.replay_storage_smm,
                                                    cfg.replay_buffer_size,
                                                    cfg.batch_size,
                                                    cfg.replay_buffer_num_workers,
                                                    False, cfg.nstep, cfg.discount)
        self.replay_loader = make_replay_loader(self.replay_storage,
                                                cfg.replay_buffer_size,
                                                cfg.batch_size,
                                                cfg.replay_buffer_num_workers,
                                                False, cfg.nstep, cfg.discount)
        self._replay_iter_smm = None
        self._replay_iter = None

        # create video recorders
        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None,
            camera_id=0 if 'quadruped' not in self.cfg.domain else 2,
            use_wandb=self.cfg.use_wandb)


        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter_smm(self):
        if self._replay_iter_smm is None:
            self._replay_iter_smm = iter(self.replay_loader_smm)
        return self._replay_iter_smm

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def pretrain_eval(self):
        step, episode, total_reward = 0, 0, 0
        meta_all = self.agent.smm.init_all_meta()
        meta = OrderedDict()
        trajectory_all = {}
        total_diayn_rw = 0

        # 한 skill당 몇번 rollout 할건지
        if self.maze_type in ['AntU','AntFb','AntMaze']:
            num_eval_each_skill = 1
        else:
            num_eval_each_skill = 5

        # 총 몇개의 skill을 rollout 할건지
        if self.cfg.agent.name in ['dads', 'aps']:
            num_eval_skills = self.agent.eval_num_skills
        else:
            num_eval_skills = self.agent.smm.z_dim


        for episode in range(self.agent.smm.z_dim):
            meta['z'] = meta_all['z'][episode]
            trajectory = []
            time_step = self.eval_env.reset()
            if (self.maze_type in ['AntU','AntFb','AntMaze']) & (episode<10): #VIDEO
                self.video_recorder.init_ant(self.eval_env, enabled=True)

            for idx in range(num_eval_each_skill):
                time_step = self.eval_env.reset()
                while not time_step.last():
                    with torch.no_grad(), utils.eval_mode(self.agent.smm):
                        action = self.agent.smm.act(time_step.observation,
                                                    meta,
                                                    self.global_step,
                                                    eval_mode=True)

                    time_step = self.eval_env.step(action)
                    if (episode<10) & (self.maze_type in ['AntU','AntFb','AntMaze']): #VIDEO
                        if idx==0:
                            self.video_recorder.record_ant(self.eval_env)
                    trajectory.append([[time_step.prev_observation[0].item(), time_step.observation[0].item()],
                                    [time_step.prev_observation[1].item(), time_step.observation[1].item()]])
                    total_reward += time_step.reward

                # terminal state에서만 diayn_rw를 재서 learned_skill 개수를 구한다
                if self.cfg.agent.name in ['diayn', 'disdain']:
                    with torch.no_grad():
                        diayn_rw = self.agent.compute_intr_reward(
                            torch.tensor(meta['skill']).unsqueeze(0).to(self.device),
                            torch.tensor(time_step.observation).unsqueeze(0).to(self.device)
                        )
                    diayn_rw = diayn_rw.item()
                    total_diayn_rw += diayn_rw
                else: #EDL, DADS, APS는 못재
                    pass

            trajectory_all[episode] = trajectory
            if (self.maze_type in ['AntU','AntFb','AntMaze']) & (episode<10): #VIDEO
                self.video_recorder.save(f'skill_{episode}_frame_{self.global_frame}.mp4')


        save_dir = self.get_dir(f'{self.exp_name}/{self.exp_name}_{self.global_frame}.png')

        self.eval_env.plot_trajectory(trajectory = trajectory_all,
                                    save_dir = save_dir,
                                    step = self.global_step,
                                    use_wandb = self.cfg.use_wandb)

        # Ant 사진뽑기
        # dummy_img = self.train_env._env.get_image(width=168,height=168)
        # rgb_img = self.train_env._env.get_image(width=168,height=168)
        # plot_img = self.train_env._env.get_image_plt(imsize=400, draw_walls=True, draw_state=True, draw_goal=False, draw_subgoals=False)
        # imageio.imwrite('abcd.png', rgb_img)

        # check state coverage (10x10 격자를 몇개 채웠는지)
        state_coveraged_1 = self.eval_env.state_coverage_1(trajectory_all=trajectory_all,
                                                           skill_dim=self.agent.smm.z_dim)
        state_coveraged_2 = self.eval_env.state_coverage_2(trajectory_all=trajectory_all,
                                                           skill_dim=self.agent.smm.z_dim)
        num_learned_skills = np.exp(total_diayn_rw / (self.agent.skill_dim * num_eval_each_skill))

        if self.maze_type in ['AntU','AntFb','AntMaze']:
            num_bucket = 150
        else:
            num_bucket = 100
        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / (episode*num_eval_each_skill))
            # log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)
            log(f'state_coveraged(out of {num_bucket} bucekts)', state_coveraged_1)
            log(f'new_state_coveraged(out of {num_bucket} bucekts)', state_coveraged_2)
            log('num_learned_skills', num_learned_skills)


    def eval(self):
        step, episode, total_reward = 0, 0, 0
        meta_all = self.agent.init_all_meta()
        meta = OrderedDict()
        trajectory_all = {}
        total_diayn_rw = 0

        # 한 skill당 몇번 rollout 할건지
        if self.maze_type in ['AntU','AntFb','AntMaze']:
            num_eval_each_skill = 1
        else:
            num_eval_each_skill = 5 

        # 총 몇개의 skill을 rollout 할건지
        if self.cfg.agent.name in ['dads', 'aps']:
            num_eval_skills = self.agent.eval_num_skills
        else:
            num_eval_skills = self.agent.skill_dim


        for episode in range(num_eval_skills):
            meta['skill'] = meta_all['skill'][episode]
            trajectory = []
            goal = self.agent.vae.get_centroids(torch.tensor(episode).to(self.device)).detach().cpu()
            time_step = self.eval_env.reset(goal=goal)
            if (self.maze_type in ['AntU','AntFb','AntMaze']) & (episode<10): #VIDEO
                self.video_recorder.init_ant(self.eval_env, enabled=True)

            for idx in range(num_eval_each_skill): 
                time_step = self.eval_env.reset(goal=goal)
                while not time_step.last():
                    with torch.no_grad(), utils.eval_mode(self.agent):
                        action = self.agent.act(time_step.observation,
                                                meta,
                                                self.global_step,
                                                eval_mode=True)
                    
                    time_step = self.eval_env.step(action)
                    if (episode<10) & (self.maze_type in ['AntU','AntFb','AntMaze']): #VIDEO
                        if idx==0:
                            self.video_recorder.record_ant(self.eval_env)
                    trajectory.append([[time_step.prev_observation[0].item(), time_step.observation[0].item()], 
                                    [time_step.prev_observation[1].item(), time_step.observation[1].item()]])
                    total_reward += time_step.reward

                # terminal state에서만 diayn_rw를 재서 learned_skill 개수를 구한다
                if self.cfg.agent.name in ['diayn', 'disdain']:
                    with torch.no_grad():
                        diayn_rw = self.agent.compute_intr_reward(
                            torch.tensor(meta['skill']).unsqueeze(0).to(self.device),
                            torch.tensor(time_step.observation).unsqueeze(0).to(self.device)
                        )
                    diayn_rw = diayn_rw.item()
                    total_diayn_rw += diayn_rw
                else: #EDL, DADS, APS는 못재
                    pass

            trajectory_all[episode] = trajectory 
            if (self.maze_type in ['AntU','AntFb','AntMaze']) & (episode<10): #VIDEO
                self.video_recorder.save(f'skill_{episode}_frame_{self.global_frame}.mp4')


        save_dir = self.get_dir(f'{self.exp_name}/{self.exp_name}_{self.global_frame}.png')
        
        self.eval_env.plot_trajectory(trajectory = trajectory_all, 
                                    save_dir = save_dir,
                                    step = self.global_step,
                                    use_wandb = self.cfg.use_wandb)
        
        # Ant 사진뽑기
        # dummy_img = self.train_env._env.get_image(width=168,height=168)
        # rgb_img = self.train_env._env.get_image(width=168,height=168)
        # plot_img = self.train_env._env.get_image_plt(imsize=400, draw_walls=True, draw_state=True, draw_goal=False, draw_subgoals=False)
        # imageio.imwrite('abcd.png', rgb_img)

        # check state coverage (10x10 격자를 몇개 채웠는지)
        state_coveraged_1 = self.eval_env.state_coverage_1(trajectory_all=trajectory_all,
                                                           skill_dim=self.agent.skill_dim)
        state_coveraged_2 = self.eval_env.state_coverage_2(trajectory_all=trajectory_all,
                                                           skill_dim=self.agent.skill_dim)
        num_learned_skills = np.exp(total_diayn_rw / (self.agent.skill_dim * num_eval_each_skill))

        if self.maze_type in ['AntU','AntFb','AntMaze']:
            num_bucket = 150
        else:
            num_bucket = 100
        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / (episode*num_eval_each_skill))
            # log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)
            log(f'state_coveraged(out of {num_bucket} bucekts)', state_coveraged_1)
            log(f'new_state_coveraged(out of {num_bucket} bucekts)', state_coveraged_2)
            log('num_learned_skills', num_learned_skills)

    def distance_reward(self, transition, goal, antigoal):
        if self.cfg.maze_type in ['AntU','AntFb','AntMaze']:
            drg = -self.dist(torch.tensor(transition.observation[:2]), goal)
            dra = -self.dist(torch.tensor(transition.observation[:2]), antigoal)
        else:
            drg = -self.train_env._env.dist(torch.tensor(transition.observation), goal)
            dra = -self.train_env._env.dist(torch.tensor(transition.observation), antigoal)
        return torch.clamp(drg - dra, -np.inf, 0)

    def sample_dataset(self, env, condition_fn=lambda x: True):
        dataset = np.zeros((self.cfg.oracle_num_samples, 2))
        for sample_idx in range(self.cfg.oracle_num_samples):
            done = False
            while not done:
                s = env.sample()
                done = condition_fn(s)
            dataset[sample_idx] = np.array(s)
        dataset = torch.from_numpy(dataset).float().to(self.cfg.device)
        return dataset

    def pretrain(self):
        train_until_step = utils.Until(self.cfg.num_pretrain_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)

        if self.maze_type in ['AntU','AntFb','AntMaze']:
            self.train_env0._env.get_image(width=200, height=200)

        episode_step, episode_reward = 0, 0
        time_step = self.train_env0.reset()
        time_step.action = np.zeros(self.agent.smm.action_dim, dtype=time_step.observation.dtype)
        meta = self.agent.smm.init_meta()
        self.replay_storage_smm.add(time_step, meta)
        metrics = None
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_storage_smm))
                        log('step', self.global_step)

                # reset env
                time_step = self.train_env0.reset()
                time_step.action = np.zeros(self.agent.smm.action_dim, dtype=time_step.observation.dtype)
                meta = self.agent.smm.init_meta()
                self.replay_storage_smm.add(time_step, meta)
                # try to save snapshot
                if self.global_frame in self.cfg.snapshots:
                    self.save_snapshot()
                episode_step = 0
                episode_reward = 0

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                self.pretrain_eval()

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent.smm):
                action = self.agent.smm.act(time_step.observation,
                                        meta,
                                        self.global_step,
                                        eval_mode=False)

            # try to update the agent
            if not seed_until_step(self.global_step):
                metrics = self.agent.smm.update(self.replay_iter_smm, self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # take env step
            time_step = self.train_env0.step(action)
            time_step.action = action
            episode_reward += time_step.reward
            self.replay_storage_smm.add(time_step, meta)
            episode_step += 1
            self._global_step += 1

    def dist(self, goal, outcome):
        return torch.sqrt(torch.sum(torch.pow(goal - outcome, 2)))

    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)

        # discard dummy_img. 처음 get_image 했을 때 zero-array가 나옴
        if self.maze_type in ['AntU','AntFb','AntMaze']:
            self.train_env._env.get_image(width=200, height=200) 

        # SMM train
        self.pretrain()
        self._global_step = 0
        dataset = list()
        for _ in range(len(self.replay_storage_smm)//self.cfg.batch_size):
            batch = next(self.replay_iter_smm)
            _, _, _, _, obs, _ = batch
            dataset.append(torch.tensor(obs[:, :2]).to(self.cfg.device))
        dataset = torch.stack(dataset).reshape(-1, 2) # self.cfg['agent']['obs_shape'][0])

        # oracle train
        # dataset = self.sample_dataset(self.train_env._env)
        # self.agent.vae.update_normalizer(dataset=dataset)
        indices = list(range(dataset.size(0)))
        for iter_idx in tqdm(range(self.cfg.oracle_dur), desc="Training"):
            # Make batch
            batch_indices = np.random.choice(indices, size=self.cfg.batch_size)
            # only xy for goal
            batch = dict(next_state=dataset[batch_indices])
            metrics = self.agent.update_vae(batch)
            self.logger.log_metrics(metrics, self.global_frame, ty='train')
 
        episode_step, episode_reward = 0, 0
        meta = self.agent.init_meta()
        goal = self.agent.vae.get_centroids(torch.tensor(meta['skill'].argmax()).to(self.cfg.device)).detach().cpu()
        time_step = self.train_env.reset(goal=goal)
        time_step.action = np.zeros(self.agent.action_dim, dtype=time_step.observation.dtype)
        if self.cfg.sibling_rivalry:
            time_step2 = time_step
            self.train_env2.reset(state=time_step.observation, goal=goal)
            self.train_env._env.set_state(self.train_env._env.sim.data.qpos, self.train_env._env.sim.data.qvel)
            time_step2.action = np.zeros(self.agent.action_dim, dtype=time_step.observation.dtype)
        self.replay_storage.add(time_step, meta)
        metrics = None
        _compress_me = list()
        is_success, is_success2 = False, False
        while True: # train_until_step(self.global_step):
            if time_step.last():
                if self.cfg.sibling_rivalry:
                    achieved0 = torch.stack([x[2] for x in _compress_me])
                    achieved1 = torch.stack([x[3] for x in _compress_me])
                    success0 = torch.stack([x[4] for x in _compress_me])
                    success1 = torch.stack([x[5] for x in _compress_me])
                    success0 = True in success0
                    success1 = True in success1
                    # antigoal0 = achieved1
                    # antigoal1 = achieved0

                    # cur_goal = self.train_env._env.goal.detach()
                    # goal = cur_goal.unsqueeze(0).repeat(achieved0.shape[0], 1)
                    cur_goal = goal
                    goal = goal.repeat(achieved0.shape[0], 1)

                    if self.maze_type in ['AntU','AntFb','AntMaze']:
                        is_0_closer = self.dist(goal, achieved0) < self.dist(goal, achieved1)
                        within_epsilon = self.dist(achieved0, achieved1) < self.sibling_epsilon
                    else:
                        is_0_closer = self.train_env._env.dist(goal, achieved0) < self.train_env._env.dist(goal, achieved1)
                        within_epsilon = self.train_env._env.dist(achieved0, achieved1) < self.sibling_epsilon

                    if is_0_closer:
                        include0 = bool(within_epsilon) or success0
                        include1 = True
                    else:
                        include0 = True
                        include1 = bool(within_epsilon) or success1

                    ep_tuples = [
                        (0, [(x[0], x[3], x[6]) for x in _compress_me], include0, cur_goal),
                        (1, [(x[1], x[2], x[6]) for x in _compress_me], include1, cur_goal)
                    ]

                    for ai, ep, include, g in ep_tuples:
                        if include:
                            for time_step, ag, meta in ep:
                                if time_step.last():
                                    time_step.reward *= 0
                                    if (ai == 0 and success0) or (ai == 1 and success1):
                                        time_step.reward += 1
                                    else:
                                        time_step.reward += self.distance_reward(time_step, g, ag).item()
                                else:
                                    time_step.reward *= 0
                                    if (ai == 0 and success0) or (ai == 1 and success1):
                                        time_step.reward += 1

                                self.replay_storage.add(time_step, meta)
                    _compress_me = list()

                self._global_episode += 1
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_storage))
                        log('step', self.global_step)

                # reset env
                meta = self.agent.init_meta()
                goal = self.agent.vae.get_centroids(torch.tensor(meta['skill'].argmax()).to(self.cfg.device)).detach().cpu()
                time_step = self.train_env.reset(goal=goal)
                time_step.action = np.zeros(self.agent.action_dim, dtype=time_step.observation.dtype)
                if self.cfg.sibling_rivalry:
                    time_step2 = self.train_env2.reset(state=time_step.observation, goal=goal)
                    time_step2.action = np.zeros(self.agent.action_dim, dtype=time_step.observation.dtype)
                    if self.cfg.maze_type in ['AntU','AntFb','AntMaze']:
                        diff = time_step.observation[:2] - np.array(goal[0, :2])
                        is_success = (np.linalg.norm(diff, ord=2) < 1.50).astype(int) if not is_success else is_success
                        diff2 = time_step2.observation[:2] - np.array(goal[0, :2])
                        is_success2 = (np.linalg.norm(diff2, ord=2) < 1.50).astype(int) if not is_success2 else is_success2
                        achieved = goal[0] if is_success else torch.tensor(time_step.observation[:2])
                        achieved2 = goal[0] if is_success2 else torch.tensor(time_step2.observation[:2])
                        _compress_me.append([time_step, time_step2,
                                             achieved, achieved2,
                                             torch.tensor(is_success), torch.tensor(is_success2),
                                             meta])
                    else:
                        _compress_me.append([time_step, time_step2,
                                             self.train_env._env.achieved, self.train_env2._env.achieved,
                                             self.train_env._env.is_success, self.train_env2._env.is_success,
                                             meta])
                else:
                    self.replay_storage.add(time_step, meta)
                # try to save snapshot
                if self.global_frame in self.cfg.snapshots:
                    print('snapshot 저장', self.global_frame)
                    self.save_snapshot(pretrain=False)
                episode_step = 0
                episode_reward = 0
                is_success, is_success2 = False, False

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                self.eval()

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation,
                                        meta,
                                        self.global_step,
                                        eval_mode=False)

            # try to update the agent
            if not seed_until_step(self.global_step):
                metrics = self.agent.update(self.replay_iter, self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # take env step
            time_step = self.train_env.step(action)
            time_step.action = action
            episode_reward += time_step.reward
            if self.cfg.sibling_rivalry:
                time_step2 = self.train_env2.step(action)
                time_step2.action = action
                if self.cfg.maze_type in ['AntU','AntFb','AntMaze']:
                    diff = time_step.observation[:2] - np.array(goal[0, :2])
                    is_success = (np.linalg.norm(diff, ord=2) < 1.50).astype(int) if not is_success else is_success
                    diff2 = time_step2.observation[:2] - np.array(goal[0, :2])
                    is_success2 = (np.linalg.norm(diff2, ord=2) < 1.50).astype(int) if not is_success2 else is_success2
                    achieved = goal[0] if is_success else torch.tensor(time_step.observation[:2])
                    achieved2 = goal[0] if is_success2 else torch.tensor(time_step2.observation[:2])
                    _compress_me.append([time_step, time_step2,
                                         achieved, achieved2,
                                         torch.tensor(is_success), torch.tensor(is_success2),
                                         meta])
                else:
                    _compress_me.append([time_step, time_step2,
                                         self.train_env._env.achieved, self.train_env2._env.achieved,
                                         self.train_env._env.is_success, self.train_env2._env.is_success,
                                         meta])
            else:
                self.replay_storage.add(time_step, meta)
            episode_step += 1
            self._global_step += 1

    def save_snapshot(self, pretrain=True):
        snapshot_dir = self.work_dir / Path(self.cfg.snapshot_dir)
        snapshot_dir.mkdir(exist_ok=True, parents=True)
        if pretrain:
            snapshot = snapshot_dir / f'snapshot_pretrain_{self.global_frame}.pt'
        else:
            snapshot = snapshot_dir / f'snapshot_{self.global_frame}.pt'
        keys_to_save = ['agent', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)
            
    def get_dir(self, file_name):
        save_dir = self.work_dir / 'eval_video'
        dir_name = file_name.split('/')[0]
        if not os.path.exists(save_dir / dir_name):
            os.makedirs(save_dir / dir_name)
        path = save_dir / file_name

        return path


@hydra.main(config_path='.', config_name='pretrain_maze')
def main(cfg):
    from pretrain_maze import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()


if __name__ == '__main__':
    main()
