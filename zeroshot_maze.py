import ipdb
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import hydra
import numpy as np
import torch
import wandb
from dm_env import specs
import omegaconf
from collections import OrderedDict

import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder

import envs.make_maze as make_maze


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
            self.train_env, self.eval_env = \
                        make_maze.make_antmaze(cfg.maze_type, cfg.maximum_timestep, 
                                                cfg.dtype, is_pretrain=False)
        else:
            self.train_env = make_maze.make(cfg.maze_type, cfg.maximum_timestep)
            self.eval_env = make_maze.make(cfg.maze_type, cfg.maximum_timestep)


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

        # initialize from pretrained
        if cfg.snapshot_ts > 0:
            pretrained_agent = self.load_snapshot()['agent']
            self.agent.init_from(pretrained_agent)

        # get meta specs
        meta_specs = self.agent.get_meta_specs()
        # create replay buffer
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))

        # create data storage
        self.replay_storage = ReplayBufferStorage(data_specs, meta_specs,
                                                  self.work_dir / 'buffer')

        # create replay buffer
        self.replay_loader = make_replay_loader(self.replay_storage,
                                                cfg.replay_buffer_size,
                                                cfg.batch_size,
                                                cfg.replay_buffer_num_workers,
                                                False, cfg.nstep, cfg.discount)
        self._replay_iter = None

        # create video recorders
        self.video_recorder = VideoRecorder(
            self.work_dir.parents[2] / 'downstream_video' /  self.cfg.agent.name / ('seed_'+str(self.cfg.snapshot_ts)))

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
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    # TODO: DIAYN에서 돌아가는 것 확인
    def eval_ant(self):
        episode, total_rw = 0, 0
        eval_until_episode = utils.Until(self.cfg.how_many_goals)
        meta_all = self.agent.init_all_meta()
        meta = OrderedDict()

        # dist_threshold = self.eval_env._env.dist_threshold
        # ant_maze의 threshold: ant의 크기(=0.75)

        meta = self.agent.init_meta()
        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            eval_goal = torch.tensor(self.eval_env._env._cur_obs['state_desired_goal'][:2]).cuda()
            print('eval_goal:',eval_goal.cpu().numpy().tolist())
            argmax_skill = torch.argmax(self.agent.diayn(eval_goal.float())[:self.agent.skill_dim])
            meta['skill'] = meta_all['skill'][argmax_skill]
            rw = 0

            self.video_recorder.init_ant(self.eval_env, enabled=True)
            while not time_step.last():  # TODO: 200맞나? 너무 빨리끝나는데 영상이
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            meta,
                                            self.global_step,
                                            eval_mode=True)
                time_step = self.eval_env.step(action)
                d_eval_goal = np.sqrt(np.square(time_step.observation[:2]-eval_goal.cpu().numpy()).sum())
                if d_eval_goal < 0.75: 
                    rw = 1
                    break

            total_rw += rw
            episode += 1
            self.video_recorder.save(f'skill_{episode}.mp4')  # TODO: 이거 제목 바꿔줘야됨
            print(f'total_rw:{total_rw}/{self.cfg.how_many_goals}')
            print(episode)
            if episode==10:
                ipdb.set_trace()

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log(f'num_success (out of {self.cfg.how_many_goals}', total_rw)



    # TODO: DIAYN은 돌아가는것 확인
    # APS는 regress 추가해야함
    def eval_2dmaze(self):
        episode, total_rw = 0, 0
        eval_until_episode = utils.Until(self.cfg.how_many_goals)
        meta_all = self.agent.init_all_meta()
        meta = self.agent.init_meta()

        dist_threshold = self.eval_env._env.dist_threshold

        while eval_until_episode(episode):
            # TODO: 우리 모델 skill 최종 개수도 skill_dim으로 하면 되나?
            time_step = self.eval_env.reset()
            eval_goal = time_step._state['eval_goal'].cuda()
            argmax_skill = torch.argmax(self.agent.diayn(eval_goal)[:self.agent.skill_dim])
            meta['skill'] = meta_all['skill'][argmax_skill]
            rw = 0

            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            meta,
                                            self.global_step,
                                            eval_mode=True)
                time_step = self.eval_env.step(action)
                d_eval_goal = time_step._state['d_eval_goal']
                if d_eval_goal.item()< dist_threshold:
                    rw = 1
                    break

            total_rw += rw
            episode += 1
            print(f'total_rw:{total_rw}/{self.cfg.how_many_goals}')

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log(f'num_success (out of {self.cfg.how_many_goals}', total_rw)


    def load_snapshot(self):
        snapshot_base_dir = Path.cwd().parents[2] / Path(self.cfg.snapshot_base_dir)
        # domain, _ = self.cfg.task.split('_', 1)
        # snapshot_dir = snapshot_base_dir / self.cfg.obs_type / domain / self.cfg.agent.name

        snapshot = snapshot_base_dir / f'snapshot_{self.cfg.snapshot_ts}.pt'
        assert snapshot.exists(), "snapshot 없는데요"
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        return payload




@hydra.main(config_path='.', config_name='zeroshot_maze')
def main(cfg):
    from zeroshot_maze import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    if cfg.maze_type in ['square_bottleneck', 'square_upside']:
        workspace.eval_2dmaze()
    else:
        workspace.eval_ant()


if __name__ == '__main__':
    main()
