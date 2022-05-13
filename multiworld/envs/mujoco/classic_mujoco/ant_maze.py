import numpy as np
import ipdb
import matplotlib.pyplot as plt
import wandb


from multiworld.envs.mujoco.classic_mujoco.ant import AntEnv

plot_colors = ['blue', 'red', 'green', 'brown', 'orange', 'pink', 'rebeccapurple', 'olive', 'purple', 'cyan', 
               'black', 'maroon', 'yellow', 'navy', 'darkgreen', 'lavender', 'peru', 'fuchsia', 'slateblue', 'oldlace',
               'darkviolet', 'silver', 'lightcoral', 'mistyrose', 'salmon', 'tomato', 'chocolate', 'bisque', 'burlywood', 'tan',
               'darkgoldenrod', 'darkkhaki', 'olivedrab', 'darkseagreen', 'lightgreen', 'limegreen', 'lime', 'aquamarine', 'turquoise', 'teal',
               'deepskyblue', 'steelblue', 'dodgerblue', 'cornflowerblue', 'royalblue', 'mediumpurple', 'blueviolet', 'indigo', 'plum', 'palevioletred']
from scipy.interpolate import interp1d


class AntMazeEnv(AntEnv):

    def __init__(
            self,
            # model_path='classic_mujoco/normal_gear_ratio_ant.xml',
            # test_mode_case_num=None,
            *args,
            **kwargs
    ):
        self.ant_radius = 0.75

        wall_collision_buffer = kwargs.get("wall_collision_buffer", 0.0)
        self.wall_radius = self.ant_radius + wall_collision_buffer

        model_path = kwargs['model_path']
        test_mode_case_num = kwargs.get('test_mode_case_num', None)

        if model_path in [
            'classic_mujoco/ant_maze2_gear30_small_dt3.xml',
            'classic_mujoco/ant_gear30_dt3_u_small.xml',  
        ]:
            self.maze_type = 'u-small'
        elif model_path in [
            'classic_mujoco/ant_gear30_dt3_u_med.xml',
            'classic_mujoco/ant_gear15_dt3_u_med.xml',
            'classic_mujoco/ant_gear10_dt3_u_med.xml',
            'classic_mujoco/ant_gear30_dt2_u_med.xml',
            'classic_mujoco/ant_gear15_dt2_u_med.xml',
            'classic_mujoco/ant_gear10_dt2_u_med.xml',
            'classic_mujoco/ant_gear30_u_med.xml',
        ]:
            self.maze_type = 'u-med'
        elif model_path in [
            'classic_mujoco/ant_maze2_gear30_big_dt3.xml',
            'classic_mujoco/ant_gear30_dt3_u_big.xml',
        ]:
            self.maze_type = 'u-big'
        elif model_path in [
            'classic_mujoco/ant_gear10_dt3_u_long.xml',
            'classic_mujoco/ant_gear15_dt3_u_long.xml',
        ]:
            self.maze_type = 'u-long'
        elif model_path in [
            'classic_mujoco/ant_gear10_dt3_no_walls_long.xml',
            'classic_mujoco/ant_gear15_dt3_no_walls_long.xml',
        ]:
            self.maze_type = 'no-walls-long'
        elif model_path == 'classic_mujoco/ant_fb_gear30_small_dt3.xml':
            self.maze_type = 'fb-small'
        elif model_path in [
            'classic_mujoco/ant_fb_gear30_med_dt3.xml',
            'classic_mujoco/ant_gear30_dt3_fb_med.xml',
            'classic_mujoco/ant_gear15_dt3_fb_med.xml',
            'classic_mujoco/ant_gear10_dt3_fb_med.xml',
        ]:
            self.maze_type = 'fb-med'
        elif model_path == 'classic_mujoco/ant_fb_gear30_big_dt3.xml':
            self.maze_type = 'fb-big'
        elif model_path == 'classic_mujoco/ant_fork_gear30_med_dt3.xml':
            self.maze_type = 'fork-med'
        elif model_path == 'classic_mujoco/ant_fork_gear30_big_dt3.xml':
            self.maze_type = 'fork-big'
        elif model_path in [
            'classic_mujoco/ant_gear10_dt3_maze_med.xml',
        ]:
            self.maze_type = 'maze-med'
        elif model_path in [
            'classic_mujoco/ant_gear10_dt3_maze_small.xml',
        ]:
            self.maze_type = 'maze-small'
        elif model_path in [
            'classic_mujoco/ant_gear10_dt3_fg_med.xml',
        ]:
            self.maze_type = 'fg-med'
        else:
            raise NotImplementedError

        if self.maze_type == 'u-small':
            # map 크기
            # 위쪽 벽중심 4.5, 두께1, 따라서 가운데~위벽까지 3.5, 따라서 세로 7
            # 오른쪽 벽중심 4.5, 두께1, 따라서 가운데~오른벽 3.5, 따라서 가로 7
            self.walls = [
                Wall(0, 1.125, 1.25, 2.375, self.wall_radius),

                Wall(0, 4.5, 3.5, 1, self.wall_radius),
                Wall(0, -4.5, 3.5, 1, self.wall_radius),
                Wall(4.5, 0, 1, 5.5, self.wall_radius),
                Wall(-4.5, 0, 1, 5.5, self.wall_radius),
            ]

            # if test_mode_case_num == 1:
            #     kwargs['reset_low'] = np.array([-2.5, 2.5])
            #     kwargs['reset_high'] = np.array([-2.25, 2.5])

            #     kwargs['goal_low'] = np.array([2.25, 2.5])
            #     kwargs['goal_high'] = np.array([2.5, 2.5])

            # train_env, test 같은 위치로 시작위치 고정
            kwargs['reset_low'] = np.array([-2.5, 2.5])
            kwargs['reset_high'] = np.array([-2.25, 2.5])
            # goal은 전체에서 random sampling
            if 'goal_low' not in kwargs:
                kwargs['goal_low'] = np.array([-3.5, -3.5])
            if 'goal_high' not in kwargs:
                kwargs['goal_high'] = np.array([3.5, 3.5])


        elif self.maze_type == 'u-med':
            # map 크기
            # 위쪽 벽중심 5.5, 두께1, 따라서 가운데~위벽까지 4.5, 따라서 세로 9
            # 오른쪽 벽중심 5.5, 두께1, 따라서 가운데~오른벽 4.5, 따라서 가로 9
            self.walls = [
                Wall(0, 1.5, 1.5, 3, self.wall_radius),

                Wall(0, 5.5, 4.5, 1, self.wall_radius),
                Wall(0, -5.5, 4.5, 1, self.wall_radius),
                Wall(5.5, 0, 1, 6.5, self.wall_radius),
                Wall(-5.5, 0, 1, 6.5, self.wall_radius),
            ]

            # if test_mode_case_num == 1:
            #     kwargs['reset_low'] = np.array([-3.25, 2.75])
            #     kwargs['reset_high'] = np.array([-2.75, 3.25])

            #     kwargs['goal_low'] = np.array([2.75, 2.75])
            #     kwargs['goal_high'] = np.array([3.25, 3.25])

            # train_env, test 같은 위치로 시작위치 고정
            kwargs['reset_low'] = np.array([-3.25, 2.75])
            kwargs['reset_high'] = np.array([-2.75, 3.25])
            # goal은 전체에서 random sampling
            if 'goal_low' not in kwargs:
                kwargs['goal_low'] = np.array([-5.5, -5.5])
            if 'goal_high' not in kwargs:
                kwargs['goal_high'] = np.array([5.5, 5.5])

        elif self.maze_type == 'u-long':
            # map 크기
            # 위쪽 벽의 중심은 10, 두께는 1, 즉 중심~위의 벽까지가 9. 따라서 세로는 18
            # 오른쪽 벽의 중심은 4.75, 두께는 1. 즉 중심 ~오른쪽 벽까지가 3.75. 따라서 가로는 7.5
            self.walls = [
                Wall(0, 1.5, 0.75, 7.5, self.wall_radius),  # 가운데 벽
                Wall(0, 10.0, 3.75, 1, self.wall_radius),   # 위쪽 벽
                Wall(0, -10.0, 3.75, 1, self.wall_radius),  # 아래쪽 벽
                Wall(4.75, 0, 1, 11.0, self.wall_radius),   # 오른쪽 벽
                Wall(-4.75, 0, 1, 11.0, self.wall_radius),  # 왼쪽 벽
            ]

            # if test_mode_case_num == 1:
            #     kwargs['reset_low'] = np.array([-2.5, 7.25])
            #     kwargs['reset_high'] = np.array([-2.0, 7.75])

            #     kwargs['goal_low'] = np.array([2.0, 7.25])
            #     kwargs['goal_high'] = np.array([2.5, 7.75])

            #train_env, test 같은 위치로 시작위치 고정
            kwargs['reset_low'] = np.array([-2.5, 7.25])
            kwargs['reset_high'] = np.array([-2.0, 7.75])

            # goal이 모든곳에서 sampling 되게 하기 위해 주석처리
            # kwargs['goal_low'] = np.array([2.0, 7.25])
            # kwargs['goal_high'] = np.array([2.5, 7.75])


            if 'goal_low' not in kwargs:
                kwargs['goal_low'] = np.array([-3.75, -9.0])
            if 'goal_high' not in kwargs:
                kwargs['goal_high'] = np.array([3.75, 9.0])

        elif self.maze_type == 'no-walls-long':
            self.walls = [
                Wall(0, 10.0, 3.75, 1, self.wall_radius),
                Wall(0, -10.0, 3.75, 1, self.wall_radius),
                Wall(4.75, 0, 1, 11.0, self.wall_radius),
                Wall(-4.75, 0, 1, 11.0, self.wall_radius),
            ]

            if test_mode_case_num == 1:
                kwargs['reset_low'] = np.array([-2.5, -7.75])
                kwargs['reset_high'] = np.array([-2.0, -7.25])

                kwargs['goal_low'] = np.array([2.0, 7.25])
                kwargs['goal_high'] = np.array([2.5, 7.75])

            if 'goal_low' not in kwargs:
                kwargs['goal_low'] = np.array([-3.75, -9.0])
            if 'goal_high' not in kwargs:
                kwargs['goal_high'] = np.array([3.75, 9.0])

        elif self.maze_type == 'fb-small':
            # O~위벽: 5.25, O~오른벽: 5.25 -> 10.5 x 10.5
            self.walls = [
                Wall(-2.0, 1.25, 0.75, 4.0, self.wall_radius),
                Wall(2.0, -1.25, 0.75, 4.0, self.wall_radius),

                Wall(0, 6.25, 5.25, 1, self.wall_radius),
                Wall(0, -6.25, 5.25, 1, self.wall_radius),
                Wall(6.25, 0, 1, 7.25, self.wall_radius),
                Wall(-6.25, 0, 1, 7.25, self.wall_radius),
            ]
            # train_env, test 같은 위치로 시작위치 고정
            kwargs['reset_low'] = np.array([-3.9, 4])
            kwargs['reset_high'] = np.array([-4.1, 4])
            # goal은 전체에서 random sampling
            if 'goal_low' not in kwargs:
                kwargs['goal_low'] = np.array([-5.5, -5.5])
            if 'goal_high' not in kwargs:
                kwargs['goal_high'] = np.array([5.5, 5.5])

        elif self.maze_type == 'fb-med':
            self.walls = [
                Wall(-2.25, 1.5, 0.75, 4.5, self.wall_radius),
                Wall(2.25, -1.5, 0.75, 4.5, self.wall_radius),

                Wall(0, 7.0, 6.0, 1, self.wall_radius),
                Wall(0, -7.0, 6.0, 1, self.wall_radius),
                Wall(7.0, 0, 1, 8.0, self.wall_radius),
                Wall(-7.0, 0, 1, 8.0, self.wall_radius),
            ]

            if test_mode_case_num == 1:
                kwargs['reset_low'] = np.array([-4.75, 4.25])
                kwargs['reset_high'] = np.array([-4.25, 4.75])

                kwargs['goal_low'] = np.array([4.25, -4.75])
                kwargs['goal_high'] = np.array([4.75, -4.25])
            elif test_mode_case_num == 2:
                kwargs['reset_low'] = np.array([-4.75, -0.25])
                kwargs['reset_high'] = np.array([-4.25, 0.25])

                kwargs['goal_low'] = np.array([4.25, -0.25])
                kwargs['goal_high'] = np.array([4.75, -0.25])
            elif test_mode_case_num == 3:
                kwargs['reset_low'] = np.array([-4.75, -4.75])
                kwargs['reset_high'] = np.array([-4.25, -4.25])

                kwargs['goal_low'] = np.array([4.25, 4.25])
                kwargs['goal_high'] = np.array([4.75, 4.75])
            elif test_mode_case_num == 4:
                kwargs['reset_low'] = np.array([-4.75, 4.25])
                kwargs['reset_high'] = np.array([-4.25, 4.75])

                kwargs['goal_low'] = np.array([-0.25, 4.25])
                kwargs['goal_high'] = np.array([0.25, 4.75])
            # custom adds
            elif test_mode_case_num == 10:
                kwargs['reset_low'] = np.array([-4.75, 4.25])
                kwargs['reset_high'] = np.array([-4.25, 4.75])

                kwargs['goal_low']  = np.array([-4.75, -0.25])
                kwargs['goal_high'] = np.array([-4.25, 0.25])
            elif test_mode_case_num == 11:
                kwargs['reset_low'] = np.array([-4.75, 4.25])
                kwargs['reset_high'] = np.array([-4.25, 4.75])

                kwargs['goal_low']  = np.array([-4.75, -4.75])
                kwargs['goal_high'] = np.array([-4.25, -4.25])
            elif test_mode_case_num == 12:
                kwargs['reset_low'] = np.array([-4.75, 4.25])
                kwargs['reset_high'] = np.array([-4.25, 4.75])

                kwargs['goal_low']  = np.array([-0.25, -4.75])
                kwargs['goal_high'] = np.array([ 0.25, -4.25])
            elif test_mode_case_num == 13:
                kwargs['reset_low'] = np.array([-4.75, 4.25])
                kwargs['reset_high'] = np.array([-4.25, 4.75])

                kwargs['goal_low']  = np.array([-0.25, -0.25])
                kwargs['goal_high'] = np.array([ 0.25,  0.25])
            elif test_mode_case_num == 14:
                kwargs['reset_low'] = np.array([-4.75, 4.25])
                kwargs['reset_high'] = np.array([-4.25, 4.75])

                kwargs['goal_low']  = np.array([-0.25, 4.25])
                kwargs['goal_high'] = np.array([ 0.25, 4.75])
            elif test_mode_case_num == 15:
                kwargs['reset_low'] = np.array([-4.75, 4.25])
                kwargs['reset_high'] = np.array([-4.25, 4.75])

                kwargs['goal_low']  = np.array([4.25, 4.25])
                kwargs['goal_high'] = np.array([4.75, 4.75])
            elif test_mode_case_num == 16:
                kwargs['reset_low'] = np.array([-4.75, 4.25])
                kwargs['reset_high'] = np.array([-4.25, 4.75])

                kwargs['goal_low']  = np.array([4.25, -0.25])
                kwargs['goal_high'] = np.array([4.75, 0.25])
            elif test_mode_case_num == 17:
                kwargs['reset_low'] = np.array([-4.75, 4.25])
                kwargs['reset_high'] = np.array([-4.25, 4.75])

                kwargs['goal_low']  = np.array([4.25, -4.75])
                kwargs['goal_high'] = np.array([4.75, -4.25])


            if 'goal_low' not in kwargs:
                kwargs['goal_low'] = np.array([-7.0, -7.0])
            if 'goal_high' not in kwargs:
                kwargs['goal_high'] = np.array([7.0, 7.0])

        elif self.maze_type == 'fb-big':
            self.walls = [
                Wall(-2.75, 2.0, 0.75, 5.5, self.wall_radius),
                Wall(2.75, -2.0, 0.75, 5.5, self.wall_radius),

                Wall(0, 8.5, 7.5, 1, self.wall_radius),
                Wall(0, -8.5, 7.5, 1, self.wall_radius),
                Wall(8.5, 0, 1, 9.5, self.wall_radius),
                Wall(-8.5, 0, 1, 9.5, self.wall_radius),
            ]
        elif self.maze_type == 'fork-med':
            self.walls = [
                Wall(-1.75, -1.5, 0.25, 3.5, self.wall_radius),
                Wall(0, 1.75, 2.0, 0.25, self.wall_radius),
                Wall(0, -1.75, 2.0, 0.25, self.wall_radius),

                Wall(0, 6.0, 5.0, 1, self.wall_radius),
                Wall(0, -6.0, 5.0, 1, self.wall_radius),
                Wall(6.0, 0, 1, 7.0, self.wall_radius),
                Wall(-6.0, 0, 1, 7.0, self.wall_radius),
            ]

            if test_mode_case_num == 1:
                kwargs['reset_low'] = np.array([-0.25, -3.75])
                kwargs['reset_high'] = np.array([0.25, -3.25])

                kwargs['goal_low'] = np.array([-3.75, -3.75])
                kwargs['goal_high'] = np.array([-3.25, -3.25])
            elif test_mode_case_num == 2:
                kwargs['reset_low'] = np.array([-0.25, -0.25])
                kwargs['reset_high'] = np.array([0.25, 0.25])

                kwargs['goal_low'] = np.array([-3.75, -0.25])
                kwargs['goal_high'] = np.array([-3.25, 0.25])

        elif self.maze_type == 'fork-big':
            self.walls = [
                Wall(-3.5, -1.5, 0.25, 5.25, self.wall_radius),
                Wall(0, -3.5, 3.75, 0.25, self.wall_radius),
                Wall(0, 0.0, 3.75, 0.25, self.wall_radius),
                Wall(0, 3.5, 3.75, 0.25, self.wall_radius),

                Wall(0, 7.75, 6.75, 1, self.wall_radius),
                Wall(0, -7.75, 6.75, 1, self.wall_radius),
                Wall(7.75, 0, 1, 8.75, self.wall_radius),
                Wall(-7.75, 0, 1, 8.75, self.wall_radius),
            ]

            if test_mode_case_num == 1:
                kwargs['reset_low'] = np.array([-5.5, -5.5])
                kwargs['reset_high'] = np.array([-5.0, -5.0])

                kwargs['goal_low'] = np.array([-5.5, -5.5])
                kwargs['goal_high'] = np.array([-5.0, -5.0])

        elif self.maze_type == 'maze-small':

            # O~위벽: 4, 0~ 오른쪽벽: 4, 따라서 8 x 8 맵
            self.walls = [
                Wall(1.1875, 1.625, 1.0875, 0.1, self.wall_radius),  # 십자가의 오른쪽가로
                Wall(-1.1875, 1.625, 1.0875, 0.1, self.wall_radius),  # 십자가의 왼쪽가로
                Wall(0, 1, 0.1, 3, self.wall_radius),  # 십자가의 세로
                Wall(3, -1.125, 1, 0.1, self.wall_radius),  #오른쪽
                Wall(-3, -1.125, 1, 0.1, self.wall_radius), #왼쪽

                Wall(0, 4.5, 4, 0.5, self.wall_radius),  # 위벽
                Wall(0, -4.5, 4, 0.5, self.wall_radius),  # 아래벽
                Wall(4.5, 0, 0.5, 5, self.wall_radius),  # 오른벽
                Wall(-4.5, 0, 0.5, 5, self.wall_radius),  # 왼벽
            ]


            #train_env, test 같은 위치로 시작위치 고정
            kwargs['reset_low'] = np.array([-1.5, 3.125])
            kwargs['reset_high'] = np.array([-1.25, 3.875])

            # goal이 모든곳에서 sampling 되게 하기 위해 주석처리
            # kwargs['goal_low'] = np.array([2.0, 7.25])
            # kwargs['goal_high'] = np.array([2.5, 7.75])

            if 'goal_low' not in kwargs:
                kwargs['goal_low'] = np.array([-4.0, -4.0])
            if 'goal_high' not in kwargs:
                kwargs['goal_high'] = np.array([4.0, 4.0])

        elif self.maze_type == 'maze-med':

            # O~위벽: 8, 0~ 오벽: 8, 따라서 16 x 16 맵
            self.walls = [
                Wall(2.375, 3.25, 1.625, 0.75, self.wall_radius),
                Wall(-2.375, 3.25, 1.625, 0.75, self.wall_radius),
                Wall(0, 2, 0.75, 6, self.wall_radius),
                Wall(6, -2.25, 2, 0.75, self.wall_radius),
                Wall(-6, -2.25, 2, 0.75, self.wall_radius),

                Wall(0, 9, 8, 1, self.wall_radius),
                Wall(0, -9, 8, 1, self.wall_radius),
                Wall(9, 0, 1, 10, self.wall_radius),
                Wall(-9, 0, 1, 10, self.wall_radius),
            ]

            if test_mode_case_num == 1:
                kwargs['reset_low'] = np.array([-2.5, 6.25])
                kwargs['reset_high'] = np.array([-2.0, 6.75])

                kwargs['goal_low'] = np.array([2.0, 6.25])
                kwargs['goal_high'] = np.array([2.5, 6.75])
            elif test_mode_case_num == 2:
                kwargs['reset_low'] = np.array([0.75, 6.25])
                kwargs['reset_high'] = np.array([1.25, 6.75])

                kwargs['goal_low'] = np.array([0.75, 6.25])
                kwargs['goal_high'] = np.array([1.25, 6.75])

            if 'goal_low' not in kwargs:
                kwargs['goal_low'] = np.array([-8.0, -8.0])
            if 'goal_high' not in kwargs:
                kwargs['goal_high'] = np.array([8.0, 8.0])

        elif self.maze_type == 'fg-med':
            self.walls = [
                Wall(0, -3.25, 3, 0.75, self.wall_radius),
                Wall(-3.75, 0, 0.75, 4, self.wall_radius),
                Wall(3.75, 0, 0.75, 4, self.wall_radius),
                Wall(0, -6, 0.75, 2, self.wall_radius),

                Wall(0, 9, 8, 1, self.wall_radius),
                Wall(0, -9, 8, 1, self.wall_radius),
                Wall(9, 0, 1, 10, self.wall_radius),
                Wall(-9, 0, 1, 10, self.wall_radius),
            ]

            if test_mode_case_num == 1:
                kwargs['reset_low'] = np.array([-2.5, -6.75])
                kwargs['reset_high'] = np.array([-2.0, -6.25])

                kwargs['goal_low'] = np.array([2.0, -6.75])
                kwargs['goal_high'] = np.array([2.5, -6.25])
            elif test_mode_case_num == 2:
                kwargs['reset_low'] = np.array([-0.25, -1.25])
                kwargs['reset_high'] = np.array([0.25, -0.75])

                kwargs['goal_low'] = np.array([2.0, -6.75])
                kwargs['goal_high'] = np.array([2.5, -6.25])
            elif test_mode_case_num == 3:
                kwargs['reset_low'] = np.array([-6.25, -0.25])
                kwargs['reset_high'] = np.array([-5.75, 0.25])

                kwargs['goal_low'] = np.array([5.75, -0.25])
                kwargs['goal_high'] = np.array([6.25, 0.25])

            # Custom adds
            if test_mode_case_num == 10:
                kwargs['reset_low'] = np.array([-2.5, -6.75])
                kwargs['reset_high'] = np.array([-2.0, -6.25])
                kwargs['goal_low'] = np.array([-6.5, -0.25])
                kwargs['goal_high'] = np.array([-6.0, 0.25])
            if test_mode_case_num == 11:
                kwargs['reset_low'] = np.array([-2.5, -6.75])
                kwargs['reset_high'] = np.array([-2.0, -6.25])
                kwargs['goal_low'] = np.array([-6.5, 5.75])
                kwargs['goal_high'] = np.array([-6.0, 6.25])
            if test_mode_case_num == 12:
                kwargs['reset_low'] = np.array([-2.5, -6.75])
                kwargs['reset_high'] = np.array([-2.0, -6.25])
                kwargs['goal_low'] = np.array([-0.25, 5.75])
                kwargs['goal_high'] = np.array([0.25, 6.25])
            if test_mode_case_num == 13:
                kwargs['reset_low'] = np.array([-2.5, -6.75])
                kwargs['reset_high'] = np.array([-2.0, -6.25])
                kwargs['goal_low'] = np.array([-0.25, -0.25])
                kwargs['goal_high'] = np.array([0.25, 0.25])
            if test_mode_case_num == 14:
                kwargs['reset_low'] = np.array([-2.5, -6.75])
                kwargs['reset_high'] = np.array([-2.0, -6.25])
                kwargs['goal_low'] = np.array([6.0, 5.75])
                kwargs['goal_high'] = np.array([6.5, 6.25])
            if test_mode_case_num == 15:
                kwargs['reset_low'] = np.array([-2.5, -6.75])
                kwargs['reset_high'] = np.array([-2.0, -6.25])
                kwargs['goal_low'] = np.array([6.0, -0.25])
                kwargs['goal_high'] = np.array([6.5, 0.25])
            if test_mode_case_num == 16:
                kwargs['reset_low'] = np.array([-2.5, -6.75])
                kwargs['reset_high'] = np.array([-2.0, -6.25])
                kwargs['goal_low'] = np.array([2.0, -6.75])
                kwargs['goal_high'] = np.array([2.5, -6.25])


            if 'goal_low' not in kwargs:
                kwargs['goal_low'] = np.array([-8.0, -8.0])
            if 'goal_high' not in kwargs:
                kwargs['goal_high'] = np.array([8.0, 8.0])
        else:
            raise NotImplementedError


        self.quick_init(locals())
        AntEnv.__init__(
            self,
            # model_path=model_path,
            *args,
            **kwargs
        )


    def _collision_idx(self, pos):
        bad_pos_idx = []
        for i in range(len(pos)):
            for wall in self.walls:
                if wall.contains_point(pos[i]):
                    bad_pos_idx.append(i)
                    break
                # # BK 
                # if wall.contains_point_ths(pos[i], ths):
                #     bad_pos_idx.append(i)
                #     break

            # if 'small' in self.model_path:
            #     if 'maze2' in self.model_path:
            #         if (-2.00 <= pos[i][0] <= 2.00) and (-2.00 <= pos[i][1]):
            #             bad_pos_idx.append(i)
            #         elif (2.75 <= pos[i][0]) or (pos[i][0] <= -2.75):
            #             bad_pos_idx.append(i)
            #         elif (2.75 <= pos[i][1]) or (pos[i][1] <= -2.75):
            #             bad_pos_idx.append(i)
            #     else:
            #         if (-2.00 <= pos[i][0] <= 2.00) and (-2.00 <= pos[i][1] <= 2.00):
            #             bad_pos_idx.append(i)
            #         elif (2.75 <= pos[i][0]) or (pos[i][0] <= -2.75):
            #             bad_pos_idx.append(i)
            #         elif (2.75 <= pos[i][1]) or (pos[i][1] <= -2.75):
            #             bad_pos_idx.append(i)
            # elif 'big' in self.model_path:
            #     if 'maze2' in self.model_path:
            #         if (-2.25 <= pos[i][0] <= 2.25) and (-2.75 <= pos[i][1]):
            #             bad_pos_idx.append(i)
            #         elif (4.75 <= pos[i][0]) or (pos[i][0] <= -4.75):
            #             bad_pos_idx.append(i)
            #         elif (4.75 <= pos[i][1]) or (pos[i][1] <= -4.75):
            #             bad_pos_idx.append(i)
            #     else:
            #         raise NotImplementedError
            # else:
            #     raise NotImplementedError

        return bad_pos_idx

    def _sample_uniform_xy(self, batch_size, mode='goal'):
        assert mode in ['reset', 'goal']

        if mode == 'reset':
            low, high = self.reset_low, self.reset_high
        elif mode == 'goal':
            low, high = self.goal_low, self.goal_high

        goals = np.random.uniform(
            low,
            high,
            size=(batch_size, 2),
        )
        bad_goals_idx = self._collision_idx(goals)
        goals = np.delete(goals, bad_goals_idx, axis=0)
        while len(bad_goals_idx) > 0:
            new_goals = np.random.uniform(
                low,
                high,
                size=(len(bad_goals_idx), 2),
            )
            bad_goals_idx = self._collision_idx(new_goals)
            new_goals = np.delete(new_goals, bad_goals_idx, axis=0)
            goals = np.concatenate((goals, new_goals))

        # if 'small' in self.model_path:
        #     goals[(0 <= goals) * (goals < 0.5)] += 1
        #     goals[(0 <= goals) * (goals < 1.25)] += 1
        #     goals[(0 >= goals) * (goals > -0.5)] -= 1
        #     goals[(0 >= goals) * (goals > -1.25)] -= 1
        # else:
        #     goals[(0 <= goals) * (goals < 0.5)] += 2
        #     goals[(0 <= goals) * (goals < 1.5)] += 1.5
        #     goals[(0 >= goals) * (goals > -0.5)] -= 2
        #     goals[(0 >= goals) * (goals > -1.5)] -= 1.5
        return goals

    def plot_trajectory(self, 
                        ax=None, 
                        trajectory_all= None, 
                        save_dir = None, 
                        step = 0,
                        use_wandb=False,
                        draw_walls=True,
                        draw_state=True,
                        draw_goal=False,
                        draw_subgoals=False,
                        small_markers=True):

        if self.model_path in [
            'classic_mujoco/ant_maze2_gear30_small_dt3.xml',
            'classic_mujoco/ant_gear30_dt3_u_small.xml', # BK: u-small
        ]:
            extent = [-3.5, 3.5, -3.5, 3.5]
        elif self.model_path in [  #BK: fb-small
            'classic_mujoco/ant_fb_gear30_small_dt3.xml',
        ]:
            extent = [-5.25, 5.25, -5.25, 5.25]
        elif self.model_path in [  # BK: maze-small
            'classic_mujoco/ant_gear10_dt3_maze_small.xml'
        ]:
            extent = [-4, 4, -4, 4]
        elif self.model_path in [
            'classic_mujoco/ant_gear30_dt3_u_med.xml',
            'classic_mujoco/ant_gear15_dt3_u_med.xml',
            'classic_mujoco/ant_gear10_dt3_u_med.xml',
            'classic_mujoco/ant_gear30_dt2_u_med.xml',
            'classic_mujoco/ant_gear15_dt2_u_med.xml',
            'classic_mujoco/ant_gear10_dt2_u_med.xml',
            'classic_mujoco/ant_gear30_u_med.xml',
        ]:
            extent = [-4.5, 4.5, -4.5, 4.5]
        elif self.model_path in [
            'classic_mujoco/ant_fb_gear30_med_dt3.xml',
            'classic_mujoco/ant_gear30_dt3_fb_med.xml',
            'classic_mujoco/ant_gear15_dt3_fb_med.xml',
            'classic_mujoco/ant_gear10_dt3_fb_med.xml',
        ]:
            extent = [-6.0, 6.0, -6.0, 6.0]
        elif self.model_path in [
            'classic_mujoco/ant_gear10_dt3_u_long.xml',
            'classic_mujoco/ant_gear15_dt3_u_long.xml',
        ]:
            extent = [-3.75, 3.75, -9.0, 9.0]
        elif self.model_path in [
            'classic_mujoco/ant_gear10_dt3_maze_med.xml',
        ]:
            extent = [-8.0, 8.0, -8.0, 8.0]
        elif self.model_path in [
            'classic_mujoco/ant_gear10_dt3_fg_med.xml',
        ]:
            extent = [-8.0, 8.0, -8.0, 8.0]
        else:
            extent = [-5.5, 5.5, -5.5, 5.5]


        plt.clf()
        if ax is None:
            f, ax = plt.subplots(1, 1, figsize=(6, 4))
            ax.set_ylim(extent[2:4])
            ax.set_xlim(extent[0:2])
        plt.subplots_adjust(bottom=0.1, right=0.7, top=0.9)

        for wall in self.walls:
            # ax.vlines(x=wall.min_x, ymin=wall.min_y, ymax=wall.max_y)
            # ax.hlines(y=wall.min_y, xmin=wall.min_x, xmax=wall.max_x)
            # ax.vlines(x=wall.max_x, ymin=wall.min_y, ymax=wall.max_y)
            # ax.hlines(y=wall.max_y, xmin=wall.min_x, xmax=wall.max_x)
            ax.vlines(x=wall.endpoint1[0], ymin=wall.endpoint2[1], ymax=wall.endpoint1[1])
            ax.hlines(y=wall.endpoint2[1], xmin=wall.endpoint3[0], xmax=wall.endpoint2[0])
            ax.vlines(x=wall.endpoint3[0], ymin=wall.endpoint3[1], ymax=wall.endpoint4[1])
            ax.hlines(y=wall.endpoint4[1], xmin=wall.endpoint4[0], xmax=wall.endpoint1[0])

        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)

        # if small_markers:
        #     color = 'cyan'
        # else:
        #     color = 'blue'
        # if self._check_flipped(ob['state_observation'][3:9][None])[0]:
        #     color = 'purple'

        # ball = plt.Circle(ob['state_observation'][:2], 0.50 * marker_factor, color=color)
        # ax.add_artist(ball)

        for idx, trajectory in trajectory_all.items():
            color = plot_colors[idx % 50]
            for movement in trajectory:
                ax.plot(movement[0], movement[1], color = color, linewidth = 0.3)
      
        plt.savefig(save_dir)
        if use_wandb:
            wandb_img = wandb.Image(save_dir.__str__())
            wandb.log({'eval_trajectory': wandb_img})

        plt.clf()
        plt.cla()
        f.clear()
        plt.close(f)

    def state_coverage_1(self, trajectory_all, skill_dim):        
        state_cov = set()

        if self.maze_type == 'u-long':
            x_grid = np.linspace(-3.75, 3.75, 11)[:10]
            y_grid = np.linspace(-9, 9, 11)[:10]
            x_interval = 7.5 / 10
            y_interval = 18 / 10
        elif self.maze_type == 'u-med':
            x_grid = np.linspace(-4.5, 4.5, 11)[:10]
            y_grid = np.linspace(-4.5, 4.5, 11)[:10]
            x_interval = 9 / 10
            y_interval = 9 / 10
        elif self.maze_type == 'u-small':
            x_grid = np.linspace(-3.5, 3.5, 11)[:10]
            y_grid = np.linspace(-3.5, 3.5, 11)[:10]
            x_interval = 7 / 10
            y_interval = 7 / 10
        elif self.maze_type == 'fb-small':
            x_grid = np.linspace(-5.5, 5.5, 11)[:10]
            y_grid = np.linspace(-5.5, 5.5, 11)[:10]
            x_interval = 11 / 10
            y_interval = 11 / 10
        elif self.maze_type == 'maze-small':
            x_grid = np.linspace(-4, 4, 11)[:10]
            y_grid = np.linspace(-4, 4, 11)[:10]
            x_interval = 11 / 10
            y_interval = 11 / 10

        for n in range(skill_dim):
            for i in x_grid:  
                for j in y_grid: 
                    ob = np.array(trajectory_all[n])[:,:,0]
                    if ob[((ob[:,0]<i+x_interval) & (ob[:,0]>=i) & (ob[:,1]<j+y_interval) & (ob[:,1]>=j))].sum() >= 1:
                        state_cov.add((i,j))

        return len(state_cov)

    def state_coverage_2(self, trajectory_all, skill_dim):        
        state_cov = set()

        if self.maze_type == 'u-long':
            x_grid = np.linspace(-3.75, 3.75, 11)[:10]
            y_grid = np.linspace(-9, 9, 11)[:10]
            x_interval = 7.5 / 10
            y_interval = 18 / 10
        elif self.maze_type == 'u-med':
            x_grid = np.linspace(-4.5, 4.5, 11)[:10]
            y_grid = np.linspace(-4.5, 4.5, 11)[:10]
            x_interval = 9 / 10
            y_interval = 9 / 10
        elif self.maze_type == 'u-small':
            x_grid = np.linspace(-3.5, 3.5, 11)[:10]
            y_grid = np.linspace(-3.5, 3.5, 11)[:10]
            x_interval = 7 / 10
            y_interval = 7 / 10
        elif self.maze_type == 'fb-small':
            x_grid = np.linspace(-5.5, 5.5, 11)[:10]
            y_grid = np.linspace(-5.5, 5.5, 11)[:10]
            x_interval = 11 / 10
            y_interval = 11 / 10
        elif self.maze_type == 'maze-small':
            x_grid = np.linspace(-4, 4, 11)[:10]
            y_grid = np.linspace(-4, 4, 11)[:10]
            x_interval = 11 / 10
            y_interval = 11 / 10

        all_dots = []
        for idx, trajectory in trajectory_all.items():
            for movement in trajectory:
                mv = np.array(movement)
                x = mv[0]
                y = mv[1]
                f = interp1d(x,y)
                x_new = np.linspace(x[0],x[1],30)
                dots = np.concatenate((x_new.reshape(-1,1), f(x_new).reshape(-1,1)), axis=1)
                all_dots.append(dots)

        for ob in all_dots:
            for i in x_grid:  
                for j in y_grid: 
                    if ob[((ob[:,0]<i+x_interval) & (ob[:,0]>=i) & (ob[:,1]<j+y_interval) & (ob[:,1]>=j))].sum() >= 1:
                        state_cov.add((i,j))

        return len(state_cov)



class Wall:
    def __init__(self, x_center, y_center, x_thickness, y_thickness, min_dist):
        self.min_x = x_center - x_thickness - min_dist
        self.max_x = x_center + x_thickness + min_dist
        self.min_y = y_center - y_thickness - min_dist
        self.max_y = y_center + y_thickness + min_dist

        self.endpoint1 = (x_center+x_thickness, y_center+y_thickness)
        self.endpoint2 = (x_center+x_thickness, y_center-y_thickness)
        self.endpoint3 = (x_center-x_thickness, y_center-y_thickness)
        self.endpoint4 = (x_center-x_thickness, y_center+y_thickness)

    def contains_point(self, point):
        return (self.min_x < point[0] < self.max_x) and (self.min_y < point[1] < self.max_y)

    def contains_points(self, points):
        return (self.min_x < points[:,0]) * (points[:,0] < self.max_x) \
               * (self.min_y < points[:,1]) * (points[:,1] < self.max_y)



if __name__ == '__main__':
    env = AntMazeEnv(
        goal_low=[-4, -4],
        goal_high=[4, 4],
        goal_is_xy=True,
        reward_type='xy_dense',
    )
    import gym
    from multiworld.envs.mujoco import register_custom_envs
    register_custom_envs()
    env = gym.make('AntMaze150RandomInitEnv-v0')
    # env = gym.make('AntCrossMaze150Env-v0')
    # env = gym.make('DebugAntMaze30BottomLeftRandomInitGoalsPreset1Env-v0')
    env = gym.make(
        # 'AntMaze30RandomInitFS20Env-v0',
        # 'AntMaze30RandomInitEnv-v0',
        # 'AntMazeSmall30RandomInitFS10Env-v0',
        # 'AntMazeSmall30RandomInitFs5Dt3Env-v0',
        # 'AntMaze30RandomInitNoVelEnv-v0',
        # 'AntMaze30StateEnv-v0',
        # 'AntMaze30QposRandomInitFS20Env-v0',
        # 'AntMazeSmall30RandomInitFs10Dt3Env-v0',
        # 'AntMazeQposRewSmall30RandomInitFs5Dt3Env-v0',
        # 'AntMazeXyRewSmall30RandomInitFs5Dt3Env-v0',
        # 'AntMazeQposRewSmall30Fs5Dt3Env-v0',
        # 'AntMazeQposRewSmall30Fs5Dt3NoTermEnv-v0',
        # 'AntMazeXyRewSmall30RandomInitFs5Dt3NoTermEnv-v0',
        # 'AntMazeXyRewSmall30Fs5Dt3NoTermEnv-v0',
        'AntMazeQposRewSmall30Fs5Dt3NoTermEnv-v0',
    )
    env.reset()
    i = 0
    while True:
        i += 1
        env.render()
        action = env.action_space.sample()
        # action = np.zeros_like(action)
        obs, reward, done, info = env.step(action)
        # print(reward, np.linalg.norm(env.sim.data.get_body_xpos('torso')[:2]
        #                              - env._xy_goal) )
        # print(env.sim.data.qpos)
        print(info)
        if i % 5 == 0:
            env.reset()
