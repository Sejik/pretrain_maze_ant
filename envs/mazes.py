# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import wandb
import numpy as np
import matplotlib.pyplot as plt
import ipdb
from sklearn.neighbors import KernelDensity

plot_colors = ['blue', 'red', 'green', 'brown', 'orange', 'pink', 'rebeccapurple', 'olive', 'purple', 'cyan', 
               'black', 'maroon', 'yellow', 'navy', 'darkgreen', 'lavender', 'peru', 'fuchsia', 'slateblue', 'oldlace',
               'darkviolet', 'silver', 'lightcoral', 'mistyrose', 'salmon', 'tomato', 'chocolate', 'bisque', 'burlywood', 'tan',
               'darkgoldenrod', 'darkkhaki', 'olivedrab', 'darkseagreen', 'lightgreen', 'limegreen', 'lime', 'aquamarine', 'turquoise', 'teal',
               'deepskyblue', 'steelblue', 'dodgerblue', 'cornflowerblue', 'royalblue', 'mediumpurple', 'blueviolet', 'indigo', 'plum', 'palevioletred']

delta_0_color = 'grey'

novel_color = 'red'


def plot_training_samples(ax, samples, label=1, size=5):
    ax.scatter(samples[:,0], samples[:,1], s=size, facecolor=plot_colors[label%50], label=label)
    ax.legend(ncol=3, bbox_to_anchor=(1.05,1))

class CircleMaze:
    def __init__(self):
        self.ring_r = 0.15
        self.stop_t = 0.05
        self.s_angle = 30

        self.mean_s0 = (
            float(np.cos(np.pi * self.s_angle / 180)),
            float(np.sin(np.pi * self.s_angle / 180))
        )
        self.mean_g = (
            float(np.cos(np.pi * (360-self.s_angle) / 180)),
            float(np.sin(np.pi * (360-self.s_angle) / 180))
        )

    def plot(self, ax=None):
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(5, 4))
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(5, 4))
        rads = np.linspace(self.stop_t * 2 * np.pi, (1 - self.stop_t) * 2 * np.pi)
        xs_i = (1 - self.ring_r) * np.cos(rads)
        ys_i = (1 - self.ring_r) * np.sin(rads)
        xs_o = (1 + self.ring_r) * np.cos(rads)
        ys_o = (1 + self.ring_r) * np.sin(rads)
        ax.plot(xs_i, ys_i, 'k', linewidth=3)
        ax.plot(xs_o, ys_o, 'k', linewidth=3)
        ax.plot([xs_i[0], xs_o[0]], [ys_i[0], ys_o[0]], 'k', linewidth=3)
        ax.plot([xs_i[-1], xs_o[-1]], [ys_i[-1], ys_o[-1]], 'k', linewidth=3)
        lim = 1.1 + self.ring_r
        ax.set_xlim([-lim, lim])
        ax.set_ylim([-lim, lim])

    def sample_start(self):
        STD = 0.1
        return self.move(self.mean_s0, (STD * np.random.randn(), STD * np.random.randn()))

    def sample_goal(self):
        STD = 0.1
        return self.move(self.mean_g, (STD * np.random.randn(), STD * np.random.randn()))

    @staticmethod
    def xy_to_rt(xy):
        x = xy[0]
        y = xy[1]
        r = np.sqrt(x ** 2 + y ** 2)
        t = np.arctan2(y, x) % (2 * np.pi)
        return r, t

    def move(self, coords, action):
        xp, yp = coords
        rp, tp = self.xy_to_rt(coords)

        xy = (coords[0] + action[0], coords[1] + action[1])

        r, t = self.xy_to_rt(xy)
        t = np.clip(t % (2 * np.pi), (0.001 + self.stop_t) * (2 * np.pi), (1 - (0.001 + self.stop_t)) * (2 * np.pi))
        x = np.cos(t) * r
        y = np.sin(t) * r

        if coords is not None:

            if xp > 0:
                if (y < 0) and (yp > 0):
                    t = self.stop_t * 2 * np.pi
                elif (y > 0) and (yp < 0):
                    t = (1 - self.stop_t) * 2 * np.pi
            x = np.cos(t) * r
            y = np.sin(t) * r

        n = 8
        xyi = np.array([xp, yp]).astype(np.float32)
        dxy = (np.array([x, y]).astype(np.float32) - xyi) / n
        new_r = float(rp)
        new_t = float(tp)

        count = 0

        def r_ok(r_):
            return (1 - self.ring_r) <= r_ <= (1 + self.ring_r)

        def t_ok(t_):
            return (self.stop_t * (2 * np.pi)) <= (t_ % (2 * np.pi)) <= ((1 - self.stop_t) * (2 * np.pi))

        while r_ok(new_r) and t_ok(new_t) and count < n:
            xyi += dxy
            new_r, new_t = self.xy_to_rt(xyi)
            count += 1

        r = np.clip(new_r, 1 - self.ring_r + 0.01, 1 + self.ring_r - 0.01)
        t = np.clip(new_t % (2 * np.pi), (0.001 + self.stop_t) * (2 * np.pi), (1 - (0.001 + self.stop_t)) * (2 * np.pi))
        x = np.cos(t) * r
        y = np.sin(t) * r

        return float(x), float(y)


class Maze:
    def __init__(self, *segment_dicts, goal_squares=None, start_squares=None,
                 min_wall_coord=None, walls_to_add=(), walls_to_remove=()):
        self._segments = {'origin': {'loc': (0.0, 0.0), 'connect': set()}}
        self._locs = set()
        self._locs.add(self._segments['origin']['loc'])
        self._walls = set()
        for direction in ['up', 'down', 'left', 'right']:
            self._walls.add(self._wall_line(self._segments['origin']['loc'], direction))
        self._last_segment = 'origin'
        self.goal_squares = None

        # These allow to implement more complex mazes
        self.min_wall_coord = min_wall_coord
        self.walls_to_add = walls_to_add
        self.walls_to_remove = walls_to_remove

        # To initialize randomly
        self.prev_shift = None

        if goal_squares is None:
            self._goal_squares = None
        elif isinstance(goal_squares, str):
            self._goal_squares = [goal_squares.lower()]
        elif isinstance(goal_squares, (tuple, list)):
            self._goal_squares = [gs.lower() for gs in goal_squares]
        else:
            raise TypeError

        if start_squares is None:
            self.start_squares = ['origin']
        elif isinstance(start_squares, str):
            self.start_squares = [start_squares.lower()]
        elif isinstance(start_squares, (tuple, list)):
            self.start_squares = [ss.lower() for ss in start_squares]
        else:
            raise TypeError

        for segment_dict in segment_dicts:
            self._add_segment(**segment_dict)
        self._finalize()

    @staticmethod
    def _wall_line(coord, direction):
        x, y = coord
        if direction == 'up':
            w = [(x - 0.5, x + 0.5), (y + 0.5, y + 0.5)]
        elif direction == 'right':
            w = [(x + 0.5, x + 0.5), (y + 0.5, y - 0.5)]
        elif direction == 'down':
            w = [(x - 0.5, x + 0.5), (y - 0.5, y - 0.5)]
        elif direction == 'left':
            w = [(x - 0.5, x - 0.5), (y - 0.5, y + 0.5)]
        else:
            raise ValueError
        w = tuple([tuple(sorted(line)) for line in w])
        return w

    def _add_segment(self, name, anchor, direction, connect=None, times=1):
        name = str(name).lower()
        original_name = str(name).lower()
        if times > 1:
            assert connect is None
            last_name = str(anchor).lower()
            for time in range(times):
                this_name = original_name + str(time)
                self._add_segment(name=this_name.lower(), anchor=last_name, direction=direction)
                last_name = str(this_name)
            return

        anchor = str(anchor).lower()
        assert anchor in self._segments

        direction = str(direction).lower()

        final_connect = set()

        if connect is not None:
            if isinstance(connect, str):
                connect = str(connect).lower()
                assert connect in ['up', 'down', 'left', 'right']
                final_connect.add(connect)
            elif isinstance(connect, (tuple, list)):
                for connect_direction in connect:
                    connect_direction = str(connect_direction).lower()
                    assert connect_direction in ['up', 'down', 'left', 'right']
                    final_connect.add(connect_direction)

        sx, sy = self._segments[anchor]['loc']
        dx, dy = 0.0, 0.0
        if direction == 'left':
            dx -= 1
            final_connect.add('right')
        elif direction == 'right':
            dx += 1
            final_connect.add('left')
        elif direction == 'up':
            dy += 1
            final_connect.add('down')
        elif direction == 'down':
            dy -= 1
            final_connect.add('up')
        else:
            raise ValueError

        new_loc = (sx + dx, sy + dy)
        assert new_loc not in self._locs

        self._segments[name] = {'loc': new_loc, 'connect': final_connect}
        for direction in ['up', 'down', 'left', 'right']:
            self._walls.add(self._wall_line(new_loc, direction))
        self._locs.add(new_loc)

        self._last_segment = name

    def _finalize(self):
        bottom_wall_coord = min([min(w[0]) for w in self._walls]) + 0.5
        left_wall_coord = min([min(w[1]) for w in self._walls]) + 0.5

        def _rm_wall(wall):
            coords = wall[0] + wall[1]
            # Check if this is the bottom wall
            if wall[0][0] < bottom_wall_coord and wall[0][1] < bottom_wall_coord:
                return False
            # Check if this is the left wall
            if wall[1][0] < left_wall_coord and wall[1][1] < left_wall_coord:
                return False
            # Remove walls in the bottom-left corner
            return all(c < self.min_wall_coord for c in coords)

        if self.min_wall_coord is not None:
            self._walls = set([w for w in self._walls if not _rm_wall(w)])

        for wall in self.walls_to_remove:
            if wall in self._walls:
                self._walls.remove(wall)

        for segment in self._segments.values():
            for c_dir in list(segment['connect']):
                wall = self._wall_line(segment['loc'], c_dir)
                if wall in self._walls:
                    self._walls.remove(wall)

        for wall in self.walls_to_add:
            self._walls.add(wall)

        if self._goal_squares is None:
            self.goal_squares = [self._last_segment]
        else:
            self.goal_squares = []
            for gs in self._goal_squares:
                assert gs in self._segments
                self.goal_squares.append(gs)

    def plot(self, ax=None):
        if ax is None:
            f, ax = plt.subplots(1, 1, figsize=(5, 4))
        for x, y in self._walls:
            ax.plot(x, y, 'k-')
        plt.clf()
        plt.cla()
        f.clear()
        plt.close(f)
    

    def plot_maze(self, ax, xx, yy, zz, all_obs):
        x = all_obs[:,0]
        y = all_obs[:,1]
        ax.pcolormesh(xx, yy, zz)
        ax.set_xlim(-1,11)
        ax.set_ylim(-1,11)
        ax.scatter(x, y, s=0.5, facecolor='white')  # training data 찍기
    
    def plot_trajectory(self, 
                        ax=None, 
                        trajectory_all= None, 
                        save_dir = None, 
                        step = 0, 
                        use_wandb=False,
                        env=None,
                        goal=None):
        plt.clf()
        if ax is None:
            f, ax = plt.subplots(1, 1, figsize=(6, 4))
        plt.subplots_adjust(bottom=0.1, right=0.7, top=0.9)
        for x, y in self._walls:
            ax.plot(x, y, 'k-')

        for idx, trajectory in trajectory_all.items():
            color = plot_colors[idx % 50]
            for movement in trajectory:
                ax.plot(movement[0], movement[1], color = color, linewidth = 0.3)
            ax.scatter(goal[idx][:,0],goal[idx][:,1], c = color, s = 100)

        ax.legend(loc = 'center left', fontsize = 'xx-small', ncol=2, bbox_to_anchor=(1, 0.5))
        plt.savefig(save_dir)
        if use_wandb:
            wandb_img = wandb.Image(save_dir.__str__())
            wandb.log({'eval_trajectory': wandb_img})
            # wandb.log({'eval_trajectory_%d'%(step): wandb_img})
        plt.clf()
        plt.cla()
        f.clear()
        plt.close(f)
        
        plt.clf()
        f2, ax2 = plt.subplots(1, 1, figsize=(6, 4))
        plt.subplots_adjust(bottom=0.1, right=0.7, top=0.9)
        for x, y in self._walls:
            ax2.plot(x, y, 'k-')

        for idx, trajectory in trajectory_all.items():
            color = plot_colors[idx % 50]
            for movement in trajectory:
                ax2.plot(movement[0], movement[1], color = color, linewidth = 0.3)

        ax2.legend(loc = 'center left', fontsize = 'xx-small', ncol=2, bbox_to_anchor=(1, 0.5))
        save_dir2 = save_dir.__str__().replace('.png', '_without_goal.png')
        plt.savefig(save_dir2)
        if use_wandb:
            wandb_img2 = wandb.Image(save_dir2.__str__())
            wandb.log({'eval_trajectory_without_goal': wandb_img2})
            # wandb.log({'eval_trajectory_%d'%(step): wandb_img})
        plt.clf()
        plt.cla()
        f.clear()
        plt.close(f2)

    def state_coverage(self, trajectory_all, skill_dim):        
        state_coveraged = 0
        state_cov = set()
        for n in range(skill_dim):
            for i in np.arange(-0.5,9.5):
                for j in np.arange(-0.5, 9.5):
                    ob = np.array(trajectory_all[n])[:,:,0]
                    if ob[((ob[:,0]<i+1) & (ob[:,0]>=i) & (ob[:,1]<j+1) & (ob[:,1]>=j))].sum() >= 1:
                        state_cov.add((i,j))
            state_coveraged += len(state_cov)
        state_coveraged_avg = state_coveraged / skill_dim
        return state_coveraged_avg

    def sample(self):
        segment_keys = list(self._segments.keys())
        square_id = segment_keys[np.random.randint(low=0, high=len(segment_keys))]
        square_loc = self._segments[square_id]['loc']
        shift = np.random.uniform(low=-0.5, high=0.5, size=(2,))
        loc = square_loc + shift
        return loc[0], loc[1]

    def sample_start(self, train_random=None):
        min_wall_dist = 0.05

        s_square = self.start_squares[np.random.randint(low=0, high=len(self.start_squares))]
        s_square_loc = self._segments[s_square]['loc']

        if train_random:
            max_loc = [-999.0, -999.0]
            min_loc = [999.0, 999.0]
            for k in self._segments:
                x = self._segments[k]['loc'][0]
                y = self._segments[k]['loc'][1]
                max_loc[0] = max(max_loc[0], x)
                max_loc[1] = max(max_loc[1], y) 
                min_loc[0] = min(min_loc[0], x)
                min_loc[1] = min(min_loc[1], y) 

        while True:
            shift = np.random.uniform(low=-0.5, high=0.5, size=(2,))
            if train_random:
                shift[0] = np.random.uniform(low=min_loc[0], high=max_loc[0], size=(1,))
                shift[1] = np.random.uniform(low=min_loc[1], high=max_loc[1], size=(1,))
            #shift = np.random.uniform(low=0, high=0, size=(2,))  # Static Starting Point
            loc = s_square_loc + shift
            dist_checker = np.array([min_wall_dist, min_wall_dist]) * np.sign(shift)
            stopped_loc = self.move(loc, dist_checker)
            if float(np.sum(np.abs((loc + dist_checker) - stopped_loc))) == 0.0:
                break
        return loc[0], loc[1]

    def sample_random_start(self, counter, num_skills):
        min_wall_dist = 0.05

        s_square = self.start_squares[np.random.randint(low=0, high=len(self.start_squares))]
        s_square_loc = self._segments[s_square]['loc']

        max_loc = [-999.0, -999.0]
        min_loc = [999.0, 999.0]
        for k in self._segments:
            x = self._segments[k]['loc'][0]
            y = self._segments[k]['loc'][1]
            max_loc[0] = max(max_loc[0], x)
            max_loc[1] = max(max_loc[1], y) 
            min_loc[0] = min(min_loc[0], x)
            min_loc[1] = min(min_loc[1], y) 

        while True:
            shift = np.random.uniform(low=-0.5, high=0.5, size=(2,))
            if counter % (5 * num_skills) == 0: 
                shift[0] = np.random.uniform(low=min_loc[0], high=max_loc[0], size=(1,))
                shift[1] = np.random.uniform(low=min_loc[1], high=max_loc[1], size=(1,))
                self.prev_shift = shift
            else:
                shift = self.prev_shift
            #shift = np.random.uniform(low=0, high=0, size=(2,))  # Static Starting Point
            loc = s_square_loc + shift
            dist_checker = np.array([min_wall_dist, min_wall_dist]) * np.sign(shift)
            stopped_loc = self.move(loc, dist_checker)
            if float(np.sum(np.abs((loc + dist_checker) - stopped_loc))) == 0.0:
                break
        return loc[0], loc[1]


    def sample_goal(self, min_wall_dist=None):
        if min_wall_dist is None:
            min_wall_dist = 0.1
        else:
            min_wall_dist = min(0.4, max(0.01, min_wall_dist))

        g_square = self.goal_squares[np.random.randint(low=0, high=len(self.goal_squares))]
        g_square_loc = self._segments[g_square]['loc']
        while True:
            shift = np.random.uniform(low=-0.5, high=0.5, size=(2,))
            loc = g_square_loc + shift
            dist_checker = np.array([min_wall_dist, min_wall_dist]) * np.sign(shift)
            stopped_loc = self.move(loc, dist_checker)
            if float(np.sum(np.abs((loc + dist_checker) - stopped_loc))) == 0.0:
                break
        return loc[0], loc[1]

    def move(self, coord_start, coord_delta, depth=None):
        if depth is None:
            depth = 0
        cx, cy = coord_start
        loc_x0 = np.round(cx)
        loc_y0 = np.round(cy)
        #assert (float(loc_x0), float(loc_y0)) in self._locs
        dx, dy = coord_delta
        loc_x1 = np.round(cx + dx)
        loc_y1 = np.round(cy + dy)
        d_loc_x = int(np.abs(loc_x1 - loc_x0))
        d_loc_y = int(np.abs(loc_y1 - loc_y0))
        xs_crossed = [loc_x0 + (np.sign(dx) * (i + 0.5)) for i in range(d_loc_x)]
        ys_crossed = [loc_y0 + (np.sign(dy) * (i + 0.5)) for i in range(d_loc_y)]

        rds = []

        for x in xs_crossed:
            r = (x - cx) / dx
            loc_x = np.round(cx + (0.999 * r * dx))
            loc_y = np.round(cy + (0.999 * r * dy))
            direction = 'right' if dx > 0 else 'left'
            crossed_line = self._wall_line((loc_x, loc_y), direction)
            if crossed_line in self._walls:
                rds.append([r, direction])

        for y in ys_crossed:
            r = (y - cy) / dy
            loc_x = np.round(cx + (0.999 * r * dx))
            loc_y = np.round(cy + (0.999 * r * dy))
            direction = 'up' if dy > 0 else 'down'
            crossed_line = self._wall_line((loc_x, loc_y), direction)
            if crossed_line in self._walls:
                rds.append([r, direction])

        # The wall will only stop the agent in the direction perpendicular to the wall
        if rds:
            rds = sorted(rds)
            r, direction = rds[0]
            if depth < 3:
                new_dx = r * dx
                new_dy = r * dy
                repulsion = float(np.abs(np.random.rand() * 0.01))
                if direction in ['right', 'left']:
                    new_dx -= np.sign(dx) * repulsion
                    partial_coords = cx + new_dx, cy + new_dy
                    remaining_delta = (0.0, (1 - r) * dy)
                else:
                    new_dy -= np.sign(dy) * repulsion
                    partial_coords = cx + new_dx, cy + new_dy
                    remaining_delta = ((1 - r) * dx, 0.0)
                return self.move(partial_coords, remaining_delta, depth+1)
        else:
            r = 1.0

        dx *= r
        dy *= r
        return cx + dx, cy + dy


def make_crazy_maze(size, seed=None):
    np.random.seed(seed)

    deltas = [
        [(-1, 0), 'right'],
        [(1, 0), 'left'],
        [(0, -1), 'up'],
        [(0, 1), 'down'],
    ]

    empty_locs = []
    for x in range(size):
        for y in range(size):
            empty_locs.append((x, y))

    locs = [empty_locs.pop(0)]
    dirs = [None]
    anchors = [None]

    while len(empty_locs) > 0:
        still_empty = []
        np.random.shuffle(empty_locs)
        for empty_x, empty_y in empty_locs:
            found_anchor = False
            np.random.shuffle(deltas)
            for (dx, dy), direction in deltas:
                c = (empty_x + dx, empty_y + dy)
                if c in locs:
                    found_anchor = True
                    locs.append((empty_x, empty_y))
                    dirs.append(direction)
                    anchors.append(c)
                    break
            if not found_anchor:
                still_empty.append((empty_x, empty_y))
        empty_locs = still_empty[:]

    locs = [str(x) + ',' + str(y) for x, y in locs[1:]]
    dirs = dirs[1:]
    anchors = [str(x) + ',' + str(y) for x, y in anchors[1:]]
    anchors = ['origin' if a == '0,0' else a for a in anchors]

    segments = []
    for loc, d, anchor in zip(locs, dirs, anchors):
        segments.append(dict(name=loc, anchor=anchor, direction=d))

    np.random.seed()
    return Maze(*segments, goal_squares='{s},{s}'.format(s=size - 1))


def make_experiment_maze(h, half_w, sz0):
    if h < 2:
        h = 2
    if half_w < 3:
        half_w = 3
    w = 1 + (2*half_w)
    # Create the starting row
    segments = [{'anchor': 'origin', 'direction': 'right', 'name': '0,1'}]
    for w_ in range(1, w-1):
        segments.append({'anchor': '0,{}'.format(w_), 'direction': 'right', 'name': '0,{}'.format(w_+1)})

    # Add each row to create H
    for h_ in range(1, h):
        segments.append({'anchor': '{},{}'.format(h_-1, w-1), 'direction': 'up', 'name': '{},{}'.format(h_, w-1)})

        c = None if h_ == sz0 else 'down'
        for w_ in range(w-2, -1, -1):
            segments.append(
                {'anchor': '{},{}'.format(h_, w_+1), 'direction': 'left', 'connect': c, 'name': '{},{}'.format(h_, w_)}
            )

    return Maze(*segments, goal_squares=['{},{}'.format(h-1, half_w+d) for d in [0]])


def make_hallway_maze(corridor_length):
    corridor_length = int(corridor_length)
    assert corridor_length >= 1

    segments = []
    last = 'origin'
    for x in range(1, corridor_length+1):
        next_name = '0,{}'.format(x)
        segments.append({'anchor': last, 'direction': 'right', 'name': next_name})
        last = str(next_name)

    return Maze(*segments, goal_squares=last)


def make_u_maze(corridor_length):
    corridor_length = int(corridor_length)
    assert corridor_length >= 1

    segments = []
    last = 'origin'
    for x in range(1, corridor_length + 1):
        next_name = '0,{}'.format(x)
        segments.append({'anchor': last, 'direction': 'right', 'name': next_name})
        last = str(next_name)

    assert last == '0,{}'.format(corridor_length)

    up_size = 2

    for x in range(1, up_size+1):
        next_name = '{},{}'.format(x, corridor_length)
        segments.append({'anchor': last, 'direction': 'up', 'name': next_name})
        last = str(next_name)

    assert last == '{},{}'.format(up_size, corridor_length)

    for x in range(1, corridor_length + 1):
        next_name = '{},{}'.format(up_size, corridor_length - x)
        segments.append({'anchor': last, 'direction': 'left', 'name': next_name})
        last = str(next_name)

    assert last == '{},0'.format(up_size)

    return Maze(*segments, goal_squares=last)



mazes_dict = dict()

mazes_dict['circle'] = {'maze': CircleMaze(), 'action_range': 0.25}

segments_a = [
    dict(name='A', anchor='origin', direction='down', times=4),
    dict(name='B', anchor='A3', direction='right', times=4),
    dict(name='C', anchor='B3', direction='up', times=4),
    dict(name='D', anchor='A1', direction='right', times=2),
    dict(name='E', anchor='D1', direction='up', times=2),
]
mazes_dict['square_a'] = {'maze': Maze(*segments_a, goal_squares=['c2', 'c3']), 'action_range': 0.95}

segments_b = [
    dict(name='A', anchor='origin', direction='down', times=4),
    dict(name='B', anchor='A3', direction='right', times=4),
    dict(name='C', anchor='B3', direction='up', times=4),
    dict(name='D', anchor='B1', direction='up', times=4),
]
mazes_dict['square_b'] = {'maze': Maze(*segments_b, goal_squares=['c2', 'c3']), 'action_range': 0.95}

segments_c = [
    dict(name='A', anchor='origin', direction='down', times=4),
    dict(name='B', anchor='A3', direction='right', times=2),
    dict(name='C', anchor='B1', direction='up', times=4),
    dict(name='D', anchor='C3', direction='right', times=2),
    dict(name='E', anchor='D1', direction='down', times=4)
]
mazes_dict['square_c'] = {'maze': Maze(*segments_c, goal_squares=['e2', 'e3']), 'action_range': 0.95}

segments_d = [
    dict(name='TL', anchor='origin', direction='left', times=3),
    dict(name='TLD', anchor='TL2', direction='down', times=3),
    dict(name='TLR', anchor='TLD2', direction='right', times=2),
    dict(name='TLU', anchor='TLR1', direction='up'),
    dict(name='TR', anchor='origin', direction='right', times=3),
    dict(name='TRD', anchor='TR2', direction='down', times=3),
    dict(name='TRL', anchor='TRD2', direction='left', times=2),
    dict(name='TRU', anchor='TRL1', direction='up'),
    dict(name='TD', anchor='origin', direction='down', times=3),
]
mazes_dict['square_d'] = {'maze': Maze(*segments_d, goal_squares=['tlu', 'tlr1', 'tru', 'trl1']), 'action_range': 0.95}

segments_crazy = [
    {'anchor': 'origin', 'direction': 'right', 'name': '1,0'},
     {'anchor': 'origin', 'direction': 'up', 'name': '0,1'},
     {'anchor': '1,0', 'direction': 'right', 'name': '2,0'},
     {'anchor': '0,1', 'direction': 'up', 'name': '0,2'},
     {'anchor': '0,2', 'direction': 'right', 'name': '1,2'},
     {'anchor': '2,0', 'direction': 'up', 'name': '2,1'},
     {'anchor': '1,2', 'direction': 'right', 'name': '2,2'},
     {'anchor': '0,2', 'direction': 'up', 'name': '0,3'},
     {'anchor': '2,1', 'direction': 'right', 'name': '3,1'},
     {'anchor': '1,2', 'direction': 'down', 'name': '1,1'},
     {'anchor': '3,1', 'direction': 'down', 'name': '3,0'},
     {'anchor': '1,2', 'direction': 'up', 'name': '1,3'},
     {'anchor': '3,1', 'direction': 'right', 'name': '4,1'},
     {'anchor': '1,3', 'direction': 'up', 'name': '1,4'},
     {'anchor': '4,1', 'direction': 'right', 'name': '5,1'},
     {'anchor': '4,1', 'direction': 'up', 'name': '4,2'},
     {'anchor': '5,1', 'direction': 'down', 'name': '5,0'},
     {'anchor': '3,0', 'direction': 'right', 'name': '4,0'},
     {'anchor': '1,4', 'direction': 'right', 'name': '2,4'},
     {'anchor': '4,2', 'direction': 'right', 'name': '5,2'},
     {'anchor': '2,4', 'direction': 'right', 'name': '3,4'},
     {'anchor': '3,4', 'direction': 'up', 'name': '3,5'},
     {'anchor': '1,4', 'direction': 'left', 'name': '0,4'},
     {'anchor': '1,4', 'direction': 'up', 'name': '1,5'},
     {'anchor': '2,2', 'direction': 'up', 'name': '2,3'},
     {'anchor': '3,1', 'direction': 'up', 'name': '3,2'},
     {'anchor': '5,0', 'direction': 'right', 'name': '6,0'},
     {'anchor': '3,2', 'direction': 'up', 'name': '3,3'},
     {'anchor': '4,2', 'direction': 'up', 'name': '4,3'},
     {'anchor': '6,0', 'direction': 'up', 'name': '6,1'},
     {'anchor': '6,0', 'direction': 'right', 'name': '7,0'},
     {'anchor': '6,1', 'direction': 'right', 'name': '7,1'},
     {'anchor': '3,4', 'direction': 'right', 'name': '4,4'},
     {'anchor': '1,5', 'direction': 'right', 'name': '2,5'},
     {'anchor': '7,1', 'direction': 'up', 'name': '7,2'},
     {'anchor': '1,5', 'direction': 'up', 'name': '1,6'},
     {'anchor': '4,4', 'direction': 'right', 'name': '5,4'},
     {'anchor': '5,4', 'direction': 'down', 'name': '5,3'},
     {'anchor': '0,4', 'direction': 'up', 'name': '0,5'},
     {'anchor': '7,2', 'direction': 'left', 'name': '6,2'},
     {'anchor': '1,6', 'direction': 'left', 'name': '0,6'},
     {'anchor': '7,0', 'direction': 'right', 'name': '8,0'},
     {'anchor': '7,2', 'direction': 'right', 'name': '8,2'},
     {'anchor': '2,5', 'direction': 'up', 'name': '2,6'},
     {'anchor': '8,0', 'direction': 'up', 'name': '8,1'},
     {'anchor': '3,5', 'direction': 'up', 'name': '3,6'},
     {'anchor': '6,2', 'direction': 'up', 'name': '6,3'},
     {'anchor': '6,3', 'direction': 'right', 'name': '7,3'},
     {'anchor': '3,5', 'direction': 'right', 'name': '4,5'},
     {'anchor': '7,3', 'direction': 'up', 'name': '7,4'},
     {'anchor': '6,3', 'direction': 'up', 'name': '6,4'},
     {'anchor': '6,4', 'direction': 'up', 'name': '6,5'},
     {'anchor': '8,1', 'direction': 'right', 'name': '9,1'},
     {'anchor': '8,2', 'direction': 'right', 'name': '9,2'},
     {'anchor': '2,6', 'direction': 'up', 'name': '2,7'},
     {'anchor': '8,2', 'direction': 'up', 'name': '8,3'},
     {'anchor': '6,5', 'direction': 'left', 'name': '5,5'},
     {'anchor': '5,5', 'direction': 'up', 'name': '5,6'},
     {'anchor': '7,4', 'direction': 'right', 'name': '8,4'},
     {'anchor': '8,4', 'direction': 'right', 'name': '9,4'},
     {'anchor': '0,6', 'direction': 'up', 'name': '0,7'},
     {'anchor': '2,7', 'direction': 'up', 'name': '2,8'},
     {'anchor': '7,4', 'direction': 'up', 'name': '7,5'},
     {'anchor': '9,4', 'direction': 'down', 'name': '9,3'},
     {'anchor': '9,4', 'direction': 'up', 'name': '9,5'},
     {'anchor': '2,7', 'direction': 'left', 'name': '1,7'},
     {'anchor': '4,5', 'direction': 'up', 'name': '4,6'},
     {'anchor': '9,1', 'direction': 'down', 'name': '9,0'},
     {'anchor': '6,5', 'direction': 'up', 'name': '6,6'},
     {'anchor': '3,6', 'direction': 'up', 'name': '3,7'},
     {'anchor': '1,7', 'direction': 'up', 'name': '1,8'},
     {'anchor': '3,7', 'direction': 'right', 'name': '4,7'},
     {'anchor': '2,8', 'direction': 'up', 'name': '2,9'},
     {'anchor': '2,9', 'direction': 'left', 'name': '1,9'},
     {'anchor': '7,5', 'direction': 'up', 'name': '7,6'},
     {'anchor': '1,8', 'direction': 'left', 'name': '0,8'},
     {'anchor': '6,6', 'direction': 'up', 'name': '6,7'},
     {'anchor': '0,8', 'direction': 'up', 'name': '0,9'},
     {'anchor': '7,5', 'direction': 'right', 'name': '8,5'},
     {'anchor': '6,7', 'direction': 'left', 'name': '5,7'},
     {'anchor': '2,9', 'direction': 'right', 'name': '3,9'},
     {'anchor': '3,9', 'direction': 'right', 'name': '4,9'},
     {'anchor': '7,6', 'direction': 'right', 'name': '8,6'},
     {'anchor': '3,7', 'direction': 'up', 'name': '3,8'},
     {'anchor': '9,5', 'direction': 'up', 'name': '9,6'},
     {'anchor': '7,6', 'direction': 'up', 'name': '7,7'},
     {'anchor': '5,7', 'direction': 'up', 'name': '5,8'},
     {'anchor': '3,8', 'direction': 'right', 'name': '4,8'},
     {'anchor': '8,6', 'direction': 'up', 'name': '8,7'},
     {'anchor': '5,8', 'direction': 'right', 'name': '6,8'},
     {'anchor': '7,7', 'direction': 'up', 'name': '7,8'},
     {'anchor': '4,9', 'direction': 'right', 'name': '5,9'},
     {'anchor': '8,7', 'direction': 'right', 'name': '9,7'},
     {'anchor': '7,8', 'direction': 'right', 'name': '8,8'},
     {'anchor': '8,8', 'direction': 'up', 'name': '8,9'},
     {'anchor': '5,9', 'direction': 'right', 'name': '6,9'},
     {'anchor': '6,9', 'direction': 'right', 'name': '7,9'},
     {'anchor': '8,9', 'direction': 'right', 'name': '9,9'},
     {'anchor': '9,9', 'direction': 'down', 'name': '9,8'}
]
mazes_dict['square_large'] = {'maze': Maze(*segments_crazy, goal_squares='9,9'), 'action_range': 0.95}


segments_tree = [
    dict(name='A', anchor='origin', direction='down', times=2),
    dict(name='BR', anchor='A1', direction='right', times=4),
    dict(name='BL', anchor='A1', direction='left', times=4),
    dict(name='CR', anchor='BR3', direction='down', times=2),
    dict(name='CL', anchor='BL3', direction='down', times=2),
    dict(name='DLL', anchor='CL1', direction='left', times=2),
    dict(name='DLR', anchor='CL1', direction='right', times=2),
    dict(name='DRL', anchor='CR1', direction='left', times=2),
    dict(name='DRR', anchor='CR1', direction='right', times=2),
    dict(name='ELL', anchor='DLL1', direction='down', times=2),
    dict(name='ELR', anchor='DLR1', direction='down', times=2),
    dict(name='ERL', anchor='DRL1', direction='down', times=2),
    dict(name='ERR', anchor='DRR1', direction='down', times=2),
]
mazes_dict['square_tree'] = {'maze': Maze(*segments_tree, goal_squares=['ELL1', 'ERR1']), 'action_range': 0.95}


segments_corridor = [
    dict(name='A', anchor='origin', direction='left', times=5),
    dict(name='B', anchor='origin', direction='right', times=5)
]
mazes_dict['square_corridor'] = {'maze': Maze(*segments_corridor, goal_squares=['b4']), 'action_range': 0.95}
mazes_dict['square_corridor2'] = {'maze': Maze(*segments_corridor, goal_squares=['b4'], start_squares=['a4']),
                                  'action_range': 0.95}


_walls_to_remove = [
    ((4.5, 4.5), (7.5, 8.5)),
    ((-0.5, 0.5), (5.5, 5.5)),
    ((2.5, 2.5), (4.5, 5.5)),
    ((3.5, 4.5), (3.5, 3.5)),
    ((4.5, 4.5), (2.5, 3.5)),
    ((4.5, 5.5), (2.5, 2.5)),
    ((3.5, 4.5), (0.5, 0.5)),
    ((4.5, 5.5), (4.5, 4.5)),
    ((5.5, 5.5), (0.5, 1.5)),
    ((8.5, 8.5), (-0.5, 0.5)),
    ((6.5, 7.5), (2.5, 2.5)),
    ((7.5, 7.5), (6.5, 7.5)),
    ((7.5, 8.5), (7.5, 7.5)),
    ((8.5, 8.5), (7.5, 8.5)),
    ((7.5, 7.5), (2.5, 3.5)),
    ((8.5, 9.5), (7.5, 7.5)),
    ((7.5, 8.5), (4.5, 4.5)),
    ((8.5, 8.5), (4.5, 5.5)),
    ((5.5, 6.5), (7.5, 7.5)),
    ((3.5, 4.5), (7.5, 7.5)),
    ((4.5, 4.5), (6.5, 7.5)),
    ((4.5, 4.5), (5.5, 6.5)),
    ((3.5, 3.5), (5.5, 6.5)),
    ((5.5, 5.5), (5.5, 6.5)),
    ((3.5, 4.5), (6.5, 6.5)),
    ((4.5, 5.5), (6.5, 6.5)),
    ((1.5, 1.5), (7.5, 8.5)),
    ((2.5, 2.5), (5.5, 6.5)),
    ((0.5, 0.5), (4.5, 5.5)),
    ((1.5, 1.5), (5.5, 6.5)),
    ((4.5, 4.5), (4.5, 5.5)),
    ((5.5, 5.5), (1.5, 2.5)),
    ((5.5, 5.5), (2.5, 3.5)),
    ((5.5, 5.5), (3.5, 4.5)),
    ((6.5, 7.5), (8.5, 8.5)),
    ((7.5, 7.5), (8.5, 9.5)),
    ((0.5, 0.5), (8.5, 9.5)),
    ((0.5, 1.5), (8.5, 8.5)),
    ((-0.5, 0.5), (7.5, 7.5)),
    ((0.5, 1.5), (6.5, 6.5)),
    ((0.5, 0.5), (6.5, 7.5)),
    ((2.5, 2.5), (6.5, 7.5)),
    ((2.5, 2.5), (7.5, 8.5)),
    ((2.5, 3.5), (8.5, 8.5)),
    ((3.5, 4.5), (8.5, 8.5)),
    ((4.5, 5.5), (8.5, 8.5)),
    ((5.5, 6.5), (8.5, 8.5)),
    ((7.5, 8.5), (5.5, 5.5)),
    ((8.5, 9.5), (6.5, 6.5)),
    ((8.5, 8.5), (5.5, 6.5)),
    ((7.5, 8.5), (3.5, 3.5)),
    ((8.5, 9.5), (2.5, 2.5)),
    ((8.5, 8.5), (2.5, 3.5)),
]
_walls_to_add = [
    ((-0.5, 0.5), (4.5, 4.5)),
    ((0.5, 1.5), (4.5, 4.5)),
    ((2.5, 3.5), (4.5, 4.5)),
    ((4.5, 4.5), (3.5, 4.5)),
    ((4.5, 4.5), (2.5, 3.5)),
    ((4.5, 4.5), (1.5, 2.5)),
    ((6.5, 6.5), (8.5, 9.5)),
]
mazes_dict['square_bottleneck'] = {'maze': Maze(*segments_crazy, goal_squares='9,9', min_wall_coord=4,
                                                walls_to_remove=_walls_to_remove, walls_to_add=_walls_to_add),
                                   'action_range': 1.0}

mazes_dict['square_upside'] = {'maze': Maze(*segments_crazy, goal_squares='9,9', min_wall_coord=4,
                                                walls_to_remove=_walls_to_remove, walls_to_add=_walls_to_add),
                                   'action_range': 0.19}

segments_empty = [
    dict(name='A', anchor='origin', direction='right', times=4),
    dict(name='B', anchor='A3', direction='down', times=4),
    dict(name='C', anchor='B3', direction='left', times=4),
    dict(name='D', anchor='C3', direction='up', times=3)
]
_walls_to_remove_forempty = [
    ((-0.5, 0.5), (-0.5, -0.5)),
    ((0.5, 1.5), (-0.5, -0.5)),
    ((1.5, 2.5), (-0.5, -0.5)),
    ((2.5, 3.5), (-0.5, -0.5)),
    ((3.5, 3.5), (-1.5, -0.5)),
    ((3.5, 3.5), (-2.5, -1.5)),
    ((3.5, 3.5), (-3.5, -2.5)),
    ((2.5, 3.5), (-3.5, -3.5)),
    ((1.5, 2.5), (-3.5, -3.5)),
    ((0.5, 1.5), (-3.5, -3.5)),
    ((0.5, 0.5), (-3.5, -2.5)),
    ((0.5, 0.5), (-2.5, -1.5)),
    ((0.5, 0.5), (-1.5, -0.5)),
]


mazes_dict['square_empty'] = {'maze': Maze(*segments_empty, goal_squares=['a1', 'a2'], 
                                                walls_to_remove=_walls_to_remove_forempty), 
                                                'action_range': 0.95}

