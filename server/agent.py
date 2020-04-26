from server.math.geom_2d import *
from shapely.geometry import Point
from copy import deepcopy
from server.setting import use_geo_image, x_lim, y_lim, use_plan,\
    random_target_out_local_view, geo_image_resolution_x, geo_image_resolution_y,\
    client_input_type, show_test, show_train, max_target_dist_in_local_view,\
    max_dist_agent_to_target, max_dist_agent_to_new_target, send_high_level_parameter
import random
from present.color import *
from server.object import Object


class Agent:
    def __init__(self, **kwargs):
        self._view_area_size = 0.02
        for key, arg in kwargs.items():
            if key.lower() == 'id':
                self._id = arg
            elif key.lower() == 'start_pos':
                self._pos = arg
            elif key.lower() == 'r':
                self._r = arg
            elif key.lower() == 'target_pos':
                self._last_target_pos = arg
                self._target_pos = arg
            elif key.lower() == 'target_r':
                self._target_r = arg
            elif key.lower() == 'vel':
                self._vel = arg
            elif key.lower() == 'view_area_size':
                self._view_area_size = arg

        self._epoch_pos = []
        self._last_action = Vector2D(0, 0)
        self._prev_pos = None
        self._reward = 0
        self._collision = False
        self._obs_collision = False
        self._land_collision = False
        self._vanish_target = False
        self._arrive_target = False
        self._arrive_last_target = False
        self._connected = False
        self._send_world = ""
        self._last_dist_to_target = 0
        self._last_dist_to_obstacle = 0
        self._first_episode = True
        self._episode_reward = 0
        self._plan_reward = 0
        self._train_episodes_reward = []
        self._test_episodes_reward = []
        self._train_plans_reward = []
        self._test_plans_reward = []
        self._last_episode_test_mode = None

    def reset(self, ag, world, episode_number_in_plan, test_mode=None):
        ag = deepcopy(ag)
        if episode_number_in_plan == 1 or not use_plan:
            if ag.pos() is None:
                while True:
                    random_x = random.random() * (x_lim[1] - x_lim[0]) + x_lim[0]
                    random_y = random.random() * (y_lim[1] - y_lim[0]) + y_lim[0]
                    if use_geo_image:
                        if world.geo_image().in_sea(random_x, random_y):
                            self._pos = Vector2D(random_x, random_y)
                            break
                    else:
                        if world.geo_image().in_map(random_x, random_y) and not any(
                                world.geo_image().geo.intersects(Point(random_x, random_y).buffer(self.r()))):
                            self._pos = Vector2D(random_x, random_y)
                            break
            else:
                self._pos = ag.pos()
            if ag.target_pos() is None:
                while True:
                    if use_plan:
                        random_x = (2 * random.random() - 1.0) * max_dist_agent_to_new_target + self.pos().x()
                        random_y = (2 * random.random() - 1.0) * max_dist_agent_to_new_target + self.pos().y()
                    else:
                        random_x = (2 * random.random() - 1.0) * max_dist_agent_to_new_target + self.pos().x()
                        random_y = (2 * random.random() - 1.0) * max_dist_agent_to_new_target + self.pos().y()
                    if use_geo_image:
                        if world.geo_image().in_sea(random_x, random_y):
                            self._target_pos = Vector2D(random_x, random_y)
                            break
                    else:
                        if world.geo_image().in_map(random_x, random_y) and not any(
                                world.geo_image().geo.intersects(Point(random_x, random_y).buffer(self.target_r()))):
                            self._target_pos = Vector2D(random_x, random_y)
                            break
            else:
                pass
        if use_plan:
            if episode_number_in_plan == 1:
                if ag.target_pos() is None:
                    self._last_target_pos = self.target_pos()
                else:
                    self._last_target_pos = ag.target_pos()
        else:
            if ag.target_pos() is None:
                self._last_target_pos = self.target_pos()
        self._last_dist_to_target = self._pos.dist(self._target_pos)
        if use_geo_image:
            self._last_dist_to_obstacle = world.geo_image().get_nearest_field(self.pos().x(), self.pos().y())
        else:
            self._last_dist_to_obstacle = min(world.geo_image().geo.distance(self.point()))

        self._prev_pos = None
        self._epoch_pos = []
        self._last_action = Vector2D(0, 0)
        self._reward = 0
        self._collision = False
        self._obs_collision = False
        self._land_collision = False
        self._vanish_target = False
        self._arrive_target = False
        self._arrive_last_target = False
        self._first_episode = True
        self._epoch_pos.append(deepcopy(self.pos()))
        self._last_episode_test_mode = test_mode
        if use_plan:
            if episode_number_in_plan == 1:
                self._plan_reward = 0
        self._episode_reward = 0

    def post_process(self, episode_number=1, call_from_episode=False):
        if call_from_episode:
            if self._last_episode_test_mode:
                self._test_episodes_reward.append(self._episode_reward)
            else:
                self._train_episodes_reward.append(self._episode_reward)

        if use_plan and call_from_episode is False:
            if self._last_episode_test_mode:
                self._test_plans_reward.append(self._plan_reward / episode_number)
            else:
                self._train_plans_reward.append(self._plan_reward / episode_number)

    def update(self):
        self._first_episode = False
        self._prev_pos = Vector2D(self.pos().x(), self.pos().y())
        self._pos += self._last_action
        self._epoch_pos.append(deepcopy(self.pos()))

    def episode_reward(self):
        return self._episode_reward

    def plan_reward(self):
        return self._plan_reward

    def train_episodes_reward(self):
        return self._train_episodes_reward

    def test_episodes_reward(self):
        return self._test_episodes_reward

    def train_plans_reward(self):
        return self._train_plans_reward

    def test_plans_reward(self):
        return self._test_plans_reward

    def is_first_episode(self):
        return self._first_episode

    def set_first_episode_false(self):
        self._first_episode = False

    def pos(self) -> Vector2D:
        return self._pos

    def prev_pos(self) -> Vector2D:
        return self._prev_pos

    def epoch_pos(self) -> list:
        return self._epoch_pos

    def vel(self) -> Vector2D:
        return self._vel

    def point(self) -> Point:
        return Point(self.pos().x(), self.pos().y()).buffer(self.r())

    def circle(self) -> Circle2D:
        return Circle2D(self.pos(), self.r())

    def target_pos(self) -> Vector2D:
        return self._target_pos

    def set_target_pos(self, target: Vector2D):
        self._target_pos = target
        self._last_dist_to_target = self._pos.dist(self._target_pos)

    def last_target_pos(self) -> Vector2D:
        return self._last_target_pos

    def target_pos_list(self) -> list:
        return [self._target_pos.x(), self._target_pos.y()]

    def target_point(self) -> Point:
        return Point(self._target_pos.x(), self._target_pos.y()).buffer(self.target_r())

    def target_circle(self) -> Circle2D:
        return Circle2D(self.target_pos(), self.target_r())

    def last_target_point(self) -> Point:
        return Point(self._last_target_pos.x(), self._last_target_pos.y()).buffer(self.target_r())

    def last_target_circle(self) -> Circle2D:
        return Circle2D(self._last_target_pos, self.target_r())

    def target_r(self):
        return self._target_r

    def r(self):
        return self._r

    def set_collision(self):
        self._collision = True

    def collision(self):
        return self._collision

    def set_land_collision(self):
        self.set_collision()
        self._land_collision = True

    def land_collision(self):
        return self._land_collision

    def set_obs_collision(self):
        self.set_collision()
        self._obs_collision = True

    def obs_collision(self):
        return self._obs_collision

    def set_vanish_target(self):
        self._vanish_target = True

    def vanish_target(self):
        return self._vanish_target

    def is_vanish_target(self, world):
        if use_plan:
            geo_x_dif = world.geo_image().geo_xlim[1] - world.geo_image().geo_xlim[0]
            geo_y_dif = world.geo_image().geo_ylim[1] - world.geo_image().geo_ylim[0]
            self_xx = int(
                (self.pos().x() - world.geo_image().geo_xlim[0]) / geo_x_dif * world.geo_image().resolution[0])
            self_yy = int(
                (self.pos().y() - world.geo_image().geo_ylim[0]) / geo_y_dif * world.geo_image().resolution[1])

            target_xx = int(
                (self.target_pos().x() - world.geo_image().geo_xlim[0]) / geo_x_dif * world.geo_image().resolution[0])
            target_yy = int(
                (self.target_pos().y() - world.geo_image().geo_ylim[0]) / geo_y_dif * world.geo_image().resolution[1])
            target_xx -= self_xx
            target_yy -= self_yy
            target_xx += 25
            target_yy += 25

            if target_xx > 50 or target_yy > 50 or target_xx < 0 or target_yy < 0:
                return True
            return False
        else:
            if self.pos().dist(self.target_pos()) > max_dist_agent_to_target:
                return True
            return False

    def id(self):
        return self._id

    def set_id(self, id_):
        self._id = id_

    def r1(self, world):
        if self._vanish_target:
            self._reward = -5.0
        elif self.land_collision():
            self._reward = -5.0
        elif self.obs_collision():
            self._reward = -5.0
        elif self._arrive_target:
            self._reward = 5.0
        else:
            new_target_dist = self.pos().dist(self._target_pos)
            if use_geo_image:
                new_obstacle_dist = world.geo_image().get_nearest_field(self.pos().x(), self.pos().y())
            else:
                new_obstacle_dist = min(world.geo_image().geo.distance(self.point()))

            max_dist = math.sqrt(pow(x_lim[1] - x_lim[0], 2) + pow(y_lim[1] - y_lim[0], 2))
            target_reward = (self._last_dist_to_target - new_target_dist) * 1000
            obstacle_reward = (self._last_dist_to_obstacle - new_obstacle_dist) * 20
            self._reward = target_reward - obstacle_reward
            self._reward -= 0.01
            self._last_dist_to_obstacle = new_obstacle_dist
            self._last_dist_to_target = new_target_dist

    # 1     -5      +5            diff(o,t)
    # 2     0       0             diff(o,t)
    # 3     -5      +5            -1
    # 4                            -1
    # 5     -4.3                  -dist
    # 6     5
    # 7             -100          -1
    # 8             -(100 - ep)   -1
    # 9     0.2     -0.2     diff(t)

    def update_reward(self, world):
        self.r1(world)
        self._episode_reward += self._reward
        self._plan_reward += self._reward

    def reward(self):
        return self._reward

    def set_arrive_target(self):
        self._arrive_target = True

    def set_arrive_last_target(self):
        self._arrive_last_target = True

    def arrive_target(self):
        return self._arrive_target

    def arrive_last_target(self):
        return self._arrive_last_target

    def set_connected(self):
        self._connected = True

    def connected(self):
        return self._connected

    def send_world(self):
        return self._send_world

    def set_send_world_position(self):
        self._send_world = str(self.pos())

    def show_view(self, ax):
        s = eval(self._send_world)['view']
        gx = []
        gy = []
        ox = []
        oy = []
        tx = []
        ty = []
        sx = []
        sy = []
        for x in range(51):
            for y in range(51):
                if 0.499 < s[x][y] < 0.51:
                    gx.append(x)
                    gy.append(y)
                elif 0.29 < s[x][y] < 0.31:
                    ox.append(x)
                    oy.append(y)
                if s[x][y] > 0.9:
                    tx.append(x)
                    ty.append(y)

        ax.plot(gx, gy, 'o', color='saddlebrown')
        ax.plot(ox, oy, 'o', color='orange')
        ax.plot(tx, ty, 'go')
        self_poses = self.calc_point_in_local_view([25, 25], self.r(), False)
        for sp in self_poses:
            sx.append(sp[0])
            sy.append(sp[1])

        ax.plot(sx, sy, 'o', color=Color.agent.value)
        ax.set_xlim([0, 50])
        ax.set_ylim([0, 50])

    def calc_point_in_local_view(self, pos, r, is_target=False):
        x_lim_diff = x_lim[1] - x_lim[0]
        y_lim_diff = y_lim[1] - y_lim[0]
        width = int(2.0 * r / x_lim_diff * geo_image_resolution_x) + 1
        height = int(2.0 * r / y_lim_diff * geo_image_resolution_y) + 1
        if width % 2 == 0:
            width += 1

        if height % 2 == 0:
            height += 1
        min_xx = int((width - 1) / 2)
        min_yy = int((height - 1) / 2)

        if is_target:
            min_xx /= 2
            min_yy /= 2
            min_xx = int(min_xx)
            min_yy = int(min_yy)
        poses = [pos]

        for xx in range(-min_xx, min_xx + 1):
            for yy in range(-min_yy, min_yy + 1):
                poses.append([pos[0] + xx, pos[1] + yy])

        return poses

    def set_send_world_local(self, world):
        view_area = None
        geo_x_dif = world.geo_image().geo_xlim[1] - world.geo_image().geo_xlim[0]
        geo_y_dif = world.geo_image().geo_ylim[1] - world.geo_image().geo_ylim[0]

        self_xx = int((self.pos().x() - world.geo_image().geo_xlim[0]) / geo_x_dif * world.geo_image().resolution[0])
        self_yy = int((self.pos().y() - world.geo_image().geo_ylim[0]) / geo_y_dif * world.geo_image().resolution[1])

        if client_input_type in ['image', 'imageparam'] or any([show_test, show_train]):
            view_area = [[0.5 for _ in range(51)] for _ in range(51)]
            xi = 0
            for xx in range(self_xx - 25, self_xx + 25 + 1):
                yi = 0
                for yy in range(self_yy - 25, self_yy + 25 + 1):
                    if xx >= world.geo_image().resolution[0] or yy >= world.geo_image().resolution[1]:
                        pass
                    elif xx < 0 or yy < 0:
                        pass
                    else:
                        if world.geo_image().image[xx][yy] == 0:
                            view_area[xi][yi] = 0
                    yi += 1
                xi += 1

            target_xx = int(
                (self.target_pos().x() - world.geo_image().geo_xlim[0]) / geo_x_dif * world.geo_image().resolution[0])
            target_yy = int(
                (self.target_pos().y() - world.geo_image().geo_ylim[0]) / geo_y_dif * world.geo_image().resolution[1])
            target_xx -= self_xx
            target_yy -= self_yy
            target_xx += 25
            target_yy += 25

            targets = self.calc_point_in_local_view([target_xx, target_yy], self.target_r(), True)

            for t in targets:
                if t[0] > 50 or t[1] > 50 or t[0] < 0 or t[1] < 0:
                    continue
                if 0.49 < view_area[t[0]][t[1]] < 0.51:
                    continue
                view_area[t[0]][t[1]] = 0.99

            for o in range(len(world._objects)):
                obj: Object = world.get_object(o)
                obj_xx = int(
                    (obj.pos().x() - world.geo_image().geo_xlim[0]) / geo_x_dif * world.geo_image().resolution[0])
                obj_yy = int(
                    (obj.pos().y() - world.geo_image().geo_ylim[0]) / geo_y_dif * world.geo_image().resolution[1])
                obj_xx -= self_xx
                obj_yy -= self_yy
                obj_xx += 25
                obj_yy += 25
                objects = self.calc_point_in_local_view([obj_xx, obj_yy], obj.r())
                for o in objects:
                    if o[0] > 50 or o[1] > 50 or o[0] < 0 or o[1] < 0:
                        continue
                    view_area[o[0]][o[1]] = 0.3

        high_level_parameter = []
        if send_high_level_parameter:
            move = [(-1, 0),
                    (-1, 1),
                    (-1, -1),
                    (0, 1),
                    (0, -1),
                    (1, 0),
                    (1, 1),
                    (1, -1)]
            objects_xx_yy = []
            for o in range(len(world._objects)):
                obj: Object = world.get_object(o)
                obj_xx = int(
                    (obj.pos().x() - world.geo_image().geo_xlim[0]) / geo_x_dif * world.geo_image().resolution[0])
                obj_yy = int(
                    (obj.pos().y() - world.geo_image().geo_ylim[0]) / geo_y_dif * world.geo_image().resolution[1])
                objects = self.calc_point_in_local_view([obj_xx, obj_yy], obj.r())
                for pos in objects:
                    objects_xx_yy.append(pos)

            for m in move:
                pos_xx = int(self_xx)
                pos_yy = int(self_yy)
                counter = 1
                while True:
                    pos_xx += m[0]
                    pos_yy += m[1]
                    if not world.geo_image().in_sea_xxyy(pos_xx, pos_yy):
                        break
                    if [pos_xx, pos_yy] in objects_xx_yy:
                        break
                    counter += 1
                high_level_parameter.append(round(counter / 1000, 4))
        self._send_world = {'view': view_area, 'param': [self.pos().x(), self.pos().y(), self.last_target_pos().x(),
                                                         self.last_target_pos().y()],
                            'hparam': high_level_parameter}
        self._send_world = str(self._send_world)

    def set_send_world(self, world):
        # self.set_send_world_position()
        self.set_send_world_local(world)

        # self._send_world = world

    def is_pos_in_local_view(self, pos):
        geo_x_dif = x_lim[1] - x_lim[0]
        geo_y_dif = y_lim[1] - y_lim[0]
        self_xx = int((self.pos().x() - x_lim[0]) / geo_x_dif * geo_image_resolution_x)
        self_yy = int((self.pos().y() - y_lim[0]) / geo_y_dif * geo_image_resolution_y)

        target_xx = int((self.target_pos().x() - x_lim[0]) / geo_x_dif * geo_image_resolution_x)
        target_yy = int((self.target_pos().y() - y_lim[0]) / geo_y_dif * geo_image_resolution_y)
        target_xx -= self_xx
        target_yy -= self_yy
        target_xx += 25
        target_yy += 25

        if target_xx > 50 or target_yy > 50 or target_xx < 0 or target_yy < 0:
            return False
        return True

    def set_last_action(self, action):
        if action == 4:
            self._last_action = Vector2D(0, self.vel().y())
        elif action == 7:
            self._last_action = Vector2D(0, self.vel().y())
        elif action == 1:
            self._last_action = Vector2D(0, -self.vel().y())
        elif action == 5:
            self._last_action = Vector2D(self.vel().x(), 0)
        elif action == 3:
            self._last_action = Vector2D(-self.vel().x(), 0)
        elif action == 8:
            self._last_action = Vector2D(self.vel().x(), self.vel().y())
        elif action == 2:
            self._last_action = Vector2D(self.vel().x(), -self.vel().y())
        elif action == 6:
            self._last_action = Vector2D(-self.vel().x(), self.vel().y())
        elif action == 0:
            self._last_action = Vector2D(-self.vel().x(), -self.vel().y())

    def __repr__(self):
        return 'id:' + str(self.id()) + 'pos:' + str(self.pos())
