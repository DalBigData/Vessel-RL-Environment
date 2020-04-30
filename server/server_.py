import server.setting as setting
import matplotlib.pyplot as plt
from descartes.patch import Polygon
from shapely.geometry import Polygon, Point, LineString
from server.agent import Agent
from server.config_vessels import vessel_list as vessel_list
import geopandas as gpd
from server.object import Object, ObjectType
import threading
import queue
from server.Message import *
import time
import copy
from datetime import datetime
# import matplotlib; matplotlib.use("TkAgg")
from present.color import Color
import signal
from server.geo_image import GeoImage
from typing import List
import os
from server.dynamic_planner import Planner
from server.math.geom_2d import *
from enum import Enum

now = datetime.now()
current_time = now.strftime("%Y-%m-%d-%H-%M-%S")
global_cycle = 0


class EpisodeResult(Enum):
    LandCollision = "LandCollision"
    ObsCollision = "ObsCollision"
    ArriveTarget = "ArriveTarget"
    VanishTarget = "VanishTarget"
    Other = "Other"

    def __repr__(self):
        return self.value

    def __str__(self):
        return self.value


test_what_happen_episode: List[EpisodeResult] = []
test_what_happen_plan: List[EpisodeResult] = []
test_what_happen_plan_len: List = []
what_happen_episode: List[EpisodeResult] = []
what_happen_plan: List[EpisodeResult] = []
what_happen_plan_len: List = []


class Plan:
    planner = None

    def __init__(self, agent: Agent, show):
        if Plan.planner is None:
            planner = Planner()
            Plan.planner = planner
        self.agent = agent
        self.origin_path_to_target = Plan.planner.get_path(agent.pos(), agent.last_target_pos())
        self.path_len = len(self.origin_path_to_target)
        self.next_targets = []
        if show:
            try:
                Planner.ploter(Plan.planner, agent.pos(), self.origin_path_to_target, 'Douglas0.001')
            except Exception as e:
                print(e)

        self.const_path_to_target = Planner.douglas_peucker(self.origin_path_to_target, 0.001)
        self.const_path_to_target.append((agent.last_target_pos().x(), agent.last_target_pos().y()))
        if show:
            print(agent.pos(), agent.last_target_pos())
            print(self.const_path_to_target)
            try:
                Planner.ploter(Plan.planner, agent.pos(), self.const_path_to_target, 'Douglas0.001')
            except Exception as e:
                print(e)
        self.path_to_target = copy.deepcopy(self.const_path_to_target)
        if len(self.path_to_target) == 0:
            raise Exception("cant find path from {} to {}".format(agent.pos(), agent.last_target_pos()))

    def get_plan_len(self, start, target):
        origin_path_to_target = Plan.planner.get_path(start, target)
        return len(origin_path_to_target)

    def get_next_target(self):
        max_target_dist = setting.max_target_dist_in_local_view * (
                setting.x_lim[1] - setting.x_lim[0]) / setting.geo_image_resolution_x
        best_candidate_in = None
        best_candidate_in_i = 0
        best_candidate_out = None
        i = 0
        for candidate in self.path_to_target:
            candidate_pos = Vector2D(candidate[0], candidate[1])
            if abs(self.agent.pos().x() - candidate_pos.x()) < max_target_dist and abs(
                    self.agent.pos().y() - candidate_pos.y()) < max_target_dist:
                best_candidate_in = candidate_pos
                best_candidate_in_i = i
            elif best_candidate_out is None:
                best_candidate_out = candidate_pos
            i += 1
        if best_candidate_in is not None:
            for i in range(best_candidate_in_i + 1):
                del self.path_to_target[0]
            self.next_targets.append(best_candidate_in)
            return best_candidate_in
        top_left_x = self.agent.pos().x()
        top_left_y = self.agent.pos().y()

        top_left_x -= max_target_dist
        top_left_y -= max_target_dist

        view_rec = Rect2D(top_left_x, top_left_y, max_target_dist * 2.0, max_target_dist * 2.0)
        line_to_target = Line2D(self.agent.pos(), best_candidate_out)

        target_number, target1, target2 = view_rec.intersection(line_to_target)
        targets = [target1, target2]
        best_target = targets[0]
        best_dist = 1000
        for t in targets:
            if not self.agent.is_pos_in_local_view(t):
                if target_number == 1:
                    break
                continue
            if t.dist(best_candidate_out) < best_dist:
                best_dist = t.dist(best_candidate_out)
                best_target = t
            if target_number == 1:
                break
        self.next_targets.append(best_target)
        return best_target


class World:
    _cycle = 0

    def __init__(self, server):
        self._server = server
        self._map: Polygon = server.map()
        self._geo_image: GeoImage = server.geo_image()
        self._objects = server.objects
        min_max_x = [setting.x_lim[1], setting.x_lim[1], setting.x_lim[0], setting.x_lim[0]]
        min_max_y = [setting.y_lim[0], setting.y_lim[1], setting.y_lim[1], setting.y_lim[0]]
        self._ground: Polygon = Polygon(zip(min_max_x, min_max_y))
        self._agent = None

    def reset_agent(self, episode_number_in_plan, test_mode):
        self._agent: Agent = self.server().agent()
        self._agent.reset(setting.agent, self, episode_number_in_plan, test_mode)

    def reset_object(self):
        World._cycle = 0
        for o in self._objects:
            id_ = o.id()
            o.reset(vessel_list[id_])

    def server(self):
        return self._server

    def show(self, show, show_agent_episode_pos=False, name=''):
        if not show:
            return
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(name, fontsize=10)

        axs[0] = self._map.plot(ax=axs[0], color=Color.ground.value[0])
        axs[0].set_title(name)
        for t in axs[0].get_xticklabels():
            t.set_rotation(90)
        axs[0].set_facecolor(Color.sea.value[0])
        static_points = []
        dynamic_points = []
        for o in self._objects:
            if o.object_type() is ObjectType.StaticObject:
                point = Point(o.pos().x(), o.pos().y()).buffer(o.r() * 3)
                static_points.append(point)
            else:
                point = Point(o.pos().x(), o.pos().y()).buffer(o.r() * 3)
                dynamic_points.append(point)
        static_points_gdf = gpd.GeoDataFrame(index=[i for i in range(len(static_points))], crs={'init': 'epsg:4326'},
                                             geometry=static_points)
        axs[0] = static_points_gdf.plot(ax=axs[0], color=Color.static_object.value[0])
        dynamic_points_gdf = gpd.GeoDataFrame(index=[i for i in range(len(dynamic_points))], crs={'init': 'epsg:4326'},
                                              geometry=dynamic_points)
        axs[0] = dynamic_points_gdf.plot(ax=axs[0], color=Color.dynamic_object.value[0])
        agents_point = []
        if not show_agent_episode_pos:
            point = Point(self.agent().pos().x(), self.agent().pos().y()).buffer(self.agent().r() * 9)
            agents_point.append(point)
        else:
            ag = self.agent()
            for p in ag.epoch_pos():
                point = Point(p.x(), p.y()).buffer(ag.r() * 9)
                agents_point.append(point)
        agent_points_gdf = gpd.GeoDataFrame(index=[i for i in range(len(agents_point))], crs={'init': 'epsg:4326'},
                                            geometry=agents_point)
        axs[0] = agent_points_gdf.plot(ax=axs[0], color=Color.agent.value)
        targets_point = list()
        targets_point.append(self.agent().target_point().buffer(self.agent().target_r()))
        target_points_gdf = gpd.GeoDataFrame(index=[i for i in range(len(targets_point))], crs={'init': 'epsg:4326'},
                                             geometry=targets_point)
        axs[0] = target_points_gdf.plot(ax=axs[0], color=Color.target.value)

        targets_point = list()
        targets_point.append(self.agent().last_target_point().buffer(self.agent().target_r()))
        target_points_gdf = gpd.GeoDataFrame(index=[i for i in range(len(targets_point))], crs={'init': 'epsg:4326'},
                                             geometry=targets_point)
        axs[0] = target_points_gdf.plot(ax=axs[0], color=Color.last_target.value)

        axs[0].set_xlim(setting.x_lim)
        axs[0].set_ylim(setting.y_lim)
        self.agent().show_view(axs[1])
        plt.show()

    def get_object(self, id_) -> Object:
        return self._objects[id_]

    def get_object_number(self) -> int:
        return len(self._objects)

    def geo_image(self) -> GeoImage:
        return self._geo_image

    def agent(self) -> Agent:
        return self._agent

    def update(self):
        global global_cycle
        global_cycle += 1
        for o in self._objects:
            o.update(self._cycle)

        self.agent().update()

        a = self.agent()
        if setting.use_geo_image:
            if not self._geo_image.in_sea(a.pos().x(), a.pos().y()):
                a.set_land_collision()
        else:
            if any(self._map.intersects(a.point())):
                a.set_land_collision()

        if not self._ground.contains(a.point()):
            a.set_land_collision()

        agent_line = LineString([[a.pos().x(), a.pos().y()], [a.prev_pos().x(), a.prev_pos().y()]]).buffer(a.r())
        for id in range(len(self._objects)):
            o = self.get_object(id)
            if o.circle().has_intersection(a.circle()):
                a.set_obs_collision()
            elif o.pos().dist(a.pos()) < 0.0204:
                if agent_line.intersects(o.point()):
                    a.set_obs_collision()

        if a.circle().has_intersection(a.target_circle()):
            a.set_arrive_target()
        elif a.pos().dist(a.target_pos()) < 0.0102:
            if agent_line.intersects(a.target_point()) > 0:
                a.set_arrive_target()

        if a.circle().has_intersection(a.last_target_circle()):
            a.set_arrive_last_target()
        elif a.pos().dist(a.last_target_pos()) < 0.0102:
            if agent_line.intersects(a.last_target_point()) > 0:
                a.set_arrive_last_target()

        if a.is_vanish_target(self):
            a.set_vanish_target()

        a.update_reward(self)
        World._cycle += 1

    def send_to_clients(self, is_test, is_last):
        agent = self.agent()
        agent.set_send_world(self)
        status = 'normal'
        if agent.is_first_episode():
            status = 'first'
            agent.set_first_episode_false()

        if agent.collision() or agent.arrive_target() or agent.vanish_target() or is_last:
            if status == 'normal':
                status = 'end'
            else:
                status = 'end'
        if is_test:
            epoch_type = 'valid'
        else:
            epoch_type = 'learn'
        message = MessageClientWorld(World._cycle, agent.send_world(), agent.reward(), status, epoch_type).build()
        self.server().sender_pipe.send(message)


class EpisodeSimulation:
    def __init__(self, server, test_mode, show, episode_number):
        self._server = server
        self.received_action = 0
        self._test_mode = test_mode
        self._show = show
        self._file_name = str(episode_number[0]) + '_' + str(episode_number[1])

    def server(self):
        return self._server

    def world(self):
        return self.server().world()

    def run(self):
        global test_what_happen_episode, what_happen_episode
        happen = False
        agent: Agent = self.server().agent()
        if not setting.use_plan:
            self.world().reset_object()
        step_counter = 0
        for i in range(setting.max_step_in_episode):
            self.world().send_to_clients(self._test_mode, i == setting.max_step_in_episode - 1)

            if setting.save_local_view:
                agent.add_local_view(agent.send_world())
            self.world().show(show=self._show, name=self._file_name)
            if agent.land_collision():
                what_happen_episode.append(EpisodeResult.LandCollision)
                if self._test_mode:
                    test_what_happen_episode.append(EpisodeResult.LandCollision)
                happen = True
                break
            elif agent.obs_collision():
                what_happen_episode.append(EpisodeResult.ObsCollision)
                if self._test_mode:
                    test_what_happen_episode.append(EpisodeResult.ObsCollision)
                happen = True
                break
            elif self.server().agent().arrive_target():
                what_happen_episode.append(EpisodeResult.ArriveTarget)
                if self._test_mode:
                    test_what_happen_episode.append(EpisodeResult.ArriveTarget)
                happen = True
                break
            elif self.server().agent().vanish_target():
                what_happen_episode.append(EpisodeResult.VanishTarget)
                if self._test_mode:
                    test_what_happen_episode.append(EpisodeResult.VanishTarget)
                happen = True
                break
            elif i == setting.max_step_in_episode - 1:
                break
            self.received_action = 0
            step_counter += 1
            while True:
                try:
                    msg = self.server().action_queue.get(block=True, timeout=30.0)
                except Exception:
                    print('server did not receive message')
                    continue
                self.action_parse(msg)
                if self.received_action == 1:
                    break
            while self.server().action_queue.qsize() > 0:
                self.server().action_queue.get()
            self.world().update()

        self.server().agent().post_process(call_from_episode=True)
        self.save(step_counter + 1)

        if not happen:
            what_happen_episode.append(EpisodeResult.Other)
            if self._test_mode:
                test_what_happen_episode.append(EpisodeResult.Other)
        if self.server().agent().collision():
            return False, step_counter
        elif self.server().agent().arrive_last_target():
            return False, step_counter
        elif self.server().agent().arrive_target():
            return True, step_counter
        elif self.server().agent().vanish_target():
            return False, step_counter
        return False, step_counter

    def action_parse(self, msg):
        message: MessageClientAction = parse(msg)
        if message.type is not 'MessageClientAction':
            return False
        self.server().agent().set_last_action(message.action)
        self.received_action += 1
        return True

    def save(self, step_counter):
        if setting.use_plan:
            return
        ag: Agent = self.server().agent()
        file = open(setting.result_path + setting.result_prefix + current_time + '_' + self._file_name, 'w')
        param = {'size': [setting.x_lim, setting.y_lim], 'path': setting.shape_path, 'crs': setting.map_crs,
                 'last_target': [ag.target_r(), ag.last_target_pos()]}
        result = {'happen': what_happen_episode[-1], 'happen_len': None,
                  'ER': ag.episode_reward(),
                  'PR': ag.episode_reward(),
                  'APR': ag.episode_reward()}
        plan_log = {}
        res = ['param,' + str(param), 'results,' + str(result), 'plan,' + str(plan_log)]
        for s in range(step_counter):
            try:
                r = ag.plan_rewards()[s]
            except:
                r = None
            step = {'step': s, 'data': [], 'reward': r}
            for i in range(self.server().world().get_object_number()):
                o = self.server().world().get_object(i)
                step['data'].append(o.log_string(s))
            agent_str = ['a', ag.r(), ag.plan_poses()[s].x(), ag.plan_poses()[s].y(), 0]
            target_str = ['t', ag.plan_targets()[s].x(), ag.plan_targets()[s].y()]
            step['data'].append(agent_str)
            step['data'].append(target_str)
            res.append(f'step{s},' + str(step))

        for s in range(step_counter):
            res.append(f'sent{s},' + str(ag.plan_local_views()[s]))
        res = '\n'.join(res)
        file.write(res)
        file.close()


class PlanSimulation:
    def __init__(self, server, test_mode, show, train_plan_number, test_plan_number):
        self._server = server
        self.received_action = 0
        self._test_mode = test_mode
        self._show = show
        self.train_plan_number = train_plan_number
        self.test_plan_number = test_plan_number
        self._file_name = str(train_plan_number) + '_' + str(test_plan_number)

    def server(self):
        return self._server

    def run(self):
        global test_what_happen_episode, test_what_happen_plan, test_what_happen_plan_len, \
            what_happen_episode, what_happen_plan, what_happen_plan_len
        world: World = self.server().world()
        world.reset_object()
        agent: Agent = self.server().agent()
        while True:
            try:
                world.reset_agent(1, self._test_mode)
                plan = Plan(agent, self._show)
                break
            except Exception as e:
                print('find plan in PlanSimulation', e)
                continue
        i = 0
        last_agent_position = agent.pos()
        step_counter = 0
        for i in range(1, setting.max_episode_in_plan + 1):
            next_target = plan.get_next_target()
            self._server.agent().set_target_pos(next_target)

            if not self._test_mode:
                ep = EpisodeSimulation(self._server, test_mode=self._test_mode, show=self._show,
                                       episode_number=(i, 0))
            else:
                ep = EpisodeSimulation(self._server, test_mode=self._test_mode, show=self._show,
                                       episode_number=(i, 0))
            last_agent_position = agent.pos()
            res = ep.run()
            step_counter += res[1]
            print('Episode Number:', i, 'Episode R:', self.server().agent().episode_reward(),
                  "test" if self._test_mode else "")
            if not res[0]:
                break
            world.reset_agent(i + 1, self._test_mode)

        agent.post_process(i)
        plan_len = plan.path_len
        if test_what_happen_episode[-1] == EpisodeResult.ArriveTarget:
            plan_len = (plan_len, plan_len)
        else:
            new_plan = plan.get_plan_len(last_agent_position, agent.last_target_pos())
            plan_len = (plan_len, max(0, plan_len - new_plan))

        what_happen_plan.append(what_happen_episode[-1])
        what_happen_plan_len.append(plan_len)
        if not self._test_mode:
            message = MessageEndPlan().build()
            self.server().sender_pipe.send(message)
        else:
            test_what_happen_plan.append(test_what_happen_episode[-1])
            test_what_happen_plan_len.append(plan_len)

        if self._test_mode:
            print('Plan R:', self.server().agent().test_plans_reward()[-1])
        else:
            print('Plan R:', self.server().agent().train_plans_reward()[-1])
        self.save(step_counter + 1, plan)
        return i

    def save(self, step_counter, plan: Plan):
        ag: Agent = self.server().agent()

        file = open(setting.result_path + setting.result_prefix + current_time + '_' + self._file_name, 'w')
        param = {'size': [setting.x_lim, setting.y_lim], 'path': setting.shape_path, 'crs': setting.map_crs,
                 'last_target': [ag.target_r(), ag.last_target_pos()]}
        result = {'happen': what_happen_plan[-1], 'happen_len': what_happen_plan_len[-1], 'ER': ag.episode_rewards(),
                  'PR': ag.plan_reward(),
                  'APR': ag.test_plans_reward()[-1] if self._test_mode else ag.train_plans_reward()[-1]}
        plan_log = {'points': plan.origin_path_to_target, 'douglas': plan.const_path_to_target,
                    'targets': plan.next_targets}
        res = ['param,' + str(param), 'results,' + str(result), 'plan,' + str(plan_log)]
        for s in range(step_counter):
            try:
                r = ag.plan_rewards()[s]
            except:
                r = None
            step = {'step': s, 'data': [], 'reward': r}
            for i in range(self.server().world().get_object_number()):
                o = self.server().world().get_object(i)
                step['data'].append(o.log_string(s))
            agent_str = ['a', ag.r(), ag.plan_poses()[s].x(), ag.plan_poses()[s].y(), 0]
            target_str = ['t', ag.plan_targets()[s].x(), ag.plan_targets()[s].y()]
            step['data'].append(agent_str)
            step['data'].append(target_str)
            res.append(f'step{s},' + str(step))
        if setting.save_local_view:
            for s in range(step_counter):
                res.append(f'sent{s},' + str(ag.plan_local_views()[s]))
        res = '\n'.join(res)
        file.write(res)
        file.close()


is_run = True
listener_work = True


def signal_handler(sig, frame):
    global is_run
    print('You pressed Ctrl+C!')
    is_run = False


signal.signal(signal.SIGINT, signal_handler)


def listener(receiver_pipe, action_queue: queue.Queue):
    global listener_work
    while listener_work:
        try:
            msg = receiver_pipe.recv()
            action_queue.put(msg)
        except Exception:
            continue


class Server:
    def __init__(self, info_pipe_server, action_pipe_server):
        self._agent = copy.deepcopy(setting.agent)
        self.objects = copy.deepcopy(vessel_list)
        self.receiver_pipe = action_pipe_server
        self.sender_pipe = info_pipe_server
        self.action_queue = queue.Queue(0)
        self.listener = threading.Thread(target=listener,
                                         args=(self.receiver_pipe, self.action_queue,))
        self._map: Polygon = gpd.read_file(setting.map_path).to_crs(setting.map_crs)
        self._geo_image: GeoImage = GeoImage()
        self._world = World(self)
        self.listener.start()
        self.connected_agent = 0

    def world(self):
        return self._world

    def add_agent(self, message: MessageClientConnectRequest):
        id_ = message.id
        self.agent().set_id(id_)
        action_resp = MessageClientConnectResponse(id_, self.agent().vel(), setting.x_lim[0], setting.x_lim[1],
                                                   setting.y_lim[0], setting.y_lim[1]).build()
        self.sender_pipe.send(action_resp)

    def connect(self):
        while True:
            if not is_run:
                return
            try:
                msg_address = self.action_queue.get(block=True, timeout=1)
            except Exception as c:
                continue
            message = parse(msg_address)
            if not message.type == 'ClientConnectRequest':
                continue

            self.add_agent(message)
            break

    def agent(self) -> Agent:
        return self._agent

    def map(self) -> Polygon:
        return self._map

    def geo_image(self) -> GeoImage:
        return self._geo_image

    def object(self, id_) -> Object:
        return self.object(id_)

    # def objects(self, id_) -> Object:
    #     return self.object(id_)

    def run(self):
        global is_run, listener_work, test_what_happen_episode
        if setting.use_plan:
            time.sleep(2)
            for test_plan_number in range(1, setting.test_plan_nb + 1):
                print('-' * 30)
                print('train_plan_number:', 0, 'test_plan_number:', test_plan_number)
                plan = PlanSimulation(self, test_mode=True, show=setting.show_test, train_plan_number=0,
                                      test_plan_number=test_plan_number)
                plan.run()

            episode_trained = 0
            train_plan_number = 0
            for train_plan_number in range(1, setting.train_plan_nb):
                print('-' * 30)
                print('train_plan_number:', train_plan_number, 'trained_episode_number:', episode_trained)
                plan = PlanSimulation(self, test_mode=False, show=setting.show_train,
                                      train_plan_number=train_plan_number, test_plan_number=0)
                episode_trained += plan.run()
                if train_plan_number % setting.test_plan_interval is 0:
                    for test_plan_number in range(1, setting.test_plan_nb + 1):
                        print('-' * 30)
                        print('train_plan_number:', train_plan_number, 'test_plan_number:', test_plan_number)
                        plan = PlanSimulation(self, test_mode=True, show=setting.show_test,
                                              train_plan_number=train_plan_number, test_plan_number=test_plan_number)
                        plan.run()
                    self.show_plot_reward()
                    self.save_reward()

            for test_plan_number in range(1, setting.test_plan_nb + 1):
                plan = PlanSimulation(self, test_mode=True, show=setting.show_test,
                                      train_plan_number=setting.train_plan_nb + 1, test_plan_number=test_plan_number)
                plan.run()
                print('-' * 30)
                print('train_plan_number:', train_plan_number, 'test_plan_number:', test_plan_number)
            self.show_plot_reward()
            self.save_reward()
        else:
            t_episode_number = 0

            for test_episode in range(setting.test_episode_nb):
                self.world().reset_agent(1, True)
                ep = EpisodeSimulation(self, test_mode=True, show=setting.show_test, episode_number=(0, test_episode))
                ep.run()
                print('####', t_episode_number, test_episode, self.agent().episode_reward())
            for episode in range(setting.train_episode_nb):
                t_episode_number = episode
                self.world().reset_agent(1, False)
                ep = EpisodeSimulation(self, test_mode=False, show=setting.show_train, episode_number=(episode, 0))
                ep.run()
                print('####', episode, self.agent().episode_reward())
                if episode % setting.test_interval is 0 and episode > 0:
                    for test_episode in range(setting.test_episode_nb):
                        self.world().reset_agent(1, True)
                        ep = EpisodeSimulation(self, test_mode=True, show=setting.show_train,
                                               episode_number=(episode, test_episode))
                        ep.run()
                        #         # ep.world().show(True, True, 0)
                        print(t_episode_number, test_episode, self.agent().episode_reward())
                    self.show_plot_reward()
                    self.save_reward()
                if not is_run:
                    break
            print('start validation')
            for test_episode in range(setting.test_episode_nb):
                self.world().reset_agent(1, True)
                ep = EpisodeSimulation(self, test_mode=True, show=setting.show_test,
                                       episode_number=(setting.train_episode_nb + 1, test_episode))
                ep.run()
                # ep.world().show(True, True, 0)
                print(test_episode, self.agent().episode_reward())
            self.show_plot_reward()
            self.save_reward()

        message = MessageClientDisconnect().build()
        self.sender_pipe.send(message)

        self.show_plot_reward()
        self.save_reward()
        is_run = False
        listener_work = False
        print(test_what_happen_episode)

    def show_plot_reward(self):
        ag = self.agent()
        train_reward = ag.train_plans_reward() if setting.use_plan else ag.train_episodes_reward()
        test_reward = ag.test_plans_reward() if setting.use_plan else ag.test_episodes_reward()

        avg_number = 20
        avg_train_reward = []
        i = 0
        sum_reward = 0
        while i < len(train_reward):
            sum_reward += ag.train_episodes_reward()[i]
            if (i % avg_number == 0 or i == len(train_reward) - 1) and i > 0:
                avg_train_reward.append(sum_reward / avg_number)
                sum_reward = 0
            i += 1
        plt.plot([i * avg_number + avg_number for i in range(len(avg_train_reward))], avg_train_reward, 'b')
        avg_test_number = setting.test_plan_nb if setting.use_plan else setting.test_episode_nb
        test_thr = setting.test_plan_interval if setting.use_plan else setting.test_interval
        avg_test_reward = []
        i = 0
        sum_reward = 0
        while i < len(test_reward):
            sum_reward += ag.test_episodes_reward()[i]
            if (i % avg_test_number == 0 or i == len(test_reward) - 1) and i > 0:
                avg_test_reward.append(sum_reward / avg_test_number)
                sum_reward = 0
            i += 1
        plt.plot([i * test_thr for i in range(len(avg_test_reward))], avg_test_reward, 'r')
        plt.show()
        plt.pause(1)

    def save_reward(self):
        global test_what_happen_episode, test_what_happen_plan, test_what_happen_plan_len
        ag = self.agent()
        file = open(setting.result_path + current_time + '_' + 'train_episode_reward_ag' + str(ag.id()), 'w')
        for r in ag.train_episodes_reward():
            file.write(str(r) + '\n')
        file.close()
        file = open(setting.result_path + current_time + '_' + 'test_episode_reward_ag' + str(ag.id()), 'w')
        for r in ag.test_episodes_reward():
            file.write(str(r) + '\n')
        file.close()
        if setting.use_plan:
            file = open(setting.result_path + current_time + '_' + 'train_plan_reward_ag' + str(ag.id()), 'w')
            for r in ag.train_plans_reward():
                file.write(str(r) + '\n')
            file.close()
            file = open(setting.result_path + current_time + '_' + 'test_plan_reward_ag' + str(ag.id()), 'w')
            for r in ag.test_plans_reward():
                file.write(str(r) + '\n')
            file.close()
        file = open(setting.result_path + current_time + '_' + 'happen', 'a')
        for h in test_what_happen_episode:
            file.write(str(h) + '\n')
        file.close()
        file = open(setting.result_path + current_time + '_' + 'happen_plan', 'a')
        for h in test_what_happen_plan:
            file.write(str(h) + '\n')
        file.close()
        file = open(setting.result_path + current_time + '_' + 'happen_plan_len', 'a')
        for h in test_what_happen_plan_len:
            file.write(str(h[0]) + ' ' + str(h[1]) + '\n')
        test_what_happen_plan_len = []
        file.close()
        test_what_happen_episode = []
        test_what_happen_plan = []


def main(info_pipe_server, action_pipe_server):
    if not os.path.isdir(setting.result_path):
        os.mkdir(setting.result_path)
    vs = Server(info_pipe_server, action_pipe_server)
    vs.connect()
    vs.run()
