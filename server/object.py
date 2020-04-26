from enum import Enum
from server.math.geom_2d import *
from shapely.geometry import Point
from ais.ais_to_episode import ais_to_episode
import random
from server.setting import change_position_of_const_object


class ObjectType(Enum):
    StaticObject = 0,
    DynamicObject = 1,
    AISObject = 2


class Object:
    def __init__(self, id_, object_type, r, episode_numbers=1):
        self._object_type = object_type
        self._r = r
        self._pos: Vector2D = None
        self._episode_numbers = episode_numbers
        self._id = id_

    def pos(self) -> Vector2D:
        return self._pos

    def pos_list(self) -> list:
        return [self.pos().x(), self.pos().y()]

    def r(self):
        return self._r

    def object_type(self):
        return self._object_type

    def update(self, cycle=0):
        pass

    def point(self) -> Point:
        return Point(self.pos().x(), self.pos().y()).buffer(self.r())

    def circle(self) -> Circle2D:
        return Circle2D(self.pos(), self.r())

    def epoch_pos(self):
        pass

    def reset(self, obj):
        pass

    def id(self):
        return self._id


class StaticVessel(Object):
    def __init__(self, id_, object_type, pos, r):
        Object.__init__(self, id_, object_type, r)
        self._start_pos: Vector2D = pos
        self._pos: Vector2D = self._start_pos
        self._circle = Circle2D(self.pos(), self.r())
        self._point = Point(self.pos().x(), self.pos().y()).buffer(self.r())

    def circle(self) -> Circle2D:
        return self._circle

    def point(self) -> Point:
        return self._point

    def reset(self, obj):
        if change_position_of_const_object:
            self._pos = Vector2D(self._start_pos.x(), self._start_pos.y())
            self._pos._x += (random.random() - 0.5) * 2.0 * 0.002
            self._pos._y += (random.random() - 0.5) * 2.0 * 0.002
            self._circle = Circle2D(self.pos(), self.r())
            self._point = Point(self.pos().x(), self.pos().y()).buffer(self.r())


class DynamicVessel(Object):
    def __init__(self, id_, object_type, start_pos, target_pos, r, episode_numbers):
        Object.__init__(self, id_, object_type, r, episode_numbers)
        self._start_pos = start_pos
        self._target_pos = target_pos
        self._pos = start_pos
        self._epoch_pos = []
        self._direct = -1

    def update(self, episode=0):
        move = self._target_pos - self._start_pos
        if episode % self._episode_numbers == 0:
            self._direct *= -1
        move *= ((episode % self._episode_numbers) / self._episode_numbers)
        if self._direct > 0:
            self._pos = self._start_pos + move
        else:
            self._pos = self._target_pos - move
        self._epoch_pos.append(self._pos)

    def epoch_pos(self):
        return self._epoch_pos

    def reset(self, obj):
        self._pos = self._start_pos
        self._epoch_pos = []
        self._epoch_pos.append(self._pos)


class AISVessel(Object):
    def __init__(self, id_, object_type, path, r, episode_real_time):
        Object.__init__(self, id_, object_type, r)
        self._poses = ais_to_episode(path, episode_real_time)
        self._cycle = 0
        self._epoch_pos = []

    def reset(self, obj):
        self._cycle = random.randint(0, len(self._poses))
        self._pos = Vector2D(self._poses[self._cycle][0], self._poses[self._cycle][1])
        self._epoch_pos = []
        self._epoch_pos.append(self._pos)

    def update(self, cycle=0, max_cycle=1):
        new_cycle = min(self._cycle + cycle, len(self._poses) - 1)
        self._pos = Vector2D(self._poses[new_cycle][0], self._poses[new_cycle][1])
        self._epoch_pos.append(self._pos)

    def epoch_pos(self):
        return self._epoch_pos

