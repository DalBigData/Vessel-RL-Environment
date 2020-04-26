import enum


class Color(enum.Enum):
    sea = [31 / 256, 184 / 256, 215 / 256],
    ground = [107 / 256, 64 / 256, 12 / 256],
    static_object = [255 / 256, 242 / 256, 0 / 256],
    target = 'lightgreen',
    pre_target = 'lightgray'
    last_target = 'green',
    dynamic_object = [255 / 256, 0 / 256, 247 / 256],
    ais_object = [255 / 256, 0 / 256, 247 / 256],
    agent = 'tomato'


print(Color.sea.value)
