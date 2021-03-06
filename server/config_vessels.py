from server.object import *
from server.setting import max_step_in_episode


vessel_list = [
    StaticVessel(0, ObjectType.StaticObject, Vector2D(-63.538, 44.602), 0.00019),
    StaticVessel(1, ObjectType.StaticObject, Vector2D(-63.57, 44.655), 0.00019),
    StaticVessel(2, ObjectType.StaticObject, Vector2D(-63.57, 44.66), 0.00019),
    StaticVessel(3, ObjectType.StaticObject, Vector2D(-63.578, 44.662), 0.00019),
    StaticVessel(4, ObjectType.StaticObject, Vector2D(-63.507, 44.615), 0.00019),
    StaticVessel(5, ObjectType.StaticObject, Vector2D(-63.51, 44.585), 0.00019),
    StaticVessel(6, ObjectType.StaticObject, Vector2D(-63.53, 44.59), 0.00019),
    StaticVessel(7, ObjectType.StaticObject, Vector2D(-63.535, 44.585), 0.00019),
    StaticVessel(8, ObjectType.StaticObject, Vector2D(-63.55, 44.61), 0.00019),
    StaticVessel(9, ObjectType.StaticObject, Vector2D(-63.56, 44.615), 0.00019),
    StaticVessel(10, ObjectType.StaticObject, Vector2D(-63.55, 44.62), 0.00019),
    StaticVessel(11, ObjectType.StaticObject, Vector2D(-63.54, 44.634), 0.00019),
    StaticVessel(12, ObjectType.StaticObject, Vector2D(-63.555, 44.635), 0.00019),
    StaticVessel(13, ObjectType.StaticObject, Vector2D(-63.56, 44.65), 0.00019),
    StaticVessel(14, ObjectType.StaticObject, Vector2D(-63.63, 44.68), 0.00019),
    StaticVessel(15, ObjectType.StaticObject, Vector2D(-63.64, 44.7), 0.00019),
    StaticVessel(16, ObjectType.StaticObject, Vector2D(-63.66, 44.71), 0.00019),
    StaticVessel(17, ObjectType.StaticObject, Vector2D(-63.65, 44.69), 0.00019),
    StaticVessel(18, ObjectType.StaticObject, Vector2D(-63.625, 44.695), 0.00019),
    StaticVessel(19, ObjectType.StaticObject, Vector2D(-63.635, 44.687), 0.00019),
    StaticVessel(20, ObjectType.StaticObject, Vector2D(-63.623, 44.687), 0.00019),
    StaticVessel(21, ObjectType.StaticObject, Vector2D(-63.655, 44.703), 0.00019),
    StaticVessel(22, ObjectType.StaticObject, Vector2D(-63.493, 44.59), 0.00019),
    StaticVessel(23, ObjectType.StaticObject, Vector2D(-63.543, 44.612), 0.00019),
    StaticVessel(24, ObjectType.StaticObject, Vector2D(-63.547, 44.603), 0.00019),
    StaticVessel(25, ObjectType.StaticObject, Vector2D(-63.555, 44.625), 0.00019),
    StaticVessel(26, ObjectType.StaticObject, Vector2D(-63.545, 44.593), 0.00019),
    StaticVessel(27, ObjectType.StaticObject, Vector2D(-63.52, 44.59), 0.00019),
    StaticVessel(28, ObjectType.StaticObject, Vector2D(-63.545, 44.63), 0.00019),
    StaticVessel(29, ObjectType.StaticObject, Vector2D(-63.64, 44.68), 0.00019),
    StaticVessel(30, ObjectType.StaticObject, Vector2D(-63.50, 44.6), 0.00019),
    StaticVessel(31, ObjectType.StaticObject, Vector2D(-63.65, 44.7), 0.00019),
    StaticVessel(32, ObjectType.StaticObject, Vector2D(-63.645, 44.709), 0.00019),
    StaticVessel(33, ObjectType.StaticObject, Vector2D(-63.59, 44.67), 0.00019),

    # StaticVessel(0, ObjectType.StaticObject, Vector2D(-63.64, 44.68), 0.005),
    # DynamicVessel(0, ObjectType.DynamicObject, Vector2D(-63.558, 44.66), Vector2D(-63.58, 44.66), 0.00019, 10),
    # DynamicVessel(1, ObjectType.DynamicObject, Vector2D(-63.63, 44.68), Vector2D(-63.6198, 44.6898), 0.00019, 20),
    # DynamicVessel(2, ObjectType.DynamicObject, Vector2D(-63.6153, 44.6768), Vector2D(-63.6149, 44.6834), 0.00019, 15),
    # DynamicVessel(3, ObjectType.DynamicObject, Vector2D(-63.558, 44.65), Vector2D(-63.58, 44.65), 0.00019, 30),
    # DynamicVessel(4, ObjectType.DynamicObject, Vector2D(-63.558, 44.65), Vector2D(-63.58, 44.65), 0.00019, 15),
    # DynamicVessel(5, ObjectType.DynamicObject, Vector2D(-63.458, 44.62), Vector2D(-63.48, 44.62), 0.00019, 15),
    AISVessel(2, ObjectType.AISObject, 'ais/ais_ferry_simple.csv', 0.001, 60)
]

