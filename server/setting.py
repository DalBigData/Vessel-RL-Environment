from server.math.geom_2d import *


class Halifax:
    def __init__(self):
        self.name = 'halifax'
        self.map_path = 'geopandas/halifax_shape/'
        self.map_crs = {'init': 'epsg:4326'}
        self.x_lim = [-63.69, -63.49]  # 0.2
        self.y_lim = [44.58, 44.73]  # 0.15
        self.geo_image_file = 'HalifaxBuffer50m2_1000t750round.image'
        self.shape_file = 'HalifaxBuffer50m2.shp'
        self.geo_image_resolution_x = 1000
        self.geo_image_resolution_y = int(
            (self.y_lim[1] - self.y_lim[0]) / (self.x_lim[1] - self.x_lim[0]) * self.geo_image_resolution_x)
        self.plan_resolution = [100, 75]
        self.planner_helper_points = [(-63.5959, 44.6313), (-63.609, 44.637), (-63.611, 44.6371), (-63.6006, 44.6328),
                                      (-63.5903, 44.6271), (-63.5878, 44.6267), (-63.5844, 44.6251), (-63.5796, 44.6232),
                                      (-63.5738, 44.619), (-63.611, 44.677)]


class Montreal:
    def __init__(self):
        self.name = 'montreal'
        self.map_path = 'geopandas/montreal_shape/'
        self.map_crs = {'init': 'epsg:4326'}
        self.x_lim = [-74.1, -73.7]  # 0.4
        self.y_lim = [45.30, 45.50]  # 0.2
        self.geo_image_file = 'MontrealBuffer50m2_2000t1000round.image'
        self.shape_file = 'QuebecLandBuffer50m.shp'
        self.geo_image_resolution_x = 2000
        self.geo_image_resolution_y = int(
            (self.y_lim[1] - self.y_lim[0]) / (self.x_lim[1] - self.x_lim[0]) * self.geo_image_resolution_x)
        self.plan_resolution = [100, 50]
        self.planner_helper_points = [(-74.0776, 45.3071), (-74.0738, 45.3058), (-74.0685, 45.3058), (-74.064, 45.3058),
                                      (-74.0596, 45.3056), (-74.0579, 45.3044), (-74.0761, 45.3018), (-74.072, 45.3018),
                                      (-74.0672, 45.3012), (-74.0441, 45.3023), (-74.0498, 45.3018), (-74.0467, 45.3021),
                                      (-74.04, 45.3028), (-74.0344, 45.3024), (-74.0276, 45.3103), (-74.0324, 45.3091),
                                      (-74.036, 45.308), (-74.0401, 45.3067), (-73.9696, 45.3183), (-73.9493, 45.3301),
                                      (-74.0024, 45.3869), (-73.8458, 45.3298), (-73.8491, 45.3293), (-73.8618, 45.3216),
                                      (-73.8573, 45.3267), (-73.8592, 45.329), (-73.9516, 45.4018), (-73.9679, 45.4061),
                                      (-73.8667, 45.3947), (-73.9363, 45.4746), (-73.9243, 45.4744), (-73.922, 45.4721),
                                      (-73.92, 45.4702), (-73.892, 45.4697)]


# Map Setting
city = Halifax()
city_name = city.name
map_path = city.map_path
shape_path = map_path + city.shape_file
map_crs = city.map_crs
x_lim = city.x_lim
y_lim = city.y_lim
geo_image_resolution_x = city.geo_image_resolution_x
geo_image_resolution_y = city.geo_image_resolution_y
geo_image_path = map_path + city.geo_image_file
plan_resolution = city.plan_resolution
planner_helper_points = city.planner_helper_points

# Episode or Plan Setting
train_episode_nb = 10000  # Number of train epoch
test_interval = 1000
test_episode_nb = 100  # Number of test epoch
max_step_in_episode = 100  # Number of episode of each epoch

use_plan: bool = True
train_plan_nb = 5000
test_plan_interval = 500
test_plan_nb = 100
max_episode_in_plan = 100

result_path = "../res1/"
result_prefix = ""

use_geo_image = True
random_target_out_local_view = True
max_target_dist_in_local_view = 24
max_dist_agent_to_new_target = min(0.02, max(x_lim[1] - x_lim[0], y_lim[1] - y_lim[0]))
max_dist_agent_to_target = 2 * max_dist_agent_to_new_target  # Vanish Target Distance

change_position_of_const_object = False

save_local_view = True
show_test = False
show_train = False

client_input_type = 'image'  # image param imageparam
send_high_level_parameter = False

from server.agent import Agent
agent_start_position = None
agent_radius = 0.00009
agent_velocity = Vector2D(0.001, 0.001)
target_position = None
target_radius = 0.001
agent_action_type = 'discrete'  # discrete or continues
continues_action_type = 'ar'  # xy or ar or complex
move_xy_max = [0.002, 0.002]  # used if discrete_action_type = 'xy'
move_r_max = 0.002
agent = Agent(id=0,
              start_pos=agent_start_position,
              r=agent_radius,
              target_pos=target_position,
              target_r=target_radius,
              vel=agent_velocity,
              agent_action_type=agent_action_type,
              continues_action_type=continues_action_type,
              move_xy_max=move_xy_max,
              move_r_max=move_r_max
              )

# client
use_keyboard = False
use_trained_network = False
trained_network_path = ''
