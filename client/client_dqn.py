from server.Message import *
from client.dqn import *
from datetime import datetime
from server.setting import result_path, client_input_type, send_high_level_parameter, \
    use_trained_network, trained_network_path


now = datetime.now()
current_time = now.strftime("%Y-%m-%d-%H-%M-%S")

min_real_x = -63.675
min_real_y = 44.58
max_real_x = -62
max_real_y = 45
vel_real_x = 1
vel_real_y = 1
max_x = 20
max_y = 20
rl: DeepQ = None


def init_agent(message: MessageClientConnectResponse):
    global vel_real_x, vel_real_y, max_x, max_y, min_real_x, max_real_x, min_real_y, max_real_y, rl

    min_real_x = message.min_real_x
    max_real_x = message.max_real_x
    min_real_y = message.min_real_y
    max_real_y = message.max_real_y

    vel_real_x = message.vel[0]
    vel_real_y = message.vel[1]
    max_x = int((message.max_real_x - message.min_real_x) / vel_real_x) + 2
    max_y = int((message.max_real_y - message.min_real_y) / vel_real_y) + 2
    use_prioritize_reply_buffer = False
    rl = DeepQ(prb=use_prioritize_reply_buffer)
    if use_trained_network:
        rl.read_model(trained_network_path, client_input_type)
    else:
        if client_input_type == 'image':
            rl.create_model_cnn_dense()
        elif client_input_type == 'param':
            input_shape = (4,)
            if send_high_level_parameter:
                input_shape = (12,)
            rl.create_model_dense(input_shape)
        else:
            rl.create_model_dense_cnn_dense()
    print('rl model created')


def normalize(x, y):
    dif_x = max_real_x - min_real_x
    dif_y = max_real_y - min_real_y
    xx = (x - min_real_x) / dif_x
    yy = (y - min_real_y) / dif_y
    return xx, yy


number = 0
image_number = 0


def main(id_, info_pipe_agent, action_pipe_agent, input_agent=None, need_agent=None):
    global image_number
    message_snd = MessageClientConnectRequest(id_).build()
    action_pipe_agent.send(message_snd)
    while True:
        try:
            message_rcv = info_pipe_agent.recv()
            message = parse(message_rcv)
            init_agent(message)
            break
        except Exception as e:
            continue

    action = 0
    world = (0, 0)
    valid_number = 0
    valid_g_num = 0
    while True:

        try:
            message_rcv = info_pipe_agent.recv()
            message = parse(message_rcv)
        except Exception as e:
            print(e)
            continue
        if message.type == 'MessageClientDisconnect':
            rl.model.save(result_path + current_time + 'agent_dq.h5')
            exit()

        if message.type == 'MessageEndPlan':
            rl.plan_number += 1
            continue

        new_world = eval(message.world)

        agent_param = new_world['param']
        agent_high_param = new_world['hparam']
        agent_view = new_world['view']
        reward = message.reward

        if client_input_type == 'image':
            view = array(agent_view, dtype=np.float)
            view = view.reshape((51, 51, 1))
            new_world = view
        elif client_input_type == 'param':
            new_world = list(normalize(agent_param[0], agent_param[1])) + list(
                normalize(agent_param[2], agent_param[3]))
            new_world[2] = new_world[2] - new_world[0]
            new_world[3] = new_world[3] - new_world[1]
            if send_high_level_parameter:
                new_world += agent_high_param
            new_world = np.array(new_world)
        else:
            view = array(agent_view, dtype=np.float)
            view = view.reshape((51, 51, 1))
            param = list(normalize(agent_param[0], agent_param[1])) + list(
                normalize(agent_param[2], agent_param[3]))
            param[2] = param[2] - param[0]
            param[3] = param[3] - param[1]
            param = np.array(param)
            new_world = [view, param]

        # update
        if message.status is not 'first':
            if message.epoch_type is 'learn':
                if message.status is 'end':
                    rl.add_to_buffer(world, action, reward)
                else:
                    rl.add_to_buffer(world, action, reward, new_world)

        if message.epoch_type != 'learn':
            if valid_number == 0:
                rl.model.save(result_path + current_time + 'agent_dq' + str(valid_g_num) + '.h5')
                valid_g_num += 1
            valid_number += 1
        else:
            valid_number = 0
        image_number += 1
        # action
        if message.status is not 'end':
            world = copy.deepcopy(new_world)
            if need_agent is not None:
                need_agent.send('need')
                while True:
                    try:
                        act_rcv = input_agent.recv()
                        action = act_rcv
                        break
                    except Exception as e:
                        print('error cant receive action:', e)
                        continue
            else:
                if message.epoch_type == 'learn':
                    action = rl.get_random_action(world)
                else:
                    action = rl.get_random_action(world, 0.0)
            message_snd = MessageClientAction(action).build()
            action_pipe_agent.send(message_snd)
