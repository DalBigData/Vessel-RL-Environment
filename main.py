from multiprocessing import Process
from multiprocessing import Queue
from multiprocessing import Pipe
from server.server_ import main as server_main
from client.client_dqn import main as agent_main
from server.setting import use_keyboard, continues_action_type, agent_action_type
import time


t1 = time.time()

action_pipe_server, action_pipe_agent = Pipe()
info_pipe_agent, info_pipe_server = Pipe()

input_agent = None
need_agent = None
if use_keyboard:
    input_agent, input_main = Pipe()
    need_main, need_agent = Pipe()

server = Process(target=server_main, args=(info_pipe_server, action_pipe_server,))
agent = Process(target=agent_main, args=(0, info_pipe_agent, action_pipe_agent, input_agent, need_agent))

server.start()
time.sleep(1)
agent.start()

if use_keyboard:
    actions_list = []
    in_action = True
    while True:
        try:
            if in_action:
                x = need_main.recv()
                print('receive from client:', x)
                in_action = False

            if len(actions_list) == 0:
                if agent_action_type == 'continues':
                    if continues_action_type == 'xy':
                        actions = input(
                            'enter action or actions like \'-0.001 0.002\' or \'0.001 0.001 0.001 -0.001\':')
                    elif continues_action_type == 'ar':
                        actions = input(
                            'enter action or actions like \'92 0.002\' or \'-175 0.001 80 -0.001\':')
                    if len(actions) > 0:
                        actions = actions.split(' ')
                        for a in range(len(actions))[::2]:
                            try:
                                actions_list.append([float(actions[a]), float(actions[a + 1])])
                            except:
                                continue
                else:
                    actions = input('enter action or actions like \'2\' or \'1 2 2 3\':')
                    if len(actions) > 0:
                        actions = actions.split(' ')
                        for a in actions:
                            try:
                                actions_list.append(int(a) - 1)
                            except:
                                continue
            if len(actions_list) > 0:
                input_main.send(actions_list[0])
                del actions_list[0]
                in_action = True
        except Exception as e:
            print(e)
            continue

agent.join()
server.join()


t2 = time.time()
print('run_time:', t2 - t1)