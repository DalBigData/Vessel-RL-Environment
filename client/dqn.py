from keras import models, layers, activations, optimizers, losses, metrics, regularizers
from keras.engine.sequential import Sequential
import random
from numpy import array
from typing import List
import sys
import copy
import keras
from keras import Model
from keras.layers.merge import concatenate
import numpy as np
from server.setting import train_episode_nb, train_plan_nb, use_plan, result_path
from client.reply_buffer_simple import Buffer
from client.replay_buffer_pri import ReplayBuffer
from client.transit import Transition


class DeepQ:
    def __init__(self, buffer_size=100000, train_interval_step=100, target_update_interval_step=200, prb=False):
        self.model_type = ''
        self.model = None
        self.target_network = None
        self.prb = prb
        self.buffer = Buffer(buffer_size) if not prb else ReplayBuffer(buffer_size)
        self.train_interval_step = train_interval_step
        self.target_update_interval_step = target_update_interval_step
        self.action_number = 9
        self.transitions: List[Transition] = []
        self.gama = 0.95
        self.episode_number = 0
        self.plan_number = 0
        self.step_number = 0
        self.use_double = False
        self.loss_values = []
        pass

    def create_model_cnn_dense(self):
        self.model_type = 'image'
        in1 = layers.Input((51, 51, 1,))
        m1 = layers.Conv2D(32, (4, 4), strides=(2, 2), activation='relu', input_shape=(51, 51, 1))(in1)
        m1 = layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(m1)
        m1 = layers.Conv2D(64, (3, 3), strides=(2, 2), activation='relu')(m1)
        m1 = layers.Conv2D(64, (2, 2), strides=(1, 1), activation='relu')(m1)
        m1 = layers.Flatten()(m1)
        conv_model = keras.Model(in1, m1)

        out = layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.001),
                           bias_regularizer=regularizers.l2(0.001))(m1)
        out = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001),
                           bias_regularizer=regularizers.l2(0.001))(out)
        out = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001),
                           bias_regularizer=regularizers.l2(0.001))(out)
        out = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001),
                           bias_regularizer=regularizers.l2(0.001))(out)
        out = layers.Dense(9)(out)

        model = keras.Model(in1, out)
        model.compile(optimizer=optimizers.RMSprop(lr=0.00025, rho=0.95), loss=losses.mse, metrics=[metrics.mse])
        model.summary()
        self.model = model
        self.target_network = keras.models.clone_model(self.model)

    def create_model_resnet(self):
        self.model_type = 'image'
        reznet = keras.applications.resnet.ResNet50(include_top=False, weights=None, input_tensor=None,
                                                    input_shape=(51, 51, 1), pooling=None, classes=1000)
        out = reznet.get_layer('conv5_block3_add').output
        out = keras.layers.Flatten()(out)
        out = keras.layers.Dense(1024, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001),
                                 bias_regularizer=keras.regularizers.l2(0.001))(out)
        out = keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001),
                                 bias_regularizer=keras.regularizers.l2(0.001))(out)
        out = keras.layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001),
                                 bias_regularizer=keras.regularizers.l2(0.001))(out)
        out = keras.layers.Dense(9)(out)
        model = keras.Model(reznet.input, out)
        model.summary()
        model.compile(optimizer=optimizers.sgd(), loss=losses.mse, metrics=[metrics.mse])
        self.model = model
        self.target_network = keras.models.clone_model(self.model)

    def create_model_dense(self, input_shape=(4,)):
        self.model_type = 'param'
        model = models.Sequential()
        model.add(layers.Dense(100, activation=activations.relu, input_shape=input_shape))
        model.add(layers.Dense(60, activation=activations.relu))
        model.add(layers.Dense(30, activation=activations.relu))
        model.add(layers.Dense(9, activation=activations.linear))
        model.compile(optimizer=optimizers.RMSprop(lr=0.00025, rho=0.95), loss=losses.mse, metrics=[metrics.mse])
        model.summary()
        self.model = model
        self.target_network = keras.models.clone_model(self.model)

    def create_model_dense_cnn_dense(self) -> Sequential:
        self.model_type = 'imageparam'
        in1 = layers.Input((51, 51, 1,))
        m1 = layers.Conv2D(32, (4, 4), strides=(2, 2), activation='relu', input_shape=(51, 51, 1))(in1)
        m1 = layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(m1)
        m1 = layers.Conv2D(64, (3, 3), strides=(2, 2), activation='relu')(m1)
        m1 = layers.Conv2D(64, (2, 2), strides=(1, 1), activation='relu')(m1)
        m1 = layers.Flatten()(m1)
        model1 = keras.Model(in1, m1)
        model1.summary()

        in2 = keras.Input((4,))
        print('a')
        m2 = layers.Dense(4)(in2)
        print('a')
        model2 = keras.Model(in2, m2)
        print('a')
        model2.summary()

        concatenated = concatenate([m1, m2])

        out = layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.001),
                           bias_regularizer=regularizers.l2(0.001))(concatenated)
        out = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001),
                           bias_regularizer=regularizers.l2(0.001))(out)
        out = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001),
                           bias_regularizer=regularizers.l2(0.001))(out)
        out = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001),
                           bias_regularizer=regularizers.l2(0.001))(out)
        out = layers.Dense(9)(out)

        model = keras.Model([in1, in2], out)
        model.compile(optimizer=optimizers.RMSprop(lr=0.00025, rho=0.95), loss=losses.mse, metrics=[metrics.mse])
        model.summary()
        self.model = model
        self.target_network = keras.models.clone_model(self.model)
        return model

    def create_model_from_cnn_model(self, path):
        cnn_model = keras.models.load_model(path)
        # for layer in cnn_model.layers:
        #     layer.trainable = False
        m1 = cnn_model.get_layer('flatten_1').output
        m1 = layers.Dense(512, activation='relu')(m1)
        m1 = layers.Dense(32, activation='tanh')(m1)
        m1 = layers.Dense(16, activation='relu')(m1)
        m1 = layers.Dense(9)(m1)
        model = Model(cnn_model.input, m1)
        model.compile(optimizer=optimizers.RMSprop(lr=0.00025, rho=0.95), loss=losses.mse, metrics=[metrics.mse])
        self.model = model
        self.target_network = keras.models.clone_model(self.model)
        print(model.summary())

    def read_model(self, path, model_type):
        self.model_type = model_type
        self.model = keras.models.load_model(path)
        self.model.compile(optimizer=optimizers.RMSprop(lr=0.00025, rho=0.95), loss=losses.mse, metrics=[metrics.mse])
        self.target_network = keras.models.clone_model(self.model)
        print(self.model.summary())

    def get_q(self, transits: List[Transition]):
        x = []
        for t in transits:
            x.append(t.state)
        x = array(x)
        y = self.model.predict(x)
        return y

    def get_best_action(self, state):
        if self.model_type == 'image':
            if state.ndim == 3:
                state = state.reshape((1, 51, 51, 1))
        elif self.model_type == 'param':
            if state.ndim == 1:
                state = state.reshape((1, state.shape[0]))
        else:
            state_view = state[0]
            if state_view.ndim == 3:
                state_view = state_view.reshape((1, 51, 51, 1))
            state_param = state[1]
            if state_param.ndim == 1:
                state_param = state_param.reshape((1, 4))
            state = [state_view, state_param]

        Y = self.model.predict(state)
        actions = np.argmax(Y, axis=1)
        max_qs = np.max(Y, axis=1).flatten()

        return actions, max_qs

    def get_random_action(self, state, p_rnd=None):
        if p_rnd is None:
            if use_plan:
                max_number = train_plan_nb
                number = self.plan_number
            else:
                max_number = train_episode_nb
                number = self.episode_number
            max_number = max_number / 2
            max_number = min(max_number, 25000)
            if number > max_number:
                p_rnd = 0.1
            else:
                p_rnd = 1.0 + number / max_number * -0.9
        if random.random() < p_rnd:
            return random.randrange(self.action_number)
        best_action, best_q = self.get_best_action(state)
        return best_action[0]

    @staticmethod
    def rotate_action(ac):
        action_rot = [2, 5, 8, 1, 4, 7, 0, 3, 6]
        return action_rot[ac]

    def add_to_buffer(self, state, action, reward, next_state=None):
        for i in range(4):
            transition = Transition(state, action, reward, next_state)
            if self.prb:
                self.buffer.add(transition, 1)
            else:
                self.buffer.add(transition)
            if self.model_type != 'image':
                break
            state = np.rot90(state)
            if next_state is not None:
                next_state = np.rot90(next_state)
            action = DeepQ.rotate_action(action)

        self.step_number += 1
        if self.step_number % self.train_interval_step == 0:
            self.update_from_buffer()
        if self.step_number % self.target_update_interval_step == 0:
            self.target_network.set_weights(self.model.get_weights())
        if next_state is None:  # End step in episode
            self.episode_number += 1

    def update_from_buffer_pre(self):
        indexes, weights = self.buffer.sampling_data_prioritized(3200)
        if len(indexes) == 0:
            return
        transits = self.buffer.sample_index(indexes)
        states_view = []
        next_states_view = []
        for t in transits:
            states_view.append(t.state)
            if t.is_end:
                next_states_view.append(t.state)
            else:
                next_states_view.append(t.next_state)

        states_view = array(states_view)

        next_states_view = array(next_states_view)

        q = self.model.predict(states_view)
        best_q_action = np.argmax(q, axis=1)
        next_q = self.target_network.predict(next_states_view)

        if self.use_double:
            next_states_max_q = []
            for i in best_q_action:
                next_states_max_q.append(next_q[len(next_states_max_q)][i])
            next_states_max_q = np.array(next_states_max_q)
        else:
            next_states_max_q = np.max(next_q, axis=1).flatten()

        new_prioritize = []
        for i in range(len(transits)):
            q_learning = transits[i].reward
            if not transits[i].is_end:
                q_learning += (self.gama * next_states_max_q[i])
            lost = q_learning - q[i][transits[i].action]
            new_prioritize.append(abs(lost))
            lost *= weights[i]
            # new_prioritize.append(abs(lost))
            q[i][transits[i].action] += lost

        history = self.model.fit(states_view, q, epochs=1, batch_size=32, verbose=1)
        history_dict = history.history
        new_prioritize = np.array(new_prioritize)
        self.buffer.update_priority(indexes, new_prioritize)

    def update_from_buffer(self):
        if self.prb:
            self.update_from_buffer_pre()
            return
        transits: List[Transition] = self.buffer.get_rand(3000)
        if len(transits) == 0:
            return
        is_image_param = False
        if self.model_type == 'imageparam':
            is_image_param = True

        print('buffer size:', self.buffer.i)
        states_view = []
        next_states_view = []
        states_param = []
        next_states_param = []
        for t in transits:
            if is_image_param:
                states_view.append(t.state[0])
                states_param.append(t.state[1])
                if t.is_end:
                    next_states_view.append(t.state[0])
                    next_states_param.append(t.state[1])
                else:
                    next_states_view.append(t.next_state[0])
                    next_states_param.append(t.next_state[1])
            else:
                states_view.append(t.state)
                if t.is_end:
                    next_states_view.append(t.state)
                else:
                    next_states_view.append(t.next_state)

        if is_image_param:
            states_view = array(states_view)
            next_states_view = array(next_states_view)
            states_param = array(states_param)
            next_states_param = array(next_states_param)
        else:
            states_view = array(states_view)
            next_states_view = array(next_states_view)

        if is_image_param:
            states_view = [states_view, states_param]
            next_states_view = [next_states_view, next_states_param]
        q = self.model.predict(states_view)
        best_q_action = np.argmax(q, axis=1)
        next_q = self.target_network.predict(next_states_view)

        if self.use_double:
            next_states_max_q = []
            for i in best_q_action:
                next_states_max_q.append(next_q[len(next_states_max_q)][i])
            next_states_max_q = np.array(next_states_max_q)
        else:
            next_states_max_q = np.max(next_q, axis=1).flatten()

        for i in range(len(transits)):
            q_learning = transits[i].reward
            if not transits[i].is_end:
                q_learning += (self.gama * next_states_max_q[i])
            diff = (q_learning - q[i][transits[i].action]) * transits[i].value
            q[i][transits[i].action] += diff

        history = self.model.fit(states_view, q,  epochs=1, batch_size=32, verbose=1)
        history_dict = history.history

        loss_values = history_dict['loss']
        self.loss_values.append(loss_values[0])
        if len(self.loss_values) == 10:
            f = open(result_path + 'agent_loss', 'a')
            for l in self.loss_values:
                f.write(str(l) + '\n')
            self.loss_values = []


if __name__ == "__main__":
    dq = DeepQ()
    a = dq.get_random_action([1, 2], 0)
    print(a)