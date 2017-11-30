#!/usr/bin/env python
import sys
if "../" not in sys.path:               # Path to utilities and other custom modules
    sys.path.append("../")
import logging
import tensorflow as tf
from inter_hrl import InterSim
import json
import time
import matplotlib.pyplot as plt
from reward import Reward
from replay import Replay
from random import random, randrange, randint
import numpy as np
from collections import deque
from keras.models import Model, Sequential
from keras.layers import Dense, Input, concatenate
from keras.optimizers import Adam
from keras import backend as keras
import log_color

__author__ = 'qzq'


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
EPISODES = 1000000
MAX_STEP = 300
Buffer_size = 100000
Safe_dis = 50.
Safe_time = 3.
Buffer = Replay(Buffer_size)
Step_size = 500
Batch_size = 128


class DQNAgent:

    # Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf_sess = tf.Session(config=config)
    keras.set_session(tf_sess)
    keras.set_learning_phase(1)

    def __init__(self, a_size, pos):
        self.sim = InterSim(1, pos, False)
        # self.buffer = Replay(Buffer_size)
        self.state = self.sim.get_state()
        self.state_size = self.state.shape[1]
        self.action_size = a_size
        self.gamma = 0.99    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.tau = 0.0001
        self.epsilon_min = 0.01
        self.epsilon_decay = 1. / 100000.
        self.learning_rate = 0.001
        self.init = pos

        self.reward = Reward()

        self.critic_model = self._build_model()
        self.target_model = self._build_model()
        self.tf_sess.run(tf.global_variables_initializer())
        # self.target_model.set_weights(self.critic_model.get_weights())

        self.batch = None
        self.batch_state = None
        self.batch_action = None
        self.batch_reward = None
        self.batch_new_state = None
        self.batch_if_done = None
        self.batch_output = None

        self.sub_crash = 0
        self.sub_success = 0
        self.sub_not_finish = 0
        self.sub_not_move = 0

        self.successes = []
        self.crashes = []
        self.not_moves = []
        self.not_finishes = []
        self.total_rewards = []
        self.loss = []

        self.if_done = False

    def update(self, e, B=False):
        self.if_done = False
        if self.epsilon > self.epsilon_min and B:
            self.epsilon -= self.epsilon_decay
        if (e + 1) % 100 == 0:
            self.sub_crash = 0
            self.sub_success = 0
            self.sub_not_finish = 0
            self.sub_not_move = 0

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        # model = Sequential()
        # model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        # model.add(Dense(24, activation='relu'))
        # model.add(Dense(self.action_size, activation='linear'))
        # model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        S = Input(shape=[self.state_size])
        A = Input(shape=[self.action_size])
        s0 = Dense(self.state_size, activation='linear')(S)
        a0 = Dense(self.action_size, activation='linear')(A)
        h0 = concatenate([s0, a0])
        # h1 = Dense(128, activation='relu')(h0)
        h2 = Dense(63, activation='relu')(h0)
        h3 = Dense(32, activation='relu')(h2)
        V = Dense(1, activation='linear')(h3)
        model = Model(input=[S, A], output=V)
        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mse', optimizer=adam)
        return model

    def update_target_model(self):
        critic_weights = self.critic_model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in range(len(critic_weights)):
            critic_target_weights[i] = self.tau * critic_weights[i] + (1 - self.tau) * critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)
        # self.target_model.set_weights(self.critic_model.get_weights())

    def act(self, s_t, if_train):
        if if_train:
            zz = if_train * max(self.epsilon, self.epsilon_min)
            if random() < zz:
                #     rr = random()
                #     if rr < 0.33:
                #         a_t = 1.
                #     else:
                #         a_t = 0. if (random() > 0.67) else -1.
                left = s_t[0][15:17]
                dis_l = left[::2]
                dis_a_l = dis_l >= Safe_dis
                dis_b_l = dis_l < 0.
                disl_ = np.array([dis_a_l, dis_b_l])
                t_l = left[1::2]
                t_a_l = t_l >= Safe_time
                t_b_l = t_l < 0.
                tl_ = np.array([t_a_l, t_b_l])
                # right = s_t[0][17:]
                # dis_r = right[::2]
                # dis_a_r = dis_r >= Safe_dis
                # dis_b_r = dis_r < 0.
                # disr_ = np.array([dis_a_r, dis_b_r])
                # t_r = right[1::2]
                # t_a_r = t_r >= Safe_time
                # t_b_r = t_r < 0.
                # tr_ = np.array([t_a_r, t_b_r])
                if np.any(disl_, axis=0).all() and np.any(tl_, axis=0).all():
                    if s_t[0][5] > 0.05:
                        a_t = 0. # if (random() > 0.2) else 1.
                    else:
                        a_t = 0. if (random() > 0.5) else 1.
                elif s_t[0][5] > 0.05:
                    a_t = 0. if (random() > 0.5) else -1.
                elif s_t[0][5] < 0.:
                    a_t = 1.
                else:
                    a_t = -1.
            # elif random() < zz:
            #     rr = random()
            #     if rr < 0.33:
            #         a_t = 1.
            #     else:
            #         a_t = 0. if (random() > 0.67) else -1.
            else:
                q1 = self.target_model.predict([s_t, -np.ones([1, 1])])
                q2 = self.target_model.predict([s_t, np.zeros([1, 1])])
                q3 = self.target_model.predict([s_t, np.ones([1, 1])])
                a_c = [-1., 0., 1.]
                a_t = a_c[np.argmax([q1, q2, q3])]
        else:
            q1 = self.target_model.predict([s_t, -np.ones([1, 1])])
            q2 = self.target_model.predict([s_t, np.zeros([1, 1])])
            q3 = self.target_model.predict([s_t, np.ones([1, 1])])
            a_c = [-1., 0., 1.]
            a_t = a_c[np.argmax([q1, q2, q3])]
        return a_t

    def replay(self, state, action, r, next_state, do, b_size, update_b=False):
        Buffer.add(state, action, r, next_state, do)
        # if update_b:
            # self.buffer.add(state, action, r, next_state, do)
            # self.batch = self.buffer.get_batch(b_size)
        # else:
        self.batch = Buffer.get_batch(b_size)
        self.batch_state = np.squeeze(np.asarray([x[0] for x in self.batch]), axis=1)
        self.batch_action = np.asarray([x[1] for x in self.batch])
        self.batch_reward = np.asarray([x[2] for x in self.batch])
        self.batch_new_state = np.squeeze(np.asarray([x[3] for x in self.batch]), axis=1)
        self.batch_if_done = np.asarray([x[4] for x in self.batch])
        # self.batch_output = np.asarray([e[2] for e in self.batch])

        q1 = self.target_model.predict([self.batch_new_state, -np.ones([self.batch_new_state.shape[0], 1])])
        q2 = self.target_model.predict([self.batch_new_state, np.zeros([self.batch_new_state.shape[0], 1])])
        q3 = self.target_model.predict([self.batch_new_state, np.ones([self.batch_new_state.shape[0], 1])])
        target_q = np.array(np.max(np.array([q1[:, 0], q2[:, 0], q3[:, 0]]), axis=0), ndmin=2).T
        # q1 = self.target_model.predict(self.batch_new_state)
        # a = np.argmax(q, axis=1)
        # target_q = self.target_model.predict(self.batch_state)
        self.batch_output = q1
        for k, d in enumerate(self.batch_if_done):
            # target_q = q[k][a[k]]
            self.batch_output[k] = self.batch_reward[k] if d else self.batch_reward[k] + self.gamma * target_q[k]
            # self.critic_model.fit(self.batch_state[], self.batch_output[k], epochs=1, verbose=0)
        # loss = self.critic_model.train_on_batch(self.batch_state, self.batch_output)
        # minibatch = random.sample(self.memory, batch_size)
        # for state, action, reward, next_state, done in minibatch:
        #     target = self.critic_model.predict(state)
        #     if done:
        #         target[0][action] = reward
        #     else:
        #         a = self.critic_model.predict(next_state)[0]
        #         t = self.target_model.predict(next_state)[0]
        #         target[0][action] = reward + self.gamma * t[np.argmax(a)]
        return float(self.critic_model.train_on_batch([self.batch_state, self.batch_action], self.batch_output))

    def load(self):
        try:
            self.critic_model.load_weights('weights/criticmodel.h5')
            self.target_model.load_weights('weights/criticmodel.h5')
        except:
            logging.warn("Cannot find the weight !")

    def save(self):
        self.critic_model.save_weights('s' + str(self.init) + '/criticmodel.h5')
        with open('s' + str(self.init) + '/criticmodel.json', "w") as outfile:
            json.dump(self.critic_model.to_json(), outfile)

    def if_exit(self, s, state, c_l, c_r, nmove, cond):
        if s >= MAX_STEP:
            # logging.warn('Not finished with max steps! Dis to SL: {0:.2f}'.format(state[4]) +
            #              ', Velocity: {0:.2f}'.format(state[0]) + ', ' + cond)
            self.sub_not_finish += 1
            self.if_done = True
        elif nmove > 0:
            # logging.warn('Not move! Dis to SL: {0:.2f}'.format(state[4]) + ', Dis to Center: {0:.2f}'.format(state[6]) +
            #              ', Dis to hv: [{0:.2f}, {1:.2f}]'.format(state[-8], state[-2]) +
            #              ', Velocity: {0:.2f}'.format(state[0]) + ', ' + cond)
            self.sub_not_move += 1
            self.if_done = True
        elif c_l > 0 or (c_r > 0):
            if c_l > 0:
                v = 'left'
            elif c_r > 0:
                v = 'right'
            else:
                v = 'front'
            # logging.warn('Crash to ' + v + ' vehicles! Dis to SL: {0:.2f}'.format(state[4]) +
            #              ', Dis to Center: {0:.2f}'.format(state[6]) +
            #              ', Dis to hv: [{0:.2f}, {1:.2f}]'.format(state[-8], state[-2]) +
            #              ', Velocity: {0:.2f}'.format(state[0]) + ', ' + cond)
            self.sub_crash += 1
            self.if_done = True
        elif state[9] <= - state[2]:
            # logging.info('Congratulations! Traverse successfully. ' + cond)
            self.sub_success += 1
            self.if_done = True
        return self.if_done, self.sub_not_finish, self.sub_not_move, self.sub_crash, self.sub_success

    def train(self, B=False, train_ind=False):
        for e in range(Step_size):
            step = 0
            total_reward = float(0.)
            mean_loss = float(0.)
            while True:
                action_t = self.act(self.state, train_ind)
                action_t = np.array([action_t], ndmin=2)
                if action_t[0][0] != 1.:
                    reward_t, collision_l, collision_r, not_move = \
                        self.reward.get_reward(self.state[0], action_t[0][0])
                    self.sim.update_vehicle(action_t[0][0], reward_t)
                    state_t1 = self.sim.get_state()
                else:
                    while True:
                        reward_t, collision_l, collision_r, not_move = \
                            self.reward.get_reward(self.state[0], action_t[0][0])
                        self.sim.update_vehicle(action_t[0][0], reward_t)
                        state_t1 = self.sim.get_state()
                        if self.state[0][9] <= - self.state[0][2] or (collision_l > 0):
                            break
                        else:
                            self.state = state_t1
                total_reward += reward_t
                done, not_finish, not_move, crash, success = \
                    self.if_exit(step, self.state[0], collision_l, collision_r, not_move, self.sim.cond)
                if train_ind:
                    mean_loss += (self.replay(self.state, action_t[0], reward_t, state_t1, done, Batch_size, B))
                if train_ind:
                    self.update_target_model()
                if done:
                    break
                self.state = state_t1
                step += 1
            self.total_rewards.append(total_reward)
            if train_ind:
                self.loss.append(mean_loss / (step + 1.))
            plt.close('all')
            visual = False if (e + 1) % 500 == 0 else False
            # logging.debug('Episode: ' + str(e) + ', Step: ' + str(step) + ', Reward: ' + str(total_reward) +
            #               ', loss: {0:.2f}'.format(self.loss[-1]) + ', Success: ' + str(success))
            if train_ind:
                self.save()
            if (e + 1) % 100 == 0:
                self.successes.append(success)
                self.crashes.append(crash)
                self.not_moves.append(not_move)
                self.not_finishes.append(not_finish)
                # logging.info('Time: {0:.2f}'.format((time.time() - tictac) / 3600.) +
                #              ', Crash: ' + str(self.crashes) + '\nNot Finished: ' + str(self.not_finishes) +
                #              '\nNot Move: ' + str(self.not_moves) + '\nSuccess: ' + str(self.successes))
                results = {'crash': self.crashes, 'unfinished': self.not_finishes, 'stop': self.not_moves,
                           'succeess': self.successes, 'reward': self.total_rewards, 'loss': self.loss}
                with open('s' + str(self.init) + '/result.txt', 'w+') as json_file:
                    jsoned_data = json.dumps(results)
                    json_file.write(jsoned_data)
                    # train_ind = False if (train_ind is True) else True

            # sim = InterSim(randrange(4), visual)
            self.if_done = False
            self.sim = InterSim(1, self.init, visual)
            self.state = self.sim.get_state()
            self.update(e, B)

if __name__ == "__main__":
    plt.ion()
    final_model = None
    action_size = 1

    init_pos = [2, 1, 0, -1, -2, -3]
    agent = []
    q = []
    tictac = time.time()
    train_pro = []
    for i in init_pos:
        tmp_agent = DQNAgent(action_size, i)
        tmp_agent.load()
        tmp_agent.train(True, True)
        # if sum(tmp_agent.successes[-(Step_size / 100):]) / (Step_size / 10.) <= 9.:
        q.append(float(np.exp(sum(tmp_agent.successes[-(Step_size / 100):]) / (Step_size / 10.))))
        # else:
        #     q.append(float(np.exp(-10.)))
        agent.append(tmp_agent)
        logging.info('Time: {0:.2f}'.format((time.time() - tictac) / 3600.) + ', cond: ' + str(tmp_agent.sim.cond) +
                     ', Success: ' + str(tmp_agent.successes))

    while True:
        q_p = np.array(q) / (sum(q))
        train_pro.append(q)
        with open('train_pro4.txt', 'w+') as json_file:
            jsoned_data = json.dumps(train_pro)
            json_file.write(jsoned_data)

        boltz_rand = random()
        if boltz_rand < q_p[0]:
            next_ind = 0
        elif q_p[0] <= boltz_rand < sum(q_p[0:2]):
            next_ind = 1
        elif sum(q_p[0:2]) <= boltz_rand < sum(q_p[0:3]):
            next_ind = 2
        elif sum(q_p[0:3]) <= boltz_rand < sum(q_p[0:4]):
            next_ind = 3
        elif sum(q_p[0:4]) <= boltz_rand < sum(q_p[0:5]):
            next_ind = 4
        else:
            next_ind = 5
        strFormat = len(q_p) * '{:2.3f} '
        logging.debug('[' + strFormat.format(*q_p) + '], ' + 'Next ind: ' + str(next_ind))

        tmp_agent = agent[next_ind]
        tmp_agent.critic_model.save_weights('weights/criticmodel.h5')
        with open('weights/criticmodel.json', "w") as outfile:
            json.dump(tmp_agent.critic_model.to_json(), outfile)
        q = []
        for k, i in enumerate(init_pos):
            # logging.debug(str(k) + ', ' + str(i))
            tmp_agent = agent[k]
            tmp_agent.load()
            if k == next_ind:
                tmp_agent.train(True, True)
            else:
                tmp_agent.train()
            # q.append(float(np.exp(improve)))
            if sum(tmp_agent.successes[-(Step_size / 50):]) / (Step_size / 5.) <= 8.:
                improve = (sum(tmp_agent.successes[-(Step_size / 100):]) -
                           sum(tmp_agent.successes[-2 * (Step_size / 100):-(Step_size / 100)])) / (Step_size / 100.)
                q.append(float(np.exp(abs(improve))))
                # q.append(float(np.exp(sum(tmp_agent.successes[-(Step_size / 100):]) / (Step_size / 10.))))
                # q[next_ind] = float(np.exp(sum(tmp_agent.successes[-(Step_size / 100):]) / (Step_size / 10.)))
            else:
                q.append(float(np.exp(-10.)))
                # q[next_ind] = float(np.exp(-10.))
            agent[k] = tmp_agent
            logging.info('Time: {0:.2f}'.format((time.time() - tictac) / 3600.) +
                         ', cond: ' + str(tmp_agent.sim.cond) + ', Success: ' + str(tmp_agent.successes))

