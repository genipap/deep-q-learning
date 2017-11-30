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


class DQNAgent:

    # Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf_sess = tf.Session(config=config)
    keras.set_session(tf_sess)
    keras.set_learning_phase(1)

    def __init__(self, state_size, action_size, batch_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.tau = 0.0001
        self.epsilon_min = 0.01
        self.epsilon_decay = 1. / 100000.
        self.learning_rate = 0.001

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

        self.if_done = False

    def update(self, e):
        self.if_done = False
        if self.epsilon > self.epsilon_min:
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
        s0 = Dense(state_size, activation='linear')(S)
        a0 = Dense(action_size, activation='linear')(A)
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
            # q1 = self.target_model.predict([s_t, -np.ones([1, 1])])
            # q2 = self.target_model.predict([s_t, np.zeros([1, 1])])
            # q3 = self.target_model.predict([s_t, np.ones([1, 1])])
            # a_c = [-1., 0., 1.]
            # a_t = a_c[np.argmax([q1, q2, q3])]
            # if a_t == 1.:
            #     return a_t
            zz = if_train * max(self.epsilon, self.epsilon_min)
            # # if rule:
            if random() < zz:
                rr = random()
                if rr < 0.33:
                    a_t = 1.
                else:
                    a_t = 0. if (random() > 0.67) else -1.
                # left = s_t[0][15:25]
                # dis_l = left[::2]
                # dis_a_l = dis_l >= Safe_dis
                # dis_b_l = dis_l < 0.
                # disl_ = np.array([dis_a_l, dis_b_l])
                # t_l = left[1::2]
                # t_a_l = t_l >= Safe_time
                # t_b_l = t_l < 0.
                # tl_ = np.array([t_a_l, t_b_l])
                # right = s_t[0][25:]
                # dis_r = right[::2]
                # dis_a_r = dis_r >= Safe_dis
                # dis_b_r = dis_r < 0.
                # disr_ = np.array([dis_a_r, dis_b_r])
                # t_r = right[1::2]
                # t_a_r = t_r >= Safe_time
                # t_b_r = t_r < 0.
                # tr_ = np.array([t_a_r, t_b_r])
                # if np.any(disl_, axis=0).all() and np.any(tl_, axis=0).all():
                #     if s_t[0][5] > 0.05:
                #         a_t = 0. if (random() > 0.5) else -1.
                #     else:
                #         a_t = 0. if (random() > 0.5) else 1.
                # elif s_t[0][5] > 0.05:
                #     a_t = 0. if (random() > 0.5) else -1.
                # else:
                #     a_t = -1.
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

    def replay(self, state, action, r, next_state, do, b_size):
        Buffer.add(state, action, r, next_state, do)
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
            self.critic_model.load_weights("weights/criticmodel.h5")
            self.target_model.load_weights("weights/criticmodel.h5")
        except:
            logging.warn("Cannot find the weight !")

    def save(self):
        self.critic_model.save_weights('weights/criticmodel.h5')
        with open("weights/criticmodel.json", "w") as outfile:
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

if __name__ == "__main__":
    plt.ion()
    # sim = InterSim(randrange(4), True)
    sim = InterSim(1, False)
    reward = Reward()
    state_t = sim.get_state()
    state_size = state_t.shape[1]
    action_size = 1
    batch_size = 128
    agent = DQNAgent(state_size, action_size, batch_size)
    agent.load()
    done = False
    train_ind = True

    loss = []
    successes = []
    crashes = []
    not_moves = []
    not_finishes = []
    total_rewards = []
    tictac = time.time()
    # action_t = -1. if random() > 0.5 else 1.

    for e in range(EPISODES):
        step = 0
        total_reward = float(0.)
        mean_loss = float(0.)
        # zz = train_ind * max(agent.epsilon, agent.epsilon_min)
        # follow_rule = True if random() < zz else False
        while True:
            action_t = agent.act(state_t, train_ind)
            # action_t -= 1. if mean_loss < 0.001 else 0.
            # if len(loss) > 100 and (np.mean(loss[-100:]) < 0.01):
            #     action_t = agent.act(state_t, follow_rule, train_ind)
            # else:
            #     action_t = 1.
            action_t = np.array([action_t], ndmin=2)
            if action_t[0][0] != 1.:
                reward_t, collision_l, collision_r, not_move = reward.get_reward(state_t[0], action_t[0][0])
                sim.update_vehicle(action_t[0][0], reward_t)
                state_t1 = sim.get_state()
            else:
                while True:
                    reward_t, collision_l, collision_r, not_move = reward.get_reward(state_t[0], action_t[0][0])
                    sim.update_vehicle(action_t[0][0], reward_t)
                    state_t1 = sim.get_state()
                    if state_t[0][9] <= - state_t[0][2] or (collision_l > 0):
                        break
                    else:
                        state_t = state_t1
            total_reward += reward_t
            done, not_finish, not_move, crash, success = \
                agent.if_exit(step, state_t[0], collision_l, collision_r, not_move, sim.cond)
            if train_ind:
                mean_loss += (agent.replay(state_t, action_t[0], reward_t, state_t1, done, batch_size))
            if train_ind:
                agent.update_target_model()
            if done:
                break
            state_t = state_t1
            step += 1
        total_rewards.append(total_reward)
        if train_ind:
            loss.append(mean_loss / (step + 1.))
        plt.close('all')
        visual = False if (e + 1) % 500 == 0 else False
        # logging.debug('Episode: ' + str(e) + ', Step: ' + str(step) + ', Reward: ' + str(total_reward) +
        #               ', loss: {0:.2f}'.format(loss[-1]) + ', Success: ' + str(success))
        if train_ind:
            agent.save()
        if (e + 1) % 100 == 0:
            successes.append(success)
            crashes.append(crash)
            not_moves.append(not_move)
            not_finishes.append(not_finish)
            logging.info('Time: {0:.2f}'.format((time.time() - tictac) / 3600.) +
                         ', Crash: ' + str(crashes) + '\nNot Finished: ' + str(not_finishes) +
                         '\nNot Move: ' + str(not_moves) + '\nSuccess: ' + str(successes))
            results = {'crash': crashes, 'unfinished': not_finishes, 'stop': not_moves, 'succeess': successes,
                       'reward': total_rewards, 'loss': loss}
            with open('vehicle_random.txt', 'w+') as json_file:
                jsoned_data = json.dumps(results)
                json_file.write(jsoned_data)

            # train_ind = False if (train_ind is True) else True

        # if len(loss) > 100 and (np.mean(loss[-100:]) < 0.01):
        #     break
        # sim = InterSim(randrange(4), visual)
        sim = InterSim(1, visual)
        state_t = sim.get_state()
        agent.update(e)
