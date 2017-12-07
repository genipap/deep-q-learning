#!/usr/bin/env python
import sys
if "../" not in sys.path:               # Path to utilities and other custom modules
    sys.path.append("../")
import logging
import numpy as np
import tensorflow as tf
import json
from approach_network.app_actor_net import AppActorNetwork
from approach_network.app_critic_net import AppCriticNetwork
from approach_network.app_replay import AppReplay
from utilities.toolfunc import ToolFunc
from keras import backend as keras
from inter_sim import InterSim
from reward_app import AppReward
import time
import matplotlib.pyplot as plt
from random import random
import log_color

__author__ = 'qzq'

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

Step_size = 500
Buffer = AppReplay(100000)


class ReinAcc(object):
    tools = ToolFunc()

    Tau = 1. / 30
    gamma = 0.99

    buffer_size = 10000
    batch_size = 128
    tau = 0.0001            # Target Network HyperParameters
    LRA = 0.001             # Learning rate for Actor
    LRC = 0.001             # Learning rate for Critic

    explore_iter = 100000.
    # explore_iter = 1000.
    episode_count = 500
    max_steps = 2000
    action_dim = 1          # Steering/Acceleration/Brake
    action_size = 1
    state_dim = 10

    # Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf_sess = tf.Session(config=config)
    keras.set_session(tf_sess)

    Speed_limit = 12

    def __init__(self, ini_pos, task_k):
        self.epsilon = 1.
        self.task = task_k

        self.sim = InterSim(ini_pos, False)
        self.reward = AppReward()
        self.total_reward = 0
        self.if_pass = False
        self.if_done = False

        self.crash = []
        self.success = []
        self.not_finish = []
        self.overspeed = []
        self.not_move = []
        self.cannot_stop = []
        self.loss = []
        self.run_time = []
        self.if_train = []

        self.total_loss = 0.
        self.total_rewards = []

        self.sub_crash = 0
        self.sub_success = 0
        self.sub_not_finish = 0
        self.sub_overspeed = 0
        self.sub_not_move = 0
        self.sub_cannot_stop = 0

        self.app_actor = None
        self.app_critic = None
        # self.buffer = AppReplay()

        self.batch = None
        self.batch_state = None
        self.batch_action = None
        self.batch_reward = None
        self.batch_new_state = None
        self.batch_if_done = None
        self.batch_output = None

        self.start_time = time.time()
        self.end_time = time.time()
        self.total_time = time.time()

    def load_weights(self):
        # logging.info('...... Loading weight ......')
        try:
            self.app_actor.model.load_weights("weights/actormodel.h5")
            self.app_critic.model.load_weights("weights/criticmodel.h5")
            self.app_actor.target_model.load_weights("weights/actormodel.h5")
            self.app_critic.target_model.load_weights("weights/criticmodel.h5")
            # logging.info("Weight load successfully")
        except:
            logging.warn("Cannot find the weight !")

    def update_weights(self):
        # logging.info('...... Updating weight ......')
        self.app_actor.model.save_weights("task" + str(self.task) + "/actormodel.h5", overwrite=True)
        with open("task" + str(self.task) + "/actormodel.json", "w") as outfile:
            json.dump(self.app_actor.model.to_json(), outfile)
        self.app_critic.model.save_weights("task" + str(self.task) + "/criticmodel.h5", overwrite=True)
        with open("task" + str(self.task) + "/criticmodel.json", "w") as outfile:
            json.dump(self.app_critic.model.to_json(), outfile)

    def update_batch(self, s, a, r, s1):
        # logging.info('...... Updating batch ......')
        Buffer.add(s, a, r, s1, self.if_done)
        self.batch = Buffer.get_batch(self.batch_size)
        self.batch_state = np.squeeze(np.asarray([e[0] for e in self.batch]), axis=1)
        self.batch_action = np.asarray([e[1] for e in self.batch])
        self.batch_reward = np.asarray([e[2] for e in self.batch])
        self.batch_new_state = np.squeeze(np.asarray([e[3] for e in self.batch]), axis=1)
        self.batch_if_done = np.asarray([e[4] for e in self.batch])
        self.batch_output = np.asarray([e[2] for e in self.batch])
        target_q_values = self.app_critic.target_model.predict(
            [self.batch_new_state, self.app_actor.target_model.predict(self.batch_new_state)])
        for k, done in enumerate(self.batch_if_done):
            self.batch_output[k] = self.batch_reward[k] if done else self.batch_reward[k] + self.gamma * target_q_values[k]

    def update_loss(self):
        # logging.info('...... Updating loss ......')
        loss = self.app_critic.model.train_on_batch([self.batch_state, self.batch_action], self.batch_output)
        actor_predict = self.app_actor.model.predict(self.batch_state)
        actor_grad = self.app_critic.gradients(self.batch_state, actor_predict)
        self.app_actor.train(self.batch_state, actor_grad)
        self.app_actor.target_train()
        self.app_critic.target_train()
        return loss

    def get_action(self, state_t, train_indicator):
        # logging.info('...... Getting action ......')
        self.epsilon -= 1.0 / self.explore_iter * train_indicator
        noise = []
        action_ori = self.app_actor.model.predict(state_t)
        for i in range(self.action_size):
            a = action_ori[0][i]
            noise.append(train_indicator * max(self.epsilon, 0) * self.tools.ou(a, -0.5, 0.5, 0.3))
        action = action_ori + np.array(noise)
        return action

    def if_exit(self, step, state, collision, not_move, cannot_stop):
        if step >= self.max_steps:
            # logging.warn('Not finished with max steps! Start: ' + str(self.sim.Stop_Line - state[-1]) +
            #              ', Dis to SL: ' + str(state[4]) + ', Dis to FL: ' + str(state[3]) +
            #              ', Velocity: ' + str(state[0]) + ', V0: ' + str(self.sim.ini_speed))
            self.sub_not_finish += 1
            self.if_pass = False
            self.if_done = True
        elif state[0] >= self.sim.Speed_limit + 2.:
            # logging.warn('Exceed Speed Limit: ' + str(self.sim.Stop_Line - state[-1]) + ', Dis to SL: ' + str(state[4]) +
            #              ', Dis to FL: ' + str(state[3]) + ', Velocity: ' + str(state[0]) +
            #              ', V0: ' + str(self.sim.ini_speed))
            self.sub_overspeed += 1
            self.if_pass = False
            self.if_done = True
        elif not_move > 0:
            # logging.warn('Not move! Start: ' + str(self.sim.Stop_Line - state[-1]) + ', Dis to SL: ' + str(state[4]) +
            #              ', Dis to FL: ' + str(state[3]) + ', Velocity: ' + str(state[0]) +
            #              ', V0: ' + str(self.sim.ini_speed))
            self.sub_not_move += 1
            self.if_pass = False
            self.if_done = True
        elif collision > 0:
            # logging.warn('Crash to other vehicles or road boundary! Start: ' + str(self.sim.Stop_Line - state[-1]) +
            #              ', Dis to SL: ' + str(state[4]) + ', Dis to FL: ' + str(state[3]) +
            #              ', Velocity: ' + str(state[0]) + ', V0: ' + str(self.sim.ini_speed))
            self.sub_crash += 1
            self.if_pass = False
            self.if_done = True
        elif cannot_stop > 0:
            # logging.warn('Did not stop at stop line! Start: ' + str(self.sim.Stop_Line - state[-1]) +
            #              ', Dis to SL: ' + str(state[4]) + ', Dis to FL: ' + str(state[3]) +
            #              ', Velocity: ' + str(state[0]) + ', V0: ' + str(self.sim.ini_speed))
            self.sub_cannot_stop += 1
            self.if_pass = False
            self.if_done = True
        elif state[4] <= 0.5 and (state[0] <= 0.1):
            # logging.info('Congratulations! Reach stop line without crashing and has stopped. Start: ' +
            #              str(self.sim.Stop_Line - state[-1]) + ', Dis to SL: ' + str(state[4]) +
            #              ', Dis to FL: ' + str(state[3]) + ', Velocity: ' + str(state[0]) +
            #              ', V0: ' + str(self.sim.ini_speed))
            self.sub_success += 1
            self.if_pass = True
            self.if_done = True

    def launch_train(self, train_indicator=1):  # 1 means Train, 0 means simply Run
        # logging.info('Launch Training Process')
        # np.random.seed(1337)
        state_t = self.sim.get_state()
        state_dim = state_t.shape[1]
        self.app_actor = AppActorNetwork(self.tf_sess, state_dim, self.action_size, self.batch_size, self.tau, self.LRA)
        self.app_critic = AppCriticNetwork(self.tf_sess, state_dim, self.action_size, self.batch_size, self.tau, self.LRC)

        self.load_weights()

        for e in range(self.episode_count):
            self.total_loss = 0.
            total_time = time.time()
            total_reward = 0.
            # logging.debug("Episode : " + str(e) + " Replay Buffer " + str(self.buffer.count()))
            step = 0
            state_t = self.sim.get_state()
            while True:
                action_t = self.get_action(state_t, train_indicator)
                reward_t, collision, not_move, cannot_stop = self.reward.get_reward(state_t[0], action_t[0][0])
                self.sim.update_vehicle(reward_t, action_t[0][0])
                state_t1 = self.sim.get_state()
                if train_indicator:
                    self.update_batch(state_t, action_t[0], reward_t, state_t1)
                loss = self.update_loss() if train_indicator else 0.

                total_reward += reward_t
                self.if_exit(step, state_t[0], collision, not_move, cannot_stop)
                step += 1
                self.total_loss += loss
                train_time = time.time() - self.start_time
                # logging.debug('Episode: ' + str(e) + ', Step: ' + str(step) + ', Dis to SL: ' + str(state_t[0][6]) +
                #               ', Dis to fv: ' + str(state_t[0][5]) + ', v: ' + str(state_t[0][0]) +
                #               ', a: ' + str(action_t) + ', r: ' + str(reward_t) + ', loss: ' + str(loss) +
                #               ', time: ' + str(train_time))
                # total_time += train_time
                if self.if_done:
                    break
                self.start_time = time.time()
                state_t = state_t1
            self.loss.append(self.total_loss)
            self.total_rewards.append(total_reward)

            plt.close('all')
            total_step = step + 1
            if train_indicator:
                self.update_weights()

            # mean_loss = total_loss / total_step
            # mean_time = total_time / total_step
            mean_time = time.time() - total_time
            # logging.debug(str(e) + "-th Episode: Steps: " + str(total_step) + ', Time: ' + str(mean_time) +
            #               ', Reward: ' + str(total_reward) + " Loss: " + str(loss) + ', Crash: ' +
            #               str(self.sub_crash) + ', Not Stop: ' + str(self.sub_cannot_stop) + ', Not Finished: ' +
            #               str(self.sub_not_finish) + ', Overspeed: ' + str(self.sub_overspeed) + ', Not Move: ' +
            #               str(self.sub_not_move) + ', Success: ' + str(self.sub_success))


            # self.sim = InterSim(True) if e % 50 == 0 else InterSim()
            # self.sim = InterSim(task_pos[self.task] + 30. * random(), False)
            self.sim = InterSim(140*random() + 10., False)
            self.total_reward = 0.
            self.if_pass = False
            self.if_done = False

            if (e + 1) % 100 == 0:
                self.if_train.append(train_indicator)
                self.crash.append(self.sub_crash)
                self.success.append(self.sub_success)
                self.not_finish.append(self.sub_not_finish)
                self.overspeed.append(self.sub_overspeed)
                self.not_move.append(self.sub_not_move)
                self.cannot_stop.append(self.sub_cannot_stop)
                self.run_time.append((time.time() - self.total_time) / 60.)

                self.sub_crash = 0
                self.sub_cannot_stop = 0
                self.sub_success = 0
                self.sub_not_finish = 0
                self.sub_overspeed = 0
                self.sub_not_move = 0
                logging.info('Crash: ' + str(self.crash) + '\nNot Stop: ' + str(self.cannot_stop) +
                             '\nNot Finished: ' + str(self.not_finish) + '\nOverspeed: ' + str(self.overspeed) +
                             '\nNot Move: ' + str(self.not_move) + '\nSuccess: ' + str(self.success) +
                             '\nLoss: ' + str(loss) + '\nTime: ' + str(self.run_time) + '\nTest: ' + str(self.if_train))

                results = {'crash': self.crash, 'not_stop': self.cannot_stop, 'unfinished': self.not_finish,
                           'stop': self.not_move, 'overspeed': self.overspeed,
                           'succeess': self.success, 'reward': self.total_rewards, 'loss': self.loss}
                with open('task' + str(self.task) + '/result.txt', 'w+') as _file:
                # with open('task' + str(self.task) + '/result.txt', 'w+') as _file:
                    js_data = json.dumps(results)
                    _file.write(js_data)
                # train_indicator = 0 if train_indicator == 1 else 1
            # if (e + 1) % 1000 == 0:
            #     self.epsilon = 1.0


if __name__ == '__main__':
    plt.ion()
    while True:
        tmp_agent = ReinAcc(140*random() + 10., 9)
        tmp_agent.launch_train(1)

    alpha = 0.1
    task_pos = [10., 40., 70., 100, 130.]
    tictac = time.time()
    train_pro = []
    agents = []
    q = []
    q_exp = []
    for k, i in enumerate(task_pos):
        pos = i + 30. * random()
        tmp_agent = ReinAcc(pos, k)
        tmp_agent.launch_train(1)
        agents.append(tmp_agent)
        q.append(sum(tmp_agent.total_rewards[-Step_size:]) / Step_size / 1000.)
        q_exp.append(float(np.exp(q[-1])))
        logging.info('Time: {0:.2f}'.format((time.time() - tictac) / 3600.) + ', cond: ' + str(k) +
                     ', Success: ' + str(tmp_agent.success))

    while True:
        q_p = np.array(q_exp) / (sum(q_exp))
        train_pro.append(q_exp)
        with open('train_pro.txt', 'w+') as json_file:
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
        else:
            next_ind = 4
        strFormat = len(q_p) * '{:2.3f} '
        logging.debug('[' + strFormat.format(*q_p) + '], ' + 'Next ind: ' + str(next_ind))

        tmp_agent = agents[next_ind]
        tmp_agent.app_actor.model.save_weights("weights/actormodel.h5", overwrite=True)
        with open("weights/actormodel.json", "w") as outfile:
            json.dump(tmp_agent.app_actor.model.to_json(), outfile)
            tmp_agent.app_critic.model.save_weights("weights/criticmodel.h5", overwrite=True)
        with open("weights/criticmodel.json", "w") as outfile:
            json.dump(tmp_agent.app_critic.model.to_json(), outfile)

        old_q = q
        q = []
        q_exp = []
        for k, i in enumerate(task_pos):
            # logging.debug(str(k) + ', ' + str(i))
            tmp_agent = agents[k]
            if k == next_ind:
                tmp_agent.launch_train(1)
            else:
                tmp_agent.launch_train(0)
            # q.append(float(np.exp(improve)))
            if sum(tmp_agent.success[-(Step_size / 50):]) / (Step_size / 5.) <= 8.0:
                # improve = (sum(tmp_agent.successes[-(Step_size / 100):]) -
                #            sum(tmp_agent.successes[-2 * (Step_size / 100):-(Step_size / 100)])) / (Step_size / 50.)
                # q.append(float(np.exp(abs(improve))))
                qq = alpha * sum(tmp_agent.total_rewards[-Step_size:]) / Step_size / 1000. + \
                     (1 - alpha) * old_q[k]
                q.append(qq)
                q_exp.append(float(np.exp(qq)))
                # q[next_ind] = float(np.exp(sum(tmp_agent.successes[-(Step_size / 100):]) / (Step_size / 10.)))
            else:
                qq = - alpha * 10. + (1 - alpha) * old_q[k]
                q_exp.append(float(np.exp(qq)))
                # q[next_ind] = float(np.exp(-10.))
            agents[k] = tmp_agent
            logging.info('Time: {0:.2f}'.format((time.time() - tictac) / 3600.) +
                         ', cond: ' + str(k) + ', Success: ' + str(tmp_agent.success))

