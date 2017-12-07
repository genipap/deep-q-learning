import matplotlib.pyplot as plt
import numpy as np
from random import randint, random, sample, uniform
from matplotlib.patches import Rectangle as Rectangle
from utilities.toolfunc import ToolFunc
import logging

__author__ = 'qzq'


class InterSim(object):
    Tau = 1. / 30
    Speed_limit = 12        # m/s
    Scenary = randint(0, 2)
    Inter_Ori = 0.
    Stop_Line = - 5. - random()
    Pass_Point = 20.
    Inter_Low = - 4.
    Inter_Up = 4.
    Inter_Left = - 4.
    Inter_Right = 4.
    FV_NO = 1
    LV_NO = 4
    RV_NO = 5
    Lane_Left = 0.
    Lane_Right = 4.
    Cft_Accel = 3.     # m/s**2

    tools = ToolFunc()

    def __init__(self, ini_pos, visual=False):
        self.Visual = visual
        self.av_pos = dict()
        self.av_pos['y'] = - ini_pos
        self.Start_Pos = self.av_pos['y']
        self.av_pos['x'] = 2.
        self.av_pos['vx'] = 0.
        self.av_pos['vy'] = max((self.Speed_limit - random() * 3.) * ini_pos / 30., self.Speed_limit - random() * 4.)
        self.ini_speed = self.av_pos['vy']
        self.av_pos['heading'] = 0
        self.av_pos['accel'] = 0
        self.av_pos['steer'] = 0
        self.av_size = [4., 2.]
        self.av_pos['w'] = 2.
        self.av_pos['l'] = 4.
        self.fv_poses = []
        for i in range(self.FV_NO):
            fv_pos = dict()
            fv_pos['y'] = self.av_pos['y'] + random() * 20. + 10.
            fv_pos['x'] = 2.
            fv_pos['vx'] = 0.
            fv_pos['vy'] = self.Speed_limit - random()
            fv_pos['w'] = 2.
            fv_pos['l'] = 4.
            self.fv_poses.append(fv_pos)
        self.target_dis = None
        self.target_v = None
        self.state = None
        self.state_dim = None

        self.state_av = []
        self.state_fv = []
        self.state_road = []

        self.fig = None
        self.ax = None

    def draw_scenary(self, av, fvs, r, a):
        if self.Visual:
            self.fig = plt.figure(1)
            self.ax = self.fig.add_subplot(1, 1, 1)
            plt.axis([-100, 100, -110, 110])
            self.ax.fill_between(np.arange(-104, self.Inter_Left, 0.5), self.Inter_Low,
                                 np.arange(self.Inter_Low, -104, -0.5), facecolor='black')
            self.ax.fill_between(np.arange(-104, self.Inter_Left, 0.5), self.Inter_Up,
                                 np.arange(self.Inter_Up, 104, 0.5), facecolor='black')
            self.ax.fill_between(np.arange(self.Inter_Right, 104, 0.5), self.Inter_Low,
                                 np.arange(-104, self.Inter_Low, 0.5), facecolor='black')
            self.ax.fill_between(np.arange(self.Inter_Right, 104, 0.5), self.Inter_Up,
                                 np.arange(104, self.Inter_Up, -0.5), facecolor='black')
            self.ax.add_patch(Rectangle((av['x'] - av['w'] / 2., av['y'] - av['l']), av['w'], av['l'], color='red'))
            for v in fvs:
                self.ax.add_patch(Rectangle((v['x'] - v['w'] / 2., v['y'] - v['l']), v['w'], v['l'], color='green'))
            self.ax.plot(list(xrange(-104, 104)), list([(self.Inter_Up + self.Inter_Low) / 2.] * 208), 'y--')
            self.ax.plot(list([(self.Inter_Right + self.Inter_Left) / 2.] * 208), list(xrange(-104, 104)), 'y--')
            self.ax.plot(list(xrange(0, 5)), list([self.Stop_Line] * 5), 'r')
            plt.text(av['x'], av['y'], 'a: {0:.2f}'.format(a) + ', v: {0:.2f}'.format(av['vy']) +
                     '\n reward: {0:.2f}'.format(r) + ', center_dis: {0:.2f}'.format(0. - av['y'])
                     + ', fv_dis: {0:.2f}'.format(self.fv_poses[0]['y'] - self.av_pos['y']),
                     color='red')
            plt.show()
            plt.pause(0.1)
            plt.clf()

    def get_state(self):
        self.state_av = [self.av_pos['vy'], self.av_pos['accel']]
        fv_dis_list = [fv_pos['y'] - self.av_pos['y'] for fv_pos in self.fv_poses]
        fv_index = np.argmin(fv_dis_list)
        fv_pos = self.fv_poses[fv_index]
        self.state_fv = [fv_pos['vy'], fv_pos['y'] - self.av_pos['y']]
        sl_dis = self.Stop_Line - self.av_pos['y']
        # ll = self.av_pos['x'] - self.av_size[1] / 2 - self.Lane_Left
        # lr = self.Lane_Right - (self.av_pos['x'] + self.av_size[1] / 2)
        start_pos = self.Start_Pos
        self.state_road = [sl_dis, start_pos]
        self.state = np.array(self.state_av + self.state_fv + self.state_road, ndmin=2)
        self.state_dim = self.state.shape[1]
        return self.state

    def update_vehicle(self, r, a=0, st=0):
        accel = self.Cft_Accel * a
        for fv_pos in self.fv_poses:
            fv_a = - 0.5 * (fv_pos['vy'] ** 2) / (self.Stop_Line - fv_pos['y'] - 4.) if (fv_pos['y'] + 4.) < self.Stop_Line - 1. \
                else self.Cft_Accel
            fv_pos['vy'] += fv_a * self.Tau
            fv_pos['vy'] = min(max(0.1, fv_pos['vy']), self.Speed_limit)
            fv_pos['y'] += fv_pos['vy'] * self.Tau + 0.5 * fv_a * (self.Tau ** 2)
        old_av_vel = self.av_pos['vy']
        new_v = self.av_pos['vy'] + accel * self.Tau
        self.av_pos['vy'] = max(0.0, new_v)
        if new_v >= 0:
            self.av_pos['y'] += old_av_vel * self.Tau + 0.5 * accel * (self.Tau ** 2)
        else:
            self.av_pos['y'] += 0.5 * abs(accel) * ((old_av_vel / accel) ** 2)
        # self.av_pos['heading'] += st
        self.av_pos['accel'] = accel
        # self.av_pos['steer'] = st
        if self.Visual:
            self.draw_scenary(self.av_pos, self.fv_poses, r, accel)


if __name__ == '__main__':
    sim = InterSim()
    plt.ion()
    while sim.av_pos['y'] <= sim.Pass_Point:
        sim.get_state()
        # sim.update_vehicle()

