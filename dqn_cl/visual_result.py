import numpy as np
import matplotlib.pyplot as plt
import json

__author__ = 'qzq'

# plt.plot(np.arange(0.0, 1.0, 0.01), 1000 * np.arange(0.0, 1.0, 0.01), 'r.')
# plt.show()

ll = [2, 1, 0, -1, -2, -3]

file_name1 = 'papere/s2/result'
file_name2 = 'papere/s1/result'
file_name3 = 'papere/s0/result'
file_name4 = 'papere/s-1/result'
file_name5 = 'papere/s-2/result'
file_name6 = 'papere/s-3/result'
file_name = 'random_v'
with open(file_name + '.txt', 'r') as json_file:
    random_r = json.load(json_file)

with open(file_name1 + '.txt', 'r') as json_file:
    results1 = json.load(json_file)
with open(file_name2 + '.txt', 'r') as json_file:
    results2 = json.load(json_file)
with open(file_name3 + '.txt', 'r') as json_file:
    results3 = json.load(json_file)
with open(file_name4 + '.txt', 'r') as json_file:
    results4 = json.load(json_file)
with open(file_name5 + '.txt', 'r') as json_file:
    results5 = json.load(json_file)
with open(file_name6 + '.txt', 'r') as json_file:
    results6 = json.load(json_file)
correct_key = {'crash', 'stop', 'succeess'}

# ep = len(results['crash']) / 2
# # results = {'crash': crash, 'non_stop': non_stop, 'unfinished': unfinished, 'overspeed': overspeed, 'stop': stop,
# train_result = dict()
# test_result = dict()
# train_qun = dict()
# test_qun = dict()
#
#
# for key in correct_key:
#     train_result[key] = np.reshape(results[key], (ep, 2))[:, 0]
#     test_result[key] = np.reshape(results[key], (ep, 2))[:, 1]
#
# ep1 = len(results['reward']) / 100
# total_ep = len(results['reward']) / 2
# quan_key = {'reward'}
# for key in quan_key:
#     train_qun[key] = np.reshape(results[key], (ep1, 100))[::2, :]
#     train_qun[key] = np.reshape(train_qun[key], (1, total_ep))[0]
#     test_qun[key] = np.reshape(results[key], (ep1, 100))[1::2, :]
#     test_qun[key] = np.reshape(test_qun[key], (1, total_ep))[0]

train_loss1 = np.reshape(results1['loss'], (len(results1['loss']), 1))[:, 0]
train_loss2 = np.reshape(results2['loss'], (len(results2['loss']), 1))[:, 0]
train_loss3 = np.reshape(results3['loss'], (len(results3['loss']), 1))[:, 0]
train_loss4 = np.reshape(results4['loss'], (len(results4['loss']), 1))[:, 0]
train_loss5 = np.reshape(results5['loss'], (len(results5['loss']), 1))[:, 0]
train_loss6 = np.reshape(results6['loss'], (len(results6['loss']), 1))[:, 0]
train_re1 = np.reshape(results1['reward'], (len(results1['reward']), 1))[:, 0]
train_re2 = np.reshape(results2['reward'], (len(results2['reward']), 1))[:, 0]
train_re3 = np.reshape(results3['reward'], (len(results3['reward']), 1))[:, 0]
train_re4 = np.reshape(results4['reward'], (len(results4['reward']), 1))[:, 0]
train_re5 = np.reshape(results5['reward'], (len(results5['reward']), 1))[:, 0]
train_re6 = np.reshape(results6['reward'], (len(results6['reward']), 1))[:, 0]

# fig1 = plt.figure(1)
# plt.subplot(211)
# # plt.xlim([0, 10])
# # plt.title('train, total time: {0:.2f} hr'.format(results['time'][-1] / 60.))
# plt.title('train')
# for key, value in train_result.iteritems():
#     plt.plot(np.arange(ep), value, 'o-', label=key)
# plt.legend(loc=1)
#
# plt.subplot(212)
# plt.title('test')
# for key, value in test_result.iteritems():
#     plt.plot(np.arange(ep), value, 'o-', label=key)
# plt.legend(loc=1)
# # plt.xlim([0, 10])
# fig1.set_size_inches(12, 9)
# # fig1.savefig('../results/' + file_name + '_1.png', dpi=fig1.dpi)

fig2 = plt.figure(2)
plt.subplot(111)
# plt.title('Loss')
plt.ylim([-100, 2000])
plt.xlabel('epoch')
plt.ylabel('Loss value')
# # plt.xlim([0, 1000])
# plt.plot(np.arange(total_ep), train_loss, 'r', label='loss')
# plt.plot(random_r['loss'][0:280000], color='pink', label='Random Curriculum')
plt.plot(train_loss1[0:90000], label='Task 1')
plt.plot(train_loss2[0:90000], label='Task 2')
plt.plot(train_loss3[0:90000], label='Task 3')
plt.plot(train_loss4[0:90000], label='Task 4')
plt.plot(train_loss5[0:90000], label='Task 5')
plt.plot(train_loss6[0:90000], label='Task 6')
plt.legend(loc=2)


fig3 = plt.figure(3)
plt.subplot(111)
# plt.title('reward: {0:.2f}'.format(np.mean(train_re1[-100:])))
# plt.xlim([0, 500])
plt.xlabel('epochs')
plt.ylabel('Reward value mean')
r_m = np.mean(np.reshape(random_r['reward'][0:280000], (280, 1000)), axis=1)
r_x = np.linspace(0, 280000, num=281)
plt.plot(random_r['reward'][0:280000], color='mistyrose')
plt.plot(r_x[0:-1], r_m, color='r', label='Random Curriculum')
# plt.plot(train_re1[0:280000], label='2')
# plt.plot(train_re2[0:280000], label='1')
# plt.plot(train_re3[0:280000], label='0')
# plt.plot(train_re4[0:280000], label='-1')
# plt.plot(train_re5[0:280000], label='-2')
# plt.plot(train_re6[0:280000], label='-3')
rr = (np.array(train_re1[0:90000]) + np.array(train_re2[0:90000])
          + np.array(train_re3[0:90000]) + np.array(train_re4[0:90000])
          + np.array(train_re5[0:90000]) + np.array(train_re6[0:90000]))/6.
c_m = np.mean(np.reshape(rr, (90, 1000)), axis=1)
c_x = np.linspace(0, 90000, num=91)
plt.plot(rr, 'lightgreen')
plt.plot(c_x[0:-1], c_m, 'g', label='Automatically Generated Curriculum')
plt.legend(loc=4)

fig1 = plt.figure(1)
plt.subplot(111)
# plt.title('Success rate every 1000 epochs')
plt.ylim([0, 101])
plt.xlabel('1000 epochs')
plt.ylabel('Success traversing rate (%)')
# plt.xlim([0, 2500])
# plt.plot(np.mean(np.reshape(results1['succeess'], (len(results1['succeess'])/5, 5)), axis=1), '.-', label='2')
# plt.plot(np.mean(np.reshape(results2['succeess'], (len(results2['succeess'])/5, 5)), axis=1), '.-', label='2')
# plt.plot(np.mean(np.reshape(results3['succeess'], (len(results3['succeess'])/5, 5)), axis=1), '.-', label='2')
# plt.plot(np.mean(np.reshape(results4['succeess'], (len(results4['succeess'])/5, 5)), axis=1), '.-', label='2')
# plt.plot(np.mean(np.reshape(results5['succeess'], (len(results5['succeess'])/5, 5)), axis=1), '.-', label='2')
# plt.plot(np.mean(np.reshape(results6['succeess'], (len(results6['succeess'])/5, 5)), axis=1), '.-', label='2')
plt.plot(results1['succeess'], '.-', color='coral', label='2')
plt.plot(results2['succeess'], '.-', color='yellow', label='1')
plt.plot(results3['succeess'], '.-', color='palegreen', label='0')
plt.plot(results4['succeess'], '.-', color='skyblue', label='-1')
plt.plot(results5['succeess'], '.-', color='pink', label='-2')
plt.plot(results6['succeess'], '.-', color='plum', label='-3')

min_len = min(len(results1['succeess']), len(results2['succeess']), len(results3['succeess']),
              len(results4['succeess']), len(results5['succeess']), len(results6['succeess']))
mean_success = (np.array(results1['succeess'][0:min_len]) + np.array(results2['succeess'][0:min_len]) +
                np.array(results3['succeess'][0:min_len]) + np.array(results4['succeess'][0:min_len]) +
                np.array(results5['succeess'][0:min_len]) + np.array(results6['succeess'][0:min_len]))/6.
mean_succ = np.mean(np.reshape(mean_success, (min_len/10, 10)), axis=1)
plt.plot(mean_succ[0:90], 'o-', color='red', label='Automatically Generated Curriculum')
ran_succ = random_r['succeess'][0:2800]
rand_succ = np.mean(np.reshape(ran_succ, (280, 10)), axis=1)
plt.plot(rand_succ, 'o-', color='g', label='Random Curriculum')
plt.legend(loc=4)
# '4h to 100%'
# # # plt.ylim([-5000, 1000])
# # # plt.xlim([0, 1000])
# plt.title('rewards: {0:.2f}'.format(np.mean(test_qun['reward'][-100:])))
# plt.plot(np.arange(total_ep), train_qun['reward'], 'r', label='train reward')
# plt.plot(np.arange(total_ep), test_qun['reward'], 'g', label='test reward')
# plt.legend(loc=1)
# plt.subplot(313)
# # plt.xlim([0, 1000])
# # plt.ylim([0, 500])
# plt.title('max jerk: {0:.2f}'.format(np.mean(test_qun['max_j'][-100:])))
# plt.plot(np.arange(total_ep), train_qun['max_j'], 'r', label='train max jerk')
# plt.plot(np.arange(total_ep), test_qun['max_j'], 'g', label='test max jerk')
# plt.legend(loc=1)
# fig2.set_size_inches(24, 18)
# fig2.savefig('../results/' + file_name + '_2.png', dpi=fig2.dpi)
#


file_name = 'train_pro5'
with open(file_name + '.txt', 'r') as json_file:
    train_pro = json.load(json_file)
train_pro = np.array(train_pro)
sump = np.sum(train_pro, axis=1)
sump = np.array(sump, ndmin=2).T
pp = np.divide(train_pro, sump)
fig4 = plt.figure(4)
plt.subplot(111)
plt.ylim([-0.1, 1.1])
plt.ylabel('Probability to be chosen')
plt.xlabel('Taining iteration')
for i in range(pp.shape[1]):
    plt.plot(pp[:, i][0:180], '.-', label='Task'+str(i+1))
plt.legend(loc=4)

plt.show()
