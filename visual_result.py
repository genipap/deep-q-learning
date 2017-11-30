import numpy as np
import matplotlib.pyplot as plt
import json

__author__ = 'qzq'

# plt.plot(np.arange(0.0, 1.0, 0.01), 1000 * np.arange(0.0, 1.0, 0.01), 'r.')
# plt.show()

a = 0
if a == 0:
    # file_name = 'ddqn1'
    file_name = 'vehicle1-2'
    with open(file_name + '.txt', 'r') as json_file:
        results = json.load(json_file)
    correct_key = {'crash', 'stop', 'succeess'}
else:
    # file_name = 'ch_rule2_2'
    # file_name = ' ch_rule_g1_2'
    # file_name = 'ch_rule_g1_4'
    # file_name = 'ch_rule_g1_5'
    file_name = 'new3'
    with open('../results/' + file_name + '.txt', 'r') as json_file:
        results = json.load(json_file)
    # with open('../results/' + file_name1 + '.txt', 'r') as json_file:
    #     results1 = json.load(json_file)
    correct_key = {'crash', 'unfinished', 'overspeed', 'stop', 'succeess'}

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

train_loss = np.reshape(results['loss'], (len(results['loss']), 1))[:, 0]
train_re = np.reshape(results['reward'], (len(results['reward']), 1))[:, 0]

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
plt.subplot(311)
plt.title('critic loss: {0:.2f}'.format(np.mean(train_loss[-100:])))
# plt.ylim([0, 5 * 1e22])
# # plt.xlim([0, 1000])
# plt.plot(np.arange(total_ep), train_loss, 'r', label='loss')
plt.ylim([0, 200])
plt.plot(train_loss, 'r', label='loss')
# plt.legend(loc=1)
plt.subplot(312)
plt.title('reward: {0:.2f}'.format(np.mean(train_re[-100:])))
plt.plot(train_re, 'y.', label='reward')
plt.subplot(313)
plt.title('Success rate every 100 epochs')
plt.plot(results['succeess'], 'b', label='success')
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

plt.show()
