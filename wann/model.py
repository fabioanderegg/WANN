from collections import namedtuple

import numpy as np
import random

import json
import sys

from classify_gym import mnist_256_test, ClassifyEnv
import ann

import argparse

np.set_printoptions(precision=2)
np.set_printoptions(linewidth=160)

final_mode = True
render_mode = False

RENDER_DELAY = False
record_video = False
MEAN_MODE = False


class Model:
    ''' simple feedforward model '''

    def __init__(self, game):
        self.env_name = game.env_name
        self.wann_file = game.wann_file
        self.input_size = game.input_size
        self.output_size = game.output_size
        self.action_select = game.action_select
        self.weight_bias = game.weight_bias

        self.wVec, self.aVec, self.wKey = ann.importNet(self.wann_file)

        self.param_count = len(self.wKey)

        self.weights = np.zeros(self.param_count)

        self.render_mode = False

    def make_env(self):
        self.render_mode = render_mode

        test_images, test_labels = mnist_256_test()
        self.env = ClassifyEnv(test_images, test_labels, batch_size=10000, accuracy_mode=True)

    def get_action(self, x):
        # if mean_mode = True, ignore sampling.
        annOut = ann.act(self.wVec, self.aVec, self.input_size, self.output_size, x)
        action = ann.selectAct(annOut, self.action_select)
        return action

    def set_model_params(self, model_params):
        assert (len(model_params) == self.param_count)
        self.weights = np.array(model_params)
        for idx in range(self.param_count):
            key = self.wKey[idx]
            self.wVec[key] = self.weights[idx] + self.weight_bias

    def load_model(self, filename):
        with open(filename) as f:
            data = json.load(f)
        print('loading file %s' % (filename))
        self.data = data
        model_params = np.array(data[0])  # assuming other stuff is in data
        self.set_model_params(model_params)

    def get_random_model_params(self, stdev=0.1):
        return np.random.randn(self.param_count) * stdev

    def get_uniform_random_model_params(self, stdev=2.0):
        return np.random.rand(self.param_count) * stdev * 2 - stdev

    def get_single_model_params(self, weight=-1.0):
        return np.array([weight] * self.param_count)


def simulate(model, train_mode=False, render_mode=True, num_episode=5, seed=-1, max_len=-1):
    reward_list = []
    t_list = []

    orig_mode = True  # hack for bipedhard's reward augmentation during training (set to false for hack)

    dct_compress_mode = False

    max_episode_length = 1000

    random.seed(seed)
    np.random.seed(seed)
    model.env.seed(seed)

    obs = model.env.reset()

    if obs is None:
        obs = np.zeros(model.input_size)

    total_reward = 0.0
    stumbled = False  # hack for bipedhard's reward augmentation during training. turned off.
    reward_threshold = 300  # consider we have won if we got more than this

    num_glimpse = 0

    for t in range(max_episode_length):
        action = model.get_action(obs)

        prev_obs = obs

        obs, reward, done, info = model.env.step(action)

        if train_mode and reward == -100 and (not orig_mode):
            # hack for bipedhard's reward augmentation during training. turned off.
            reward = 0
            stumbled = True

        total_reward += reward

        if done:
            if train_mode and (not stumbled) and (total_reward > reward_threshold) and (not orig_mode):
                # hack for bipedhard's reward augmentation during training. turned off.
                total_reward += 100
            break

    if render_mode:
        print("reward", total_reward, "timesteps", t)

    reward_list.append(total_reward)
    t_list.append(t)

    return reward_list, t_list


def main():
    global RENDER_DELAY

    parser = argparse.ArgumentParser(description=('Train policy on OpenAI Gym environment '
                                                  'using pepg, ses, openes, ga, cma'))
    parser.add_argument('gamename', type=str, help='robo_pendulum, robo_ant, robo_humanoid, etc.')
    parser.add_argument('-f', '--filename', type=str, help='json filename', default='none')
    parser.add_argument('-e', '--eval_steps', type=int, default=100, help='evaluate this number of step if final_mode')
    parser.add_argument('-s', '--seed_start', type=int, default=0, help='initial seed')
    parser.add_argument('-w', '--single_weight', type=float, default=-100, help='single weight parameter')
    parser.add_argument('--stdev', type=float, default=2.0, help='standard deviation for weights')
    parser.add_argument('--sweep', type=int, default=-1, help='sweep a set of weights from -2.0 to 2.0 sweep times.')
    parser.add_argument('--lo', type=float, default=-2.0, help='slow side of sweep.')
    parser.add_argument('--hi', type=float, default=2.0, help='high side of sweep.')

    args = parser.parse_args()

    assert len(sys.argv) > 1, 'python model.py gamename path_to_mode.json'

    Game = namedtuple('Game', ['env_name', 'input_size', 'output_size', 'wann_file', 'action_select', 'weight_bias'])
    game = Game(env_name='MNISTTEST256-v0',
      input_size=256,
      output_size=10,
      wann_file='mnist.out',
      action_select='softmax', # all, soft, hard
      weight_bias=0.0,
    )


    filename = args.filename

    print("filename", filename)

    the_seed = args.seed_start

    model = Model(game)
    print('model size', model.param_count)

    eval_steps = args.eval_steps

    model.make_env()

    model.load_model(filename)

    ''' random uniform params
    params = model.get_uniform_random_model_params(stdev=weight_stdev)-game.weight_bias
    model.set_model_params(params)
    '''
    reward, steps_taken = simulate(model, train_mode=False, render_mode=False, num_episode=1,
                                   seed=the_seed)
    print(reward)


if __name__ == "__main__":
    main()
