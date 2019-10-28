from model import Model
from collections import namedtuple
import random
import numpy as np

random.seed(1)
np.random.seed(1)

Game = namedtuple('Game', ['env_name', 'input_size', 'output_size', 'wann_file', 'action_select', 'weight_bias'])
game = Game(env_name='MNISTTESTFEATURES-v0',
  input_size=20,
  output_size=10,
  wann_file='mnist_features_best.out',
  action_select='softmax', # all, soft, hard
  weight_bias=0.0,
)
model = Model(game)
model.make_env()
model.env.seed(1)
model.load_model('log/mnistfeaturestrain.cma.4.32.best.json')

batch = model.env.reset()
output = model.get_action(batch)

classes = np.argmax(output, axis=1)

correct_count = np.count_nonzero(classes == model.env.target[model.env.currIndx])

print('accuracy: ', correct_count / len(output))
