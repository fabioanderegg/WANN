import numpy as np
import gym

def make_env(env_name, seed=-1, render_mode=False):
  if (env_name.startswith("RocketLander")):
    from box2d.rocket import RocketLander
    env = RocketLander()
  elif (env_name.startswith("VAERacing")):
    from vae_racing import VAERacing
    env = VAERacing()
  elif (env_name.startswith("BipedalWalker")):
    if (env_name.startswith("BipedalWalkerHardcore")):
      from box2d.biped import BipedalWalkerHardcore
      env = BipedalWalkerHardcore()
    else:
      from box2d.biped import BipedalWalker
      env = BipedalWalker()
  elif (env_name.startswith("CartPoleSwingUp")):
    print("cartpole_swingup_started")
    from custom_envs.cartpole_swingup import CartPoleSwingUpEnv
    env = CartPoleSwingUpEnv()
  elif (env_name.startswith("MNIST")):
    print("MNIST256_started")
    from custom_envs.classify_gym import ClassifyEnv, mnist_256, mnist_256_test, mnist_784_test, mnist_784, \
        mnist_features_test, mnist_features, mnist_autoencoder, mnist_autoencoder_test
    if env_name.startswith("MNISTTEST256"):
      test_images, test_labels  = mnist_256_test()
      env = ClassifyEnv(test_images, test_labels, batch_size=10000, accuracy_mode=True)
    elif env_name.startswith("MNISTTRAIN256"):
      train_images, train_labels  = mnist_256()
      env = ClassifyEnv(train_images, train_labels, batch_size=60000, accuracy_mode=True)
    elif env_name.startswith("MNISTTEST784"):
      test_images, test_labels  = mnist_784_test()
      env = ClassifyEnv(test_images, test_labels, batch_size=10000, accuracy_mode=True)
    elif env_name.startswith("MNISTTRAIN784"):
      train_images, train_labels  = mnist_784()
      env = ClassifyEnv(train_images, train_labels, batch_size=60000, accuracy_mode=True)
    elif env_name.startswith("MNISTTESTFEATURES"):
      test_images, test_labels  = mnist_features_test()
      env = ClassifyEnv(test_images, test_labels, batch_size=10000, accuracy_mode=True)
    elif env_name.startswith("MNISTTRAINFEATURES"):
      train_images, train_labels  = mnist_features()
      env = ClassifyEnv(train_images, train_labels, batch_size=60000, accuracy_mode=True)
    elif env_name.startswith("MNISTTESTAUTOENCODER"):
      test_images, test_labels  = mnist_autoencoder_test()
      env = ClassifyEnv(test_images, test_labels, batch_size=10000, accuracy_mode=True)
    elif env_name.startswith("MNISTTRAINAUTOENCODER"):
      train_images, train_labels  = mnist_autoencoder()
      env = ClassifyEnv(train_images, train_labels, batch_size=60000, accuracy_mode=True)
    else:
      trainSet, target  = mnist_256()
      env = ClassifyEnv(trainSet, target)
  if (seed >= 0):
    env.seed(seed)
  '''
  print("environment details")
  print("env.action_space", env.action_space)
  print("high, low", env.action_space.high, env.action_space.low)
  print("environment details")
  print("env.observation_space", env.observation_space)
  print("high, low", env.observation_space.high, env.observation_space.low)
  assert False
  '''
  return env
