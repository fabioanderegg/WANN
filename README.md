# Overview
This repo contains multiple experiments to train and evaluate weight agnostic neural networks on MNIST and compare them to standard (convolutional) neural networks.
The repo is a fork of https://github.com/google/brain-tokyo-workshop/.

# Experiments
Every experiment is in a subdirectory, each one contains a README.md which explains what the idea of the experiment is and how it works.
Training the WANNs only requires CPUs and cannot use a GPU. Training the CNNs was done on https://colab.research.google.com/ with a GPU runtime
but can easily be changed to work on a workstation with a GPU.

## Instructions
General approach for all experiments:
* Install dependencies: `pip install numpy mpi4py gym mnist cma opencv-python pyradiomics jupyter`
* Run notebooks to extract features from MNIST contained in the subdirectory (expect for the basic MNIST 28x28 experiment in the wann/ directory)
* Copy the extracted features into the WANNRelease directory (probably do this on a server where scripts can run for a long time)
* Build the neural network architecture with WANN (using the train_mnist*.sh scripts in the WANNRelease/WANN directory), this takes hours to days(!) to get a usable result
* Extract the best architecture from the population with the best_mnist*.sh scripts in the WANNRelease/WANN directory
* Finetune the model weights (we still use weights for every neuron, because with only one weight the accuracy would be even worse then the alreay are)
  * Copy the best population file to the WANNRelease/WANNTool directory using the copy_mnist*.sh script
  * Run the finetuning with finetune*.sh scripts, until the accuracy does not increase any more (should take not more than some hours, probably less)
* Use the notebooks in each experiment subdirectory to evaluate the results from the WANN

## Available experiments
* wann: Train a WANN on the full sized MNIST dataset, the original paper only trained on resized 18x18 pixel images.
* wann_features: Train a WANN on features extracted from MNIST images using pyradiomics and OpenCV
* wann_ddfe: Use an autoencoder to extract features from MNIST images, train a WANN on these extracted features

## Results
* wann: Accuracy: 0.8637
* wann_features: Accuracy: 0.3
* wann_ddfe: ?

# Neat
Using python-neat was also explored (in the neat/ subdirectory), but was never brought to a working state.
