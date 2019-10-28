# Overview
The original GitHub release of the WANN paper already contains an experiment for MNIST, but it only uses resized 16x16 pixel images.
This experiment trains a WANN with the full size 28x28 resolution.

## WANN Training
Training the WANN takes a long time. Therefore running the training process on a server.
Here are the setup instructions to run it on a Ubuntu 18.04 server:

```bash
git clone https://github.com/fabioanderegg/brain-tokyo-workshop.git
cd brain-tokyo-workshop
sudo apt install -y python3-venv python3-dev build-essential libopenmpi-dev openmpi-bin
python3 -m venv venv
source venv/bin/activate
pip install wheel
pip install numpy mpi4py gym mnist cma
```

Switch into the WANNRelease/WANN directory and run `train_mnist784.sh` for some hours/days. This
generates the WANN architecture. Next, run `best_mnist784.sh`. This extracts the best indiviual from the last
population of the WANNs. Switch to the WANNRelease/WANNTool directory.
Run `copy_mnist784.sh`, then `finetune_mnist784.sh`.

## Evaluation
The following jupyter notebooks are used to evaluate the WANN:

* `wann16x16.ipynb`: Evaluate the WANN model provided in the original paper GitHub repo
* `wann28x28_accuracy.ipynb`: Calculates the accuracy of the WANN
* `wann28x28_evaluate_occlusion.ipynb`: Occlusion interpretability method
* `wann28x28_evaluate_rise.ipynb`: RISE interpretability method
* `wann28x28_rise_check.ipynb`: Validate that RISE actually shows the important pixels, result: More or less
* `wann28x28_rise_plot.ipynb`: Plot how many pixels from the RISE output have to be removed from the input image until the detected class changes

The following notebooks train a CNN and run the same evaluations as the WANN notebooks, so the CNN and WANN can be compared:
* `mnist16x16_cnn.ipynb`: Trains and evaluates a CNN on the 16x16 dataset
* `mnist28x28_train.ipynb`: Trains a CNN on the 28x28 dataset
* `mnist28x28_evaluate.ipynb`: Occlusion and RISE interpretability method
* `mnist28x28_rise_check.ipynb`: Validate that RISE actually shows the important pixels, result: More or less
* `mnist28x28_rise_plot.ipynb`: Plot how many pixels from the RISE output have to be removed from the input image until the detected class changes
