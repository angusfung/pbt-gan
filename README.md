# pbt-gan

PBT is an optimization algorithm that maximizes the performance of a network by optimizating a population of models and their hyperparameters. It determines a schedule of hyperparameter settings using an evolutionary strategy of exploration and exploitation - a much more powerful method than simply using a fixed set of hyperparameters throughout the entire training or using grid-search and hand-tuning, which is time-extensive and difficult.

Implementation of PBT-GAN experiments from [paper](https://arxiv.org/pdf/1711.09846.pdf).  
(Refer [here](https://github.com/angusfung/population-based-training) for Toy Experiments from paper)  

## Setup
It is recommended to run from a virtual environment to ensure all dependencies are met.  
Compatible with both python 2 and 3.
```
virtualenv -p python pbt_env
source pbt_env/bin/activate.csh
pip install -r requirements.txt
```
## Memory Utilization
Memory limits can be set on a per-worker basis (as a percentage) by uncommenting `gpu_options` in `pbt_main.py` which can be desired for synchronous training.

## Training

## Results

## Saved Sessions

## Credits
GAN templates from [here](https://github.com/hwalsuklee/tensorflow-generative-model-collections) and [here](https://github.com/igul222/improved_wgan_training)

