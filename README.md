# Artificial Neural Network trained with Particle Swarm Optimization for compressive strength analysis of concrete

Authors: Antoine and Arne

## Description



## Installation venv

`python3 -m venv .venv`

### Troubleshooting

`sudo apt install python3.10-venv`

## Activation venv

`source .venv/bin/activate`

## Going further

### Dataset

- using crossvalidation to evaluate model performance
- using a validation dataset

### PSO

- change the activation function
- change the topology of the input ANN (number of layers, number of neurons per layer)

### Error evaluation

- implement other error metrics (MSE, RMSE, R2)

### Informants selection
- implement different informants selection strategies (radius-based, k-nearest neighbors, random every iteration)

### ANN Backpropagation
- implement backpropagation training to train further the best MLP found by PSO