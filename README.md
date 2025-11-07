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

- ~~[ ] using crossvalidation to evaluate model performance~~
- ~~[ ] using a validation dataset~~
- [-] make output proportional by dividing by max value of target in training set

### PSO

- ~~[ ] change the activation function~~
- [x] change the topology of the input ANN (number of layers, number of neurons per layer)

### Error evaluation

- [-] implement other error metrics (MSE, RMSE, R2)

### Informants selection
- [-] implement different informants selection strategies (radius-based, **k-nearest neighbors**, random every iteration)

### ANN Backpropagation
- ~~[ ] implement backpropagation training to train further the best MLP found by PSO~~

### Sequential Model

- [x] implement Sequential model to have variable number of layers and neurons and activation functions

### Same velocity as position to explore far

### Clip velocity and position
- ~~[] bounce back at the boundaries~~
- [x] reset position/velocity when out of bounds

### Accuracy

- [x] implement accuracy metric for regression

### Genetic Algorithm
- [x] Limit training with an amount of time instead of epochs
- [x] Don't overtrain, stop training if the loss has not decreased for 20 epochs
