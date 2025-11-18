# W9 & 10 - Little Changes and Tests

Table of Contents


## 1. Previously in week 8

We added a Genetic Algorithm (GA) to optimize the hyperparameters of our Artificial Neural Network (ANN) for predicting concrete's compressive strength and a web interface to train online and visualize results.

## 2. Little Changes

After sitting together some times and discussing the project, we decided on some things to imrpove and had a look back at the going further steps planned for the project.
Fortunately, we had written them down to not forget them!

We crossed out some of the steps that we had already implemented, and we also removed some others that we considered not necessary for the final delivery of the project.

```
## Going further

### Dataset

- ~~~ using crossvalidation to evaluate model performance
- ~~~ using a validation dataset
- [x] make output proportional by dividing by max value of target in training set

### PSO
- [x] change the activation function
- [x] change the topology of the input ANN (number of layers, number of neurons per layer)

### Error evaluation
- [ ] implement other error metrics (MSE, RMSE, R2)

### Informants selection
- [ ] implement different informants selection strategies (radius-based, **k-nearest neighbors**, random every iteration)

### ANN Backpropagation
- ~~~ implement backpropagation training to train further the best MLP found by PSO

### Sequential Model

- [x] implement Sequential model to have variable number of layers and neurons and activation functions

### Clip velocity and position
- ~~~ bounce back at the boundaries
- [x] reset position/velocity when out of bounds

### Accuracy
- [x] implement accuracy metric

### Genetic Algorithm
- [x] Limit training with an amount of time instead of epochs
- [x] Don't overtrain, stop training if the loss has not decreased for 20 epochs

### Graphs
- [ ] plot fitness over hyperparameters (number of particles, inertia weight, cognitive and social coefficients, number of informants, topology)
- [ ] plot two hyperparameters to compare them:  eg. plot cognitive over social with point size for fitnesses
```

What we had already implemented is marked with an 'x':
- Changed the activation function. Indeed, we can already choose between 'relu', 'tanh' and 'sigmoid' activation functions.
- Changed the topology of the input ANN (number of layers, number of neurons per layer). This is done in the GA part.
- Implemented a Sequential model to have variable number of layers and neurons and activation functions. This greatly helped with the previous point.
- Reset position/velocity when out of bounds.
- Implemented accuracy metric for regression.
- Limited training with an amount of time instead of epochs in the GA part.
- Added early stopping to the GA part to avoid overtraining.

What we decided to not implement is marked with '~~~':
- Using crossvalidation to evaluate model performance. (too time consuming and would increase the computing time a lot)
- Implement backpropagation training to train further the best MLP found by PSO. (as we train enough with PSO already, we think that this would not bring much improvement for the time spent)
- Bounce back at the boundaries for clipping velocity and position. (we think that resetting position/velocity when out of bounds is sufficient for our case)

What we still wanted to implement is marked with '[ ]':
- Make output proportional by dividing by max value of target in training set.
- Implement other error metrics (MSE, RMSE, R2).
- Implement different informants selection strategies: k-nearest neighbors, radius-based, random every iteration.

### 2.1 Output Proportionality

To make the output proportional, we divided the target values by the maximum value of the target in the training set. This helps in normalizing the output and can improve the performance of the model.

### 2.2 Other Error Metrics

We implemented additional error metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R2) to evaluate the performance of our model more comprehensively.

### 2.3 Informants Selection Strategies

We implemented k-nearest neighbors as an informants selection strategy. This allows particles to consider the best positions of their k-nearest neighbors when updating their velocities, potentially leading to better exploration of the search space.

This change we really needed because over the last 3 multiday runs online we noticed that the best PSO always had a social weight of 0, meaning that particles were not using informants at all. By implementing k-nearest neighbors, we hope to improve the performance of the PSO algorithm and have a social that is actually used.

### 2.4 Parallel Processing

To speed up the evaluation of particles in the PSO algorithm, we implemented parallel processing. The idea was to parallelise the training of POSs in each generation and thus reduce the overall computation time significantly.

However, when we tried to use it with streamlit for the online training and visualization, we encountered some issues. Streamlit does not play well with multiprocessing due to its architecture, which can lead to unexpected behavior and crashes.

Thus, we decided to leave it to avoid losing more time on it.

## 3. Fixes

While running and reviewing our code over the past weeks, we noticed some issues popping up here and there. We fixed them as we went along.

### 3.1 Architecture limitation breached in GA crossover

One of our best PSO configurations found by the GA had a topology with 7 layers although we had set a maximum of 3 layers. After investigation we noticed that the problem came from the way we crossed over individuals in the GA. Indeed, when crossing over two individuals we would split their topology lists in two parts randomly and concatenate the first part of the first parent with the second part of the second parent. This could lead to a child topology longer than the maximum allowed length.

This was the crossover code in class PsoGenome before the fix:
```
if random.random() < crossover_rate:
    # one-point crossover on layer lists
    la, lb = list(a.ann_layers), list(b.ann_layers)
    cut_a = random.randrange(len(la))
    cut_b = random.randrange(len(lb))
    new_layers = tuple(la[:cut_a] + lb[cut_b:])
    child.ann_layers = new_layers
```

In this code if we had for exemple a.ann_layers = (10, 20, 30) and b.ann_layers = (15, 25, 35) and if cut_a = 2 and cut_b = 1, the new_layers would be (10, 20, 25, 35) which has a length of 4, greater than the maximum allowed of 3.

To fix this, we modified the crossover code to ensure that the resulting child topology does not exceed the maximum allowed length. The solution was simply to have one cut position that is the same for both parents. Here is the updated code:

```
if random.random() < crossover_rate:
    # one-point crossover on layer lists
    la, lb = list(a.ann_layers), list(b.ann_layers)
    cut = random.randrange(len(la))
    new_layers = tuple(la[:cut] + lb[cut:])
    child.ann_layers = new_layers
```

### 3.2 Accuracy problems

#### 3.2.1 Final accuracy of trained PSO

We noticed that we did not only take the final accuracy of the PSO. We also copied the accuracy every 10 iterations of its training and added them to the accuracies list. Took also during training and calculated the average accuracy on all of them.

This leads to two possibilties:
- Either the final accuracy we print which is supposed to represent the accuracy of the trained PSO is not the average of the 3 runs' final accuracies but an average accuracy over the training process as well. As the accuracy usually increases during training, this would add lots of lower accuracies to the average and thus lower the final accuracy printed.
- Or we recalculated it somewhere else and thus the printed final accuracy is correct.

#### The cache wasn't transferred from one run to another in GA evaluation

When evaluating individuals in the GA, we used a cache to store the results of previously evaluated PSO configurations to avoid redundant computations. However, we realized that this cache was not being transferred from one run to another, leading to unnecessary re-evaluations of the same configurations.

This produces a significant slowdown in the GA evaluation process, especially when the same configurations are evaluated multiple times across different runs. And it also reduces the effectiveness of the GA in exploring the search space efficiently.
Indeed, keeping the cache between runs allows us to build upon previous evaluations, making the solutions found more reliable and consistent. The more they are evaluated, the more reliable the results are.