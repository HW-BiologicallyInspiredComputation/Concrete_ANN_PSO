# W6 - Part 1 - Data preprocessing

Table of Contents

- [1. Reading the coursework's instructions](#1-reading-the-courseworks-instructions)
- [2. Understanding the dataset](#2-understanding-the-dataset)
- [3. Next Steps](#3-next-steps)


## 1. Reading the coursework's instructions

We first started by reading the coursework's instructions carefully to understand the requirements and expectations.
This helped us to plan our approach and ensure that we met all the necessary criteria.

The goal of this coursework is to create an Artificial Neural Network (ANN) that can accurately predict concrete's compressive strength based on its ingredients and age.
In order to improve the performance of our ANN, we are required to implement a Particle Swarm Optimization (PSO) algorithm.
The PSO should optimize the hyperparameters of the network.

## 2. Understanding the dataset

After reading the instructions, we proceeded to understand the dataset provided for the coursework.
The dataset contains various features related to the ingredients of concrete, such as cement, blast furnace slag, fly ash, water, superplasticizer, coarse aggregate, fine aggregate, and age.
The target variable is the compressive strength of the concrete.

![Concrete Compressive Strength Data Set](img/W6_1_Dataset.png)

Using the Pandas library, we loaded the dataset and split it in a way that 70% of the data is used for training and 30% for testing.
For the splitting we used the following code:

```
train_df = df.sample(frac=0.7, random_state=42)
test_df = df.drop(train_df.index)
```

This ensured that we got 70% of teh data fot the training set but also that the training set would be randomly selected depending on the random state.
This way, we can reproduce the same split in future runs by using the same random state. This is important for consistency and comparability of results.

## 3. Next Steps

With the dataset understood and split into training and testing sets, we are now ready to proceed with the next steps of the coursework.
Next week we will focus on implementing the ANN and potentially the PSO algorithm to optimize its hyperparameters.