# PSO Hyperparameters:

Run 10 times per combination of hyperparameters

1. Swarm size

2. Acceleration Coefficients
    a. Inertial weight
    b. Cognitial weight
    c. Social weight
    d. Best particle weight
    e. Jump size

2. Informants
    a. number
    b. strategy
        i. random
        ii. k-nearest

4. Boundary limitations
    a. reset random
    b. reset center
    c. bounce off (not for us)

5. Velocity limitations
    a. Maximum velocity (Vmax)
    b. Velocity clamping strategy

6. Convergence / stopping criteria
    a. Number of iterations
    b. Target error
    c. Stagnation threshold

7. Initialization parameters
    a. Position initialization distribution
    b. Velocity initialization scale

8. Error measurement (fitness function)
    a. MAE
    b. MSE
    c. RMSE
    d. R2


# ANN Hyperparameters:

1. Topologie
    a. layers
    b. neurons
    c. Activation functions (ReLU, tanh, etc.)

3. Weight initialization method
