import pandas as pd

# Function to load and preprocess the concrete data


def load_data(path: str = "./data/concrete_data.csv"):
    # Load the concrete data
    df = pd.read_csv(path)

    # Separate into train and test datasets
    # random state ensures we always get the same sample
    train_df = df.sample(frac=0.7, random_state=42)
    test_df = df.drop(train_df.index)

    # copy the target column from the input data into separate numpy arrays
    train_targets = train_df[" concrete_compressive_strength"].to_numpy()
    test_targets = test_df[" concrete_compressive_strength"].to_numpy()

    # drop the target column from the input data to create input feature arrays
    train_features = train_df.drop(
        columns=[" concrete_compressive_strength"]
    ).to_numpy()
    test_features = test_df.drop(columns=[" concrete_compressive_strength"]).to_numpy()

    # make output proportional by dividing by max value of target in training set
    # We noticed that our predicted values from the first itaration were always way smaller than the actual values.
    # This normalization step helps the model to predict values on a similar scale as the actual values
    # This is important for model convergence and performance
    max_target_value = train_targets.max()
    train_targets = train_targets / max_target_value
    test_targets = test_targets / max_target_value

    return (train_features, train_targets), (test_features, test_targets)


if __name__ == "__main__":
    (train_X, train_y), (test_X, test_y) = load_data()
    print("Training features shape:", train_X.shape)
    print("Training targets shape:", train_y.shape)
    print("Testing features shape:", test_X.shape)
    print("Testing targets shape:", test_y.shape)
