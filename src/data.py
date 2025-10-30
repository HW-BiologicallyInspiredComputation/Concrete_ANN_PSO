import pandas as pd

def load_data():
    # Load the concrete data
    df = pd.read_csv("../data/concrete_data.csv")

    # Separate into train and test datasets
    train_df = df.sample(frac=0.7, random_state=42) # random state ensures we always get the same sample
    test_df = df.drop(train_df.index)

    # copy the target column from the input data into separate numpy arrays
    train_targets = train_df[' concrete_compressive_strength'].to_numpy()
    test_targets = test_df[' concrete_compressive_strength'].to_numpy()

    # drop the target column from the input data to create input feature arrays
    train_features = train_df.drop(columns=[' concrete_compressive_strength']).to_numpy()
    test_features = test_df.drop(columns=[' concrete_compressive_strength']).to_numpy()

    return (train_features, train_targets), (test_features, test_targets)

if __name__ == "__main__":
    (train_X, train_y), (test_X, test_y) = load_data()
    print("Training features shape:", train_X.shape)
    print("Training targets shape:", train_y.shape)
    print("Testing features shape:", test_X.shape)
    print("Testing targets shape:", test_y.shape)
