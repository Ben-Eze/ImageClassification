import torch
import sklearn.datasets, sklearn.model_selection


def get_MNIST(TEST_SIZE):
    X_np, y_np = sklearn.datasets.load_digits(return_X_y=True)  # 1797 data pts
    X = torch.from_numpy(X_np).type(torch.float32)
    y = torch.from_numpy(y_np).type(torch.int64)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=TEST_SIZE, random_state=0
    )
    return X, X_train, X_test, \
           y, y_train, y_test