import torch
import numpy as np
import sklearn.datasets, sklearn.model_selection


def get_data(dataset="digits", reshape=None, test_size=0.2):
    if dataset=="digits":
        # 1797 data pts, 8x8 images
        X_np, y_np = sklearn.datasets.load_digits(return_X_y=True)
        X = torch.from_numpy(X_np).type(torch.float32)
        y = torch.from_numpy(y_np).type(torch.int64)

    else:
        data = sklearn.datasets.fetch_openml(dataset, version=1, cache=True)
        X_np, y_np = data["data"], data["target"]
        X = torch.from_numpy(X_np.to_numpy().astype(float)).type(torch.float32)
        y = torch.from_numpy(y_np.to_numpy().astype(int)).type(torch.int64)

    if reshape is not None:
        X = X.reshape(reshape)
    
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=test_size, random_state=0
    )

    print("Data loaded successfully")

    return X, X_train, X_test, \
           y, y_train, y_test