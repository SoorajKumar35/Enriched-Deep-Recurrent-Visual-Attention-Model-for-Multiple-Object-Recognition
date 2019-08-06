import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms


def one_hot(y, num_classes=10):
    """Change label at end of vectors of size 7 to be a one-hot vector"""
    N = len(y)
    y_where = y[:, :6]
    y_label = y[:, 6].astype(np.int64)
    y_label_one_hot = np.zeros((N, num_classes))
    y_label_one_hot[np.arange(N), y_label] = 1
    return np.concatenate((y_where, y_label_one_hot), axis=1)


def load_data(outfile="mnist_sequence3_sample_8distortions9x9"):
    X_train = np.load(outfile + "/X_train.npy")
    y_train = np.load(outfile + "/y_train.npy")
    X_valid = np.load(outfile + "/X_valid.npy")
    y_valid = np.load(outfile + "/y_valid.npy")
    X_test = np.load(outfile + "/X_test.npy")
    y_test = np.load(outfile + "/y_test.npy")
    #y_train, y_valid, y_test = one_hot(y_train), one_hot(y_valid), one_hot(y_test)
    return X_train, y_train, X_valid, y_valid, X_test, y_test


def prepare_datasets(X_train, y_train, X_valid, y_valid, X_test, y_test,
                     train_batch_size, val_batch_size, test_batch_size):

    X_train, y_train = torch.Tensor(X_train).reshape(-1, 1, 100, 100), torch.Tensor(y_train)
    X_valid, y_valid = torch.Tensor(X_valid).reshape(-1, 1, 100, 100), torch.Tensor(y_valid)
    X_test, y_test = torch.Tensor(X_test).reshape(-1, 1, 100, 100), torch.Tensor(y_test)


    # Normalization
        
    mu = X_train.mean()
    std = X_train.std()
    X_train = (X_train - mu) / std
    X_valid = (X_valid - mu) / std
    X_test = (X_test - mu) / std
    
    
    train_set = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size,
                                               shuffle=True, num_workers=0)
    val_set = torch.utils.data.TensorDataset(X_valid, y_valid)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=val_batch_size,
                                             shuffle=False, num_workers=0)
    test_set = torch.utils.data.TensorDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=test_batch_size,
                                              shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader
