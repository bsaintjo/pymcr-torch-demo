import itertools
from typing import Optional, Any
from pymcr import LinearRegression
import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn


class TorchLeastSquares(LinearRegression):
    """
    Simple example class using pytorch to create a model compatible with pyMCR.

    This inherits from the LinearRegression abstract base class. and needs to implement the
    fit abstract method that takes two arguments A and B. The fit method should store the result
    from solving Ax=B into self.X_.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.rank = None
        self.sv = None

    def fit(self, A: NDArray, B: NDArray):
        """ "
        Solve Ax=B using pytorch least squares. A and B are numpy arrays, so must be converted to pytorch Tensors.

        Once X is solved for the result needs to be converted back into numpy arrays and stored in self.X_
        """
        A = torch.from_numpy(A)
        B = torch.from_numpy(B)

        X_, residuals, self.rank, self.sv = torch.linalg.lstsq(A, B)

        self.X_ = X_.numpy()
        self.residuals = residuals.numpy()


class Autoencoder(nn.Module):
    def __init__(self, n_features: int):
        super(Autoencoder, self).__init__()

        self.ac = torch.nn.ReLU()
        self.fc1 = nn.Linear(n_features, 128, dtype=torch.double)
        self.fc2 = nn.Linear(128, 128, dtype=torch.double)
        self.fc3 = nn.Linear(128, n_features, dtype=torch.double)

    def forward(self, x):
        x = self.ac(self.fc1(x))  # activation function for hidden layer
        x = self.ac(self.fc2(x))
        x = self.fc3(x)
        return x


class AutoEncoderAxB(LinearRegression):
    """
    Simple example class using pytorch to create a model compatible with pyMCR.

    This inherits from the LinearRegression abstract base class. and needs to implement the
    fit abstract method that takes two arguments A and B. The fit method should store the result
    from solving Ax=B into self.X_.
    """

    def __init__(self):
        super().__init__()
        self.x_ = torch.randn(2301, 2, dtype=torch.double, requires_grad=True)

        self.net = Autoencoder(n_features=2301)
        self.criterion = nn.MSELoss()

        # Changing to a smaller learning rate will shift the predicted spectra a lot
        self.optimizer = torch.optim.Adam(
            itertools.chain([self.x_], self.net.parameters()), lr=0.001
        )

    def fit(self, A: NDArray, B: NDArray):
        """ "
        Solve Ax=B using pytorch least squares. A and B are numpy arrays, so must be converted to pytorch Tensors.

        Once X is solved for the result needs to be converted back into numpy arrays and stored in self.X_
        """
        A = torch.from_numpy(A)
        B = torch.from_numpy(B)

        self.net.train()
        for _ in range(1000):
            x = self.net(self.x_.T)

            Axb = torch.matmul(A, x)
            loss = self.criterion(Axb, B)
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

        self.net.eval()
        x = self.net(self.x_.T)
        self.X_ = x.detach().numpy().astype(np.float64)


class TorchGrad(LinearRegression):
    """
    Simple example class using pytorch to create a model compatible with pyMCR.

    This inherits from the LinearRegression abstract base class. and needs to implement the
    fit abstract method that takes two arguments A and B. The fit method should store the result
    from solving Ax=B into self.X_.
    """

    def __init__(
        self,
        lr: float = 0.01,
        n_iter: int = 1000,
        criterion: Optional[Any] = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.x_ = torch.randn(2, 2301, dtype=torch.double, requires_grad=True)
        if criterion is None:
            self.criterion = nn.MSELoss()
        else:
            self.criterion = criterion
        self.n_iter = n_iter

        # Changing to a smaller learning rate will shift the predicted spectra a lot
        self.optimizer = torch.optim.Adam([self.x_], lr=lr)

    def fit(self, A: NDArray, B: NDArray):
        """ "
        Solve Ax=B using pytorch least squares. A and B are numpy arrays, so must be converted to pytorch Tensors.

        Once X is solved for the result needs to be converted back into numpy arrays and stored in self.X_
        """
        A = torch.from_numpy(A)
        B = torch.from_numpy(B)

        # self.net.train()
        for _ in range(self.n_iter):
            Ax = torch.matmul(A, self.x_)
            loss = self.criterion(Ax, B)
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

        # self.net.eval()
        self.X_ = self.x_.data.detach().numpy().astype(np.float64)
