from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from pymcr.regressors import LinearRegression
from pyro.infer.autoguide import AutoDiagonalNormal
import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
from torchtyping import TensorType


class Encoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, z_dim: int):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.z_dim = z_dim

        self.fc_mean = nn.Linear(in_features=hidden_size, out_features=z_dim)
        self.fc_std = nn.Linear(in_features=hidden_size, out_features=z_dim)

        self.fc1 = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.relu = nn.ReLU()

    def forward(
        self,
        x: TensorType["batch", "input_size"],  # noqa: F821
    ) -> Tuple[TensorType["batch", "z_dim"], TensorType["batch", "z_dim"]]:  # noqa: F821
        x = self.relu(self.fc1(x))
        z_loc = self.fc_mean(x)
        z_scale = torch.exp(self.fc_std(x))

        return z_loc, z_scale


class Decoder(nn.Module):
    def __init__(self, output_size: int, hidden_size: int, z_dim: int):
        super().__init__()

        self.fc_mean_std = nn.Linear(in_features=z_dim, out_features=hidden_size)

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(hidden_size, output_size)

    def forward(
        self,
        z: TensorType["batch", "z_dim"],  # noqa: F821
    ) -> TensorType["batch", "output_size"]:  # noqa: F821
        hidden = self.relu(self.fc_mean_std(z))
        loc_out = self.fc1(hidden)

        return loc_out


class VAE(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        z_dim: int,
        use_gpu: bool = False,
        encoder: Optional[nn.Module] = None,
        decoder: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.z_dim = z_dim
        self.use_gpu = use_gpu

        if encoder is None:
            self.encoder = Encoder(
                input_size=input_size, hidden_size=hidden_size, z_dim=self.z_dim
            )
        else:
            self.encoder = encoder

        if decoder is None:
            self.decoder = Decoder(
                output_size=input_size, hidden_size=hidden_size, z_dim=self.z_dim
            )
        else:
            self.decoder = decoder

        if use_gpu:
            if torch.backends.mps.is_available():
                torch.set_default_device("mps")
            elif torch.cuda.is_available():
                torch.set_default_device("cuda")
            else:
                print("GPU is not available")

    def model(self, A: TensorType, B: TensorType):
        pyro.module("encoder", self.encoder)
        pyro.module("decoder", self.decoder)

        x = pyro.sample(
            "x",
            dist.Normal(
                torch.zeros(A.shape[1], B.shape[1]), torch.ones(B.shape[1])
            ).to_event(2),
        )
        z_loc, z_scale = self.encoder(x)

        # Sample values according the latent vector
        z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(2))

        loc_out = pyro.deterministic("x_recon", self.decoder(z))
        Axb = pyro.deterministic("A @ x_recon", torch.matmul(A, loc_out))
        sigma = pyro.sample("sigma", dist.Uniform(1.0, 10.0))
        pyro.sample(
            "B",
            dist.Normal(loc=Axb, scale=sigma, validate_args=True).to_event(2),
            obs=B,
        )

        return loc_out

    def reconstruct_spectra(self, X):
        z_loc, z_scale = self.encoder(X)
        z = dist.Normal(z_loc, z_scale).sample()
        loc = self.decoder(z)

        return loc


class McrVae(LinearRegression):
    """The pyMCR regressor using variational autoencoders for performing multivariate curve resolution - alternating regression."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        z_dim: int,
        learning_rate: float = 1e-4,
    ):
        super().__init__()
        pyro.clear_param_store()
        self.vae = VAE(input_size=input_size, hidden_size=hidden_size, z_dim=z_dim)
        self.optimizer = pyro.optim.Adam(optim_args={"lr": learning_rate})
        self.guide = AutoDiagonalNormal(self.vae.model)
        self.svi = pyro.infer.SVI(
            model=self.vae.model,
            guide=self.guide,
            optim=self.optimizer,
            loss=pyro.infer.Trace_ELBO(),
        )

    def fit(self, A: NDArray, B: NDArray):
        """ "
        Solve Ax=B using Variational Autoencoders using Pyro. A and B are numpy arrays, so must be converted to pytorch Tensors.

        Once X is solved for the result needs to be converted back into numpy arrays and stored in self.X_
        """
        A = torch.from_numpy(A.astype(np.float32))
        B = torch.from_numpy(B.astype(np.float32))

        epoch_loss = 0
        for _ in range(1000):
            loss = self.svi.step(A, B)
            epoch_loss += loss

        print("epoch loss", epoch_loss)
        print("A, B shape = ", A.shape, B.shape)
        guide_dict = self.guide(A, B)
        latent = guide_dict["latent"]
        x = self.vae.decoder(latent)
        self.X_ = x.detach().numpy().astype(np.float64)
