
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod
from scipy.interpolate import interp1d
from typing import Union
import argparse
from utility.survival import compute_unique_counts, make_monotonic

from utility.loss import mtlr_nll, cox_nll

from utility.torch_distributions import ParametrizedGaussian, ScaleMixtureGaussian, InverseGamma

class BayesianBaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def reset_parameters(self):
        pass

    @abstractmethod
    def log_prior(self):
        pass

    @abstractmethod
    def log_variational_posterior(self):
        pass

    def get_name(self):
        return self._get_name()

class BayesEleCox(BayesianBaseModel):
    def __init__(self, in_features: int, config: argparse.Namespace):
        super().__init__()
        if in_features < 1:
            raise ValueError("The number of input features must be at least 1")
        self.config = config
        self.in_features = in_features
        self.hidden_size = in_features
        self.time_bins = None
        self.cum_baseline_hazard = None
        self.baseline_survival = None
        self.l1 = BayesianElementwiseLinear(self.in_features, config)
        self.l2 = BayesianLinear(self.in_features, 1, config)

    def forward(self, x: torch.Tensor, sample: bool, n_samples) -> torch.Tensor:
        x = F.dropout(F.relu(self.l1(x, n_samples=n_samples)), p=self.config.dropout)
        outputs = self.l2(x, sample, n_samples)

        outputs = outputs.squeeze(dim=-1)
        return outputs

    def calculate_baseline_survival(self, x, t, e):
        outputs = self(x, sample=True, n_samples=self.config.n_samples_train).mean(dim=0)
        self.time_bins, self.cum_baseline_hazard, self.baseline_survival = baseline_hazard(outputs, t, e)

    def log_prior(self):
        return self.l1.log_prior + self.l2.log_prior

    def log_variational_posterior(self):
        return self.l1.log_variational_posterior + self.l2.log_variational_posterior

    def sample_elbo(
            self,
            x,
            t: torch.Tensor,
            e: torch.Tensor,
            dataset_size: int
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        num_batch = dataset_size / self.config.batch_size
        n_samples = self.config.n_samples_train
        outputs = self(x, sample=True, n_samples=n_samples)
        log_prior = self.log_prior() / n_samples
        log_variational_posterior = self.log_variational_posterior() / n_samples
        # remark if average is needed or not
        nll = cox_nll(outputs.mean(dim=0), t, e, model=self, C1=0)

        # Shouldn't here be batch_size instead?
        loss = (log_variational_posterior - log_prior) / (32 * dataset_size) + nll
        return loss, log_prior, log_variational_posterior / dataset_size, nll

    def reset_parameters(self):
        """Reinitialize the model."""
        self.l1.reset_parameters()
        self.l2.reset_parameters()
        return self

    def __repr__(self):
        return (f"{self.__class__.__name__}(in_features={self.in_features}, "
                f"hidden_size={self.hidden_size})")

class BayesianLinear(nn.Module):
    """
    Single linear layer of a mixture gaussian prior.
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            config: argparse.Namespace,
            use_mixture: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Scale to initialize weights
        self.config = config
        if self.config.mu_scale is None:
            self.weight_mu = nn.init.xavier_uniform_(nn.Parameter(torch.Tensor(out_features, in_features)))
        else:
            self.weight_mu = nn.init.uniform_(nn.Parameter(torch.Tensor(out_features, in_features)),
                                              -self.config.mu_scale, self.config.mu_scale)

        self.weight_rho = nn.Parameter(torch.ones([out_features, in_features]) * self.config.rho_scale)
        self.weight = ParametrizedGaussian(self.weight_mu, self.weight_rho)
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.zeros(1, out_features))
        self.bias_rho = nn.Parameter(torch.ones([1, out_features]) * self.config.rho_scale)
        self.bias = ParametrizedGaussian(self.bias_mu, self.bias_rho)
        # Prior distributions
        if use_mixture:
            pi = config.pi
        else:
            pi = 1
        self.weight_prior = ScaleMixtureGaussian(pi, config.sigma1, config.sigma2)
        self.bias_prior = ScaleMixtureGaussian(pi, config.sigma1, config.sigma2)

        # Initial values of the different parts of the loss function
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(
            self,
            x: torch.Tensor,
            sample: bool = True,
            n_samples: int = 1
    ):
        if self.training or sample:
            weight = self.weight.sample(n_samples=n_samples)
            bias = self.bias.sample(n_samples=n_samples)
        else:
            print("No sampling")
            weight = self.weight.mu.expand(n_samples, -1, -1)
            bias = self.bias.mu.expand(n_samples, -1, -1)

        if self.training:
            self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
            self.log_variational_posterior = self.weight.log_prob(weight) + self.bias.log_prob(bias)
        else:
            self.log_prior, self.log_variational_posterior = 0, 0

        # For a single layer network, x would have 2 dimension [n_data, n_feature]
        # But sometime x would be the sampled output from the previous layer,
        # which will have 3 dimension [n_samples, n_data, n_feature]
        n_data = x.shape[-2]
        bias = bias.repeat(1, n_data, 1)
        # If x is 3-d, this expand command will make x remains the same.
        x = x.expand(n_samples, -1, -1)
        # b: n_samples; i: n_data; j: input features size; k: output size
        return torch.einsum('bij,bkj->bik', x, weight) + bias

    def reset_parameters(self):
        """Reinitialize parameters"""
        nn.init.xavier_uniform_(self.weight_mu)
        nn.init.constant_(self.weight_rho, self.config.rho_scale)
        nn.init.constant_(self.bias_mu, 0)
        nn.init.constant_(self.bias_rho, self.config.rho_scale)
        self.weight = ParametrizedGaussian(self.weight_mu, self.weight_rho)
        self.bias = ParametrizedGaussian(self.bias_mu, self.bias_rho)

class BayesianElementwiseLinear(nn.Module):
    """
    Single elementwise linear layer of a mixture gaussian prior.
    """

    def __init__(
            self,
            input_output_size: int,
            config: argparse.Namespace
    ):
        """
        Initialize gaussian layer using reparameterization.

        :param input_output_size: number of input features
        :param config: hyperparameters
        """
        super().__init__()
        self.input_output_size = input_output_size
        self.config = config
        if self.config.mu_scale is None:
            self.config.mu_scale = 1. * np.sqrt(6. / input_output_size)

        self.weight_mu = nn.init.uniform_(nn.Parameter(torch.Tensor(input_output_size)),
                                          -self.config.mu_scale, self.config.mu_scale)
        self.weight_rho = nn.Parameter(torch.ones([input_output_size]) * self.config.rho_scale)
        self.weight = ParametrizedGaussian(self.weight_mu, self.weight_rho)

        self.weight_prior = ScaleMixtureGaussian(config.pi, config.sigma1, config.sigma2)

        # Initial values of the different parts of the loss function
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(
            self,
            x: torch.tensor,
            sample: bool = True,
            n_samples: int = 1
    ):
        if self.training or sample:
            weight = self.weight.sample(n_samples=n_samples)
        else:
            print("No sampling")
            weight = self.weight.mu.expand(n_samples, -1, -1)

        if self.training:
            self.log_prior = self.weight_prior.log_prob(weight)
            self.log_variational_posterior = self.weight.log_prob(weight)
        else:
            self.log_prior, self.log_variational_posterior = 0, 0

        # b: n_samples; i: n_data; j: input output size; k: input output size
        weight = torch.einsum('bj, jk->bjk', weight,
                              torch.eye(weight.shape[1], dtype=weight.dtype, device=weight.device))
        x = x.expand(n_samples, -1, -1)
        return torch.einsum('bij,bjk->bik', x, weight)

    def reset_parameters(self):
        """Reinitialize parameters"""
        nn.init.uniform_(self.weight_mu, -self.config.mu_scale, self.config.mu_scale)
        nn.init.constant_(self.weight_rho, self.config.rho_scale)
        self.weight = ParametrizedGaussian(self.weight_mu, self.weight_rho)

class BayesCox(BayesEleCox):
    def __init__(self, in_features: int, config: argparse.Namespace):
        super(BayesEleCox, self).__init__()
        if in_features < 1:
            raise ValueError("The number of input features must be at least 1")
        self.config = config
        self.in_features = in_features
        self.hidden_size = config.hidden_size
        self.time_bins = None
        self.cum_baseline_hazard = None
        self.baseline_survival = None
        self.l1 = BayesianLinear(self.in_features, self.hidden_size, config)
        self.l2 = BayesianLinear(self.hidden_size, 1, config)
        
def baseline_hazard(
        logits: torch.Tensor,
        time: torch.Tensor,
        event: torch.Tensor
) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    """
    Calculate the baseline cumulative hazard function and baseline survival function using Breslow estimator
    :param logits: logit outputs calculated from the Cox-based network using training data.
    :param time: Survival time of training data.
    :param event: Survival indicator of training data.
    :return:
    uniq_times: time bins correspond of the baseline hazard/survival.
    cum_baseline_hazard: cumulative baseline hazard
    baseline_survival: baseline survival curve.
    """
    risk_score = torch.exp(logits)
    order = torch.argsort(time)
    risk_score = risk_score[order]
    uniq_times, n_events, n_at_risk, _ = compute_unique_counts(event, time, order)

    divisor = torch.empty(n_at_risk.shape, dtype=torch.float, device=n_at_risk.device)
    value = torch.sum(risk_score)
    divisor[0] = value
    k = 0
    for i in range(1, len(n_at_risk)):
        d = n_at_risk[i - 1] - n_at_risk[i]
        value -= risk_score[k:(k + d)].sum()
        k += d
        divisor[i] = value

    assert k == n_at_risk[0] - n_at_risk[-1]

    hazard = n_events / divisor
    # Make sure the survival curve always starts at 1
    if 0 not in uniq_times:
        uniq_times = torch.cat([torch.tensor([0]).to(uniq_times.device), uniq_times], 0)
        hazard = torch.cat([torch.tensor([0]).to(hazard.device), hazard], 0)
    # TODO: torch.cumsum with cuda array will generate a non-monotonic array. Need to update when torch fix this bug
    # See issue: https://github.com/pytorch/pytorch/issues/21780
    cum_baseline_hazard = torch.cumsum(hazard.cpu(), dim=0).to(hazard.device)
    baseline_survival = torch.exp(- cum_baseline_hazard)
    if baseline_survival.isinf().any():
        print(f"Baseline survival contains \'inf\', need attention. \n"
              f"Baseline survival distribution: {baseline_survival}")
        last_zero = torch.where(baseline_survival == 0)[0][-1].item()
        baseline_survival[last_zero + 1:] = 0
    baseline_survival = make_monotonic(baseline_survival)
    return uniq_times, cum_baseline_hazard, baseline_survival

class BayesLinCox(BayesianBaseModel):
    def __init__(self, in_features: int, config: argparse.Namespace):
        super().__init__()
        if in_features < 1:
            raise ValueError("The number of input features must be at least 1")
        self.config = config
        self.in_features = in_features
        self.time_bins = None
        self.cum_baseline_hazard = None
        self.baseline_survival = None

        self.l1 = BayesianLinear(self.in_features, 1, config)

    def forward(self, x: torch.Tensor, sample: bool, n_samples) -> torch.Tensor:
        outputs = self.l1(x, sample, n_samples)
        return outputs

    def calculate_baseline_survival(self, x, t, e):
        outputs = self.forward(x, sample=True, n_samples=self.config.n_samples_train).mean(dim=0)
        self.time_bins, self.cum_baseline_hazard, self.baseline_survival = baseline_hazard(outputs, t, e)

    def log_prior(self):
        return self.l1.log_prior

    def log_variational_posterior(self):
        return self.l1.log_variational_posterior

    def sample_elbo(self, x, t, e, dataset_size) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        n_samples = self.config.n_samples_train
        outputs = self(x, sample=True, n_samples=n_samples)
        log_prior = self.log_prior() / n_samples
        log_variational_posterior = self.log_variational_posterior() / n_samples
        # remark if average is needed or not
        nll = cox_nll(outputs.mean(dim=0), t, e, model=self, C1=0)

        # Shouldn't here be batch_size instead?
        loss = (log_variational_posterior - log_prior) / (dataset_size) + nll
        return loss, log_prior, log_variational_posterior, nll

    def reset_parameters(self):
        self.l1.reset_parameters()
        return self

    def __repr__(self):
        return f"{self.__class__.__name__}(in_features={self.in_features}"