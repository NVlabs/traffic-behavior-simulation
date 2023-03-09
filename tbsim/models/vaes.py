"""Variants of Conditional Variational Autoencoder (C-VAE)"""
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from tbsim.utils.loss_utils import KLD_0_1_loss, KLD_gaussian_loss, KLD_discrete
from tbsim.utils.torch_utils import reparameterize
from tbsim.models.base_models import MLP, SplitMLP
import tbsim.utils.tensor_utils as TensorUtils
from torch.distributions import Categorical, kl_divergence


class Prior(nn.Module):
    def __init__(self, latent_dim, input_dim=None, device=None):
        """
        A generic prior class
        Args:
            latent_dim: dimension of the latent code (e.g., mu, logvar)
            input_dim: (Optional) dimension of the input feature vector, for conditional prior
            device:
        """
        super(Prior, self).__init__()
        self._latent_dim = latent_dim
        self._input_dim = input_dim
        self._net = None
        self._device = device

    def forward(self, inputs: torch.Tensor = None):
        """
        Get a batch of prior parameters.

        Args:
            inputs (torch.Tensor): (Optional) A feature vector for priors that are input-conditional.

        Returns:
            params (dict): A dictionary of prior parameters with the same batch size as the inputs (1 if inputs is None)
        """
        raise NotImplementedError

    @staticmethod
    def get_mean(prior_params):
        """
        Extract the "mean" of a prior distribution (not supported by all priors)

        Args:
            prior_params (torch.Tensor): a batch of prior parameters

        Returns:
            mean (torch.Tensor): the "mean" of the distribution
        """
        raise NotImplementedError

    def sample(self, n: int, inputs: torch.Tensor = None):
        """
        Take samples with the prior distribution

        Args:
            n (int): number of samples to take
            inputs (torch.Tensor): (Optional) A feature vector for priors that are input-conditional.

        Returns:
            samples (torch.Tensor): a batch of latent samples with shape [input_batch_size, n, latent_dim]
        """
        prior_params = self.forward(inputs=inputs)
        return self.sample_with_parameters(prior_params, n=n)

    @staticmethod
    def sample_with_parameters(params: dict, n: int):
        """
        Take samples using given a batch of distribution parameters, e.g., mean and logvar of a unit Gaussian

        Args:
            params (dict): a batch of distribution parameters
            n (int): number of samples to take

        Returns:
            samples (torch.Tensor): a batch of latent samples with shape [param_batch_size, n, latent_dim]
        """
        raise NotImplementedError

    def kl_loss(self, posterior_params: dict, inputs: torch.Tensor = None) -> torch.Tensor:
        """
        Compute kl loss between the prior and the posterior distributions.

        Args:
            posterior_params (dict): a batch of distribution parameters
            inputs (torch.Tensor): (Optional) A feature vector for priors that are input-conditional.

        Returns:
            kl_loss (torch.Tensor): kl divergence value
        """
        raise NotImplementedError

    @property
    def posterior_param_shapes(self) -> dict:
        """
        Shape of the posterior parameters

        Returns:
            shapes (dict): a dictionary of parameter shapes
        """
        raise NotImplementedError

    @property
    def latent_dim(self):
        """
        Shape of the latent code

        Returns:
            latent_dim (int)
        """
        return self._latent_dim


class FixedGaussianPrior(Prior):
    """An unassuming unit Gaussian Prior"""
    def __init__(self, latent_dim, input_dim=None, device=None):
        super(FixedGaussianPrior, self).__init__(
            latent_dim=latent_dim, input_dim=input_dim, device=device)
        self._params = nn.ParameterDict({
            "mu": nn.Parameter(data=torch.zeros(self._latent_dim), requires_grad=False),
            "logvar": nn.Parameter(data=torch.zeros(self._latent_dim), requires_grad=False)
        })

    @staticmethod
    def get_mean(prior_params):
        return prior_params["mu"]

    def forward(self, inputs: torch.Tensor = None):
        """
        Get a batch of prior parameters.

        Args:
            inputs (torch.Tensor): (Optional) A feature vector for priors that are input-conditional.

        Returns:
            params (dict): A dictionary of prior parameters with the same batch size as the inputs (1 if inputs is None)
        """

        batch_size = 1 if inputs is None else inputs.shape[0]
        params = TensorUtils.unsqueeze_expand_at(self._params, size=batch_size, dim=0)
        return params

    @staticmethod
    def sample_with_parameters(params, n: int):
        """
        Take samples using given a batch of distribution parameters, e.g., mean and logvar of a unit Gaussian

        Args:
            params (dict): a batch of distribution parameters
            n (int): number of samples to take

        Returns:
            samples (torch.Tensor): a batch of latent samples with shape [param_batch_size, n, latent_dim]
        """

        batch_size = params["mu"].shape[0]
        params_tiled = TensorUtils.repeat_by_expand_at(params, repeats=n, dim=0)
        samples = reparameterize(params_tiled["mu"], params_tiled["logvar"])
        samples = TensorUtils.reshape_dimensions(samples, begin_axis=0, end_axis=1, target_dims=(batch_size, n))
        return samples

    def kl_loss(self, posterior_params, inputs=None):
        """
        Compute kl loss between the prior and the posterior distributions.

        Args:
            posterior_params (dict): a batch of distribution parameters
            inputs (torch.Tensor): (Optional) A feature vector for priors that are input-conditional.

        Returns:
            kl_loss (torch.Tensor): kl divergence value
        """

        assert posterior_params["mu"].shape[1] == self._latent_dim
        assert posterior_params["logvar"].shape[1] == self._latent_dim
        return KLD_0_1_loss(
            mu=posterior_params["mu"],
            logvar=posterior_params["logvar"]
        )

    @property
    def posterior_param_shapes(self) -> OrderedDict:
        return OrderedDict(mu=(self._latent_dim,), logvar=(self._latent_dim,))


class ConditionalCategoricalPrior(Prior):
    """
    A class that holds functionality for learning categorical priors for use
    in VAEs.
    """
    def __init__(self, latent_dim, input_dim=None, device=None):
        """
        Args:
            latent_dim (int): size of latent dimension for the prior
            device (torch.Device): where the module should live (i.e. cpu, gpu)
        """
        super(ConditionalCategoricalPrior, self).__init__(latent_dim=latent_dim, input_dim=input_dim, device=device)
        assert isinstance(input_dim, int) and input_dim > 0
        self.device = device
        self._latent_dim = latent_dim
        self._prior_net = MLP(input_dim=input_dim, output_dim=latent_dim)

    def sample(self, n: int, inputs: torch.Tensor = None):
        """
        Returns a batch of samples from the prior distribution.
        Args:
            n (int): this argument is used to specify the number
                of samples to generate from the prior.
            inputs (torch.Tensor): conditioning feature for prior
        Returns:
            z (torch.Tensor): batch of sampled latent vectors.
        """

        # check consistency between n and obs_dict
        if self.learnable:

            # forward to get parameters
            out = self.forward(batch_size=n, obs_dict=obs_dict, goal_dict=goal_dict)
            prior_logits = out["logit"]

            # sample one-hot latents from categorical distribution
            dist = Categorical(logits=prior_logits)
            z = TensorUtils.to_one_hot(dist.sample(), num_class=self.categorical_dim)

        else:
            # try to include a categorical sample for each class if possible (ensuring rough uniformity)
            if (self.latent_dim == 1) and (self.categorical_dim <= n):
                # include samples [0, 1, ..., C - 1] and then repeat until batch is filled
                dist_samples = torch.arange(n).remainder(self.categorical_dim).unsqueeze(-1).to(self.device)
            else:
                # sample one-hot latents from uniform categorical distribution for each latent dimension
                probs = torch.ones(n, self.latent_dim, self.categorical_dim).float().to(self.device)
                dist_samples = Categorical(probs=probs).sample()
            z = TensorUtils.to_one_hot(dist_samples, num_class=self.categorical_dim)

        # reshape [B, D, C] to [B, D * C] to be consistent with other priors that return flat latents
        z = z.reshape(*z.shape[:-2], -1)
        return z

    def kl_loss(self, posterior_params, inputs = None):
        """
        Computes KL divergence loss between the Categorical distribution
        given by the unnormalized logits @logits and the prior distribution.
        Args:
            posterior_params (dict): dictionary with key "logits" corresponding
                to torch.Tensor batch of unnormalized logits of shape [B, D * C]
                that corresponds to the posterior categorical distribution
        Returns:
            kl_loss (torch.Tensor): KL divergence loss
        """
        prior_logits = self._prior_net(inputs)

        prior_dist = Categorical(logits=prior_logits)
        posterior_dist = Categorical(logits=posterior_params["logits"])

        # sum over latent dimensions, but average over batch dimension
        kl_loss = kl_divergence(posterior_dist, prior_dist)
        assert len(kl_loss.shape) == 2
        return kl_loss.sum(-1).mean()

    def forward(self, inputs: torch.Tensor = None):
        """
        Get a batch of prior parameters.

        Args:
            inputs (torch.Tensor): (Optional) A feature vector for priors that are input-conditional.

        Returns:
            params (dict): A dictionary of prior parameters with the same batch size as the inputs (1 if inputs is None)
        """
        prior_logits = self._prior_net(inputs)
        return prior_logits


class LearnedGaussianPrior(FixedGaussianPrior):
    """A Gaussian prior with learnable parameters"""
    def __init__(self, latent_dim, input_dim=None, device=None):
        super(LearnedGaussianPrior, self).__init__(
            latent_dim=latent_dim, input_dim=input_dim, device=device)
        self._params = nn.ParameterDict({
            "mu": nn.Parameter(data=torch.zeros(self._latent_dim), requires_grad=True),
            "logvar": nn.Parameter(data=torch.zeros(self._latent_dim), requires_grad=True)
        })

    def kl_loss(self, posterior_params, inputs=None):
        """
        Compute kl loss between the prior and the posterior distributions.

        Args:
            posterior_params (dict): a batch of distribution parameters
            inputs (torch.Tensor): (Optional) A feature vector for priors that are input-conditional.

        Returns:
            kl_loss (torch.Tensor): kl divergence value
        """

        assert posterior_params["mu"].shape[1] == self._latent_dim
        assert posterior_params["logvar"].shape[1] == self._latent_dim

        batch_size = posterior_params["mu"].shape[0]
        prior_params = TensorUtils.unsqueeze_expand_at(self._params, size=batch_size, dim=0)
        return KLD_gaussian_loss(
            mu_1=posterior_params["mu"],
            logvar_1=posterior_params["logvar"],
            mu_2=prior_params["mu"],
            logvar_2=prior_params["logvar"]
        )


class CVAE(nn.Module):
    def __init__(
            self,
            q_net: nn.Module,
            c_net: nn.Module,
            decoder: nn.Module,
            prior: Prior
    ):
        """
        A basic Conditional Variational Autoencoder Network (C-VAE)

        Args:
            q_net (nn.Module): a model that encodes data (x) and condition inputs (x_c) to posterior (q) parameters
            c_net (nn.Module): a model that encodes condition inputs (x_c) into condition feature (c)
            decoder (nn.Module): a model that decodes latent (z) and condition feature (c) to data (x')
            prior (nn.Module): a model containing information about distribution prior (kl-loss, prior params, etc.)
        """
        super(CVAE, self).__init__()
        self.q_net = q_net
        self.c_net = c_net
        self.decoder = decoder
        self.prior = prior

    def sample(self, condition_inputs, n: int, condition_feature=None, decoder_kwargs=None):
        """
        Draw data samples (x') given a batch of condition inputs (x_c) and the VAE prior.

        Args:
            condition_inputs (dict, torch.Tensor): condition inputs (x_c)
            n (int): number of samples to draw
            condition_feature (torch.Tensor): Optional - externally supply condition code (c)
            decoder_kwargs (dict): Extra keyword args for decoder (e.g., dynamics model states)

        Returns:
            dictionary of batched samples (x') of size [B, n, ...]
        """
        if condition_feature is not None:
            c = condition_feature
        else:
            c = self.c_net(condition_inputs)  # [B, ...]
        z = self.prior.sample(n=n, inputs=c)  # z of shape [B (from c), N, ...]
        z_samples = TensorUtils.join_dimensions(z, begin_axis=0, end_axis=2)  # [B * N, ...]
        c_samples = TensorUtils.repeat_by_expand_at(c, repeats=n, dim=0)  # [B * N, ...]
        decoder_kwargs = dict() if decoder_kwargs is None else decoder_kwargs
        x_out = self.decoder(latents=z_samples, condition_features=c_samples, **decoder_kwargs)
        x_out = TensorUtils.reshape_dimensions(x_out, begin_axis=0, end_axis=1, target_dims=(c.shape[0], n))
        return x_out

    def predict(self, condition_inputs, condition_feature=None, decoder_kwargs=None):
        """
        Generate a prediction based on latent prior (instead of sample) and condition inputs

        Args:
            condition_inputs (dict, torch.Tensor): condition inputs (x_c)
            condition_feature (torch.Tensor): Optional - externally supply condition code (c)
            decoder_kwargs (dict): Extra keyword args for decoder (e.g., dynamics model states)

        Returns:
            dictionary of batched predictions (x') of size [B, ...]

        """
        if condition_feature is not None:
            c = condition_feature
        else:
            c = self.c_net(condition_inputs)  # [B, ...]

        prior_params = self.prior(c)  # [B, ...]
        mu = self.prior.get_mean(prior_params)  # [B, ...]
        decoder_kwargs = dict() if decoder_kwargs is None else decoder_kwargs
        x_out = self.decoder(latents=mu, condition_features=c, **decoder_kwargs)
        return x_out

    def forward(self, inputs, condition_inputs, decoder_kwargs=None):
        """
        Pass the input through encoder and decoder (using posterior parameters)
        Args:
            inputs (dict, torch.Tensor): encoder inputs (x)
            condition_inputs (dict, torch.Tensor): condition inputs - (x_c)
            decoder_kwargs (dict): Extra keyword args for decoder (e.g., dynamics model states)

        Returns:
            dictionary of batched samples (x')
        """
        c = self.c_net(condition_inputs)  # [B, ...]
        q_params = self.q_net(inputs=inputs, condition_features=c)
        z = self.prior.sample_with_parameters(q_params, n=1).squeeze(dim=1)
        decoder_kwargs = dict() if decoder_kwargs is None else decoder_kwargs
        x_out = self.decoder(latents=z, condition_features=c, **decoder_kwargs)
        return {"x_recons": x_out, "q_params": q_params, "z": z, "c": c}

    def compute_kl_loss(self, outputs: dict):
        """
        Compute KL Divergence loss

        Args:
            outputs (dict): outputs of the self.forward() call

        Returns:
            a dictionary of loss values
        """
        return self.prior.kl_loss(outputs["q_params"], inputs=outputs["c"])


class DiscreteCVAE(nn.Module):
    def __init__(
            self,
            q_net: nn.Module,
            p_net: nn.Module,
            c_net: nn.Module,
            decoder: nn.Module,
            K: int,
            recon_loss_fun=None,
            logpi_clamp = None,
    ):
        """
        A basic Conditional Variational Autoencoder Network (C-VAE)

        Args:
            q_net (nn.Module): a model that encodes data (x) and condition inputs (x_c) to posterior (q) parameters
            p_net (nn.Module): a model that encodes condition feature (c) to latent (p) parameters
            c_net (nn.Module): a model that encodes condition inputs (x_c) into condition feature (c)
            decoder (nn.Module): a model that decodes latent (z) and condition feature (c) to data (x')
            K (int): cardinality of the discrete latent
            recon_loss: loss function handle for reconstruction loss
            logpi_clamp (float): lower bound of the logpis, for numerical stability
        """
        super(DiscreteCVAE, self).__init__()
        self.q_net = q_net
        self.p_net = p_net
        self.c_net = c_net
        self.decoder = decoder
        self.K = K
        self.logpi_clamp= logpi_clamp
        if recon_loss_fun is None:
            self.recon_loss_fun = nn.MSELoss(reduction="none")
        else:
            self.recon_loss_fun = recon_loss_fun

    def sample(self, condition_inputs, n: int, condition_feature=None, decoder_kwargs=None):
        """
        Draw data samples (x') given a batch of condition inputs (x_c) and the VAE prior.

        Args:
            condition_inputs (dict, torch.Tensor): condition inputs (x_c)
            n (int): number of samples to draw
            condition_feature (torch.Tensor): Optional - externally supply condition code (c)
            decoder_kwargs (dict): Extra keyword args for decoder (e.g., dynamics model states)

        Returns:
            dictionary of batched samples (x') of size [B, n, ...]
        """
        assert n<=self.K
        
        if condition_feature is not None:
            c = condition_feature
        else:
            c = self.c_net(condition_inputs)  # [B, ...]
        logp = self.p_net(c)["logp"]
        p = torch.exp(logp)
        p = p.nan_to_num(nan=0.0, posinf=1.0, neginf=0.0)
        p = p/p.sum(dim=-1,keepdim=True)
        # z = (-logp).argsort()[...,:n]
        # z = F.one_hot(z,self.K)

        dis_p = Categorical(probs=p)  # [n_sample, batch] -> [batch, n_sample]
        z = dis_p.sample((n,)).permute(1, 0)
        z = F.one_hot(z, self.K)

        z_samples = TensorUtils.join_dimensions(z, begin_axis=0, end_axis=2)  # [B * N, ...]
        c_samples = TensorUtils.repeat_by_expand_at(c, repeats=n, dim=0)  # [B * N, ...]
        decoder_kwargs = dict() if decoder_kwargs is None else decoder_kwargs
        x_out = self.decoder(latents=z_samples, condition_features=c_samples, **decoder_kwargs)
        x_out = TensorUtils.reshape_dimensions(x_out, begin_axis=0, end_axis=1, target_dims=(c.shape[0], n))
        x_out["z"] = z_samples
        return x_out

    def predict(self, condition_inputs, condition_feature=None, decoder_kwargs=None):
        """
        Generate a prediction based on latent prior (instead of sample) and condition inputs

        Args:
            condition_inputs (dict, torch.Tensor): condition inputs (x_c)
            condition_feature (torch.Tensor): Optional - externally supply condition code (c)
            decoder_kwargs (dict): Extra keyword args for decoder (e.g., dynamics model states)

        Returns:
            dictionary of batched predictions (x') of size [B, ...]

        """
        if condition_feature is not None:
            c = condition_feature
        else:
            c = self.c_net(condition_inputs)  # [B, ...]

        logp = self.p_net(c)["logp"]
        z = logp.argmax(dim=-1)
        
        decoder_kwargs = dict() if decoder_kwargs is None else decoder_kwargs
        x_out = self.decoder(latents=F.one_hot(z,self.K), condition_features=c, **decoder_kwargs)
        return x_out

    def forward(self, inputs, condition_inputs, n=None, decoder_kwargs=None):
        """
        Pass the input through encoder and decoder (using posterior parameters)
        Args:
            inputs (dict, torch.Tensor): encoder inputs (x)
            condition_inputs (dict, torch.Tensor): condition inputs - (x_c)
            n (int): number of samples, if not given, then n=self.K
            decoder_kwargs (dict): Extra keyword args for decoder (e.g., dynamics model states)

        Returns:
            dictionary of batched samples (x')
        """
        if n is None:
            n = self.K
        c = self.c_net(condition_inputs)  # [B, ...]
        logq = self.q_net(inputs=inputs, condition_features=c)["logq"]
        logp = self.p_net(c)["logp"]
        if self.logpi_clamp is not None:
            logq = logq.clamp(min=self.logpi_clamp,max=2.0)
            logp = logp.clamp(min=self.logpi_clamp,max=2.0)
        
        q = torch.exp(logq)
        q = q/q.sum(dim=-1,keepdim=True)
        
        p = torch.exp(logp)
        p = p/p.sum(dim=-1,keepdim=True)
        z = (-logq).argsort()[...,:n]
        z = F.one_hot(z,self.K)
        decoder_kwargs = dict() if decoder_kwargs is None else decoder_kwargs
        c_tiled = c.unsqueeze(1).repeat(1,n,1)
        x_out = self.decoder(latents=z.reshape(-1,self.K), condition_features=c_tiled.reshape(-1,c.shape[-1]), **decoder_kwargs)
        x_out = TensorUtils.reshape_dimensions(x_out,0,1,(z.shape[0],n))
        return {"x_recons": x_out, "q": q, "p": p, "z": z, "c": c}

    def compute_kl_loss(self, outputs: dict):
        """
        Compute KL Divergence loss

        Args:
            outputs (dict): outputs of the self.forward() call

        Returns:
            a dictionary of loss values
        """
        p = outputs["p"]
        q = outputs["q"]
        return (p*(torch.log(p)-torch.log(q))).sum(dim=-1).mean()

    def compute_losses(self,outputs,targets,gamma=1):
        recon_loss = 0
        for k,v in outputs['x_recons'].items():
            if k in targets:
                if isinstance(self.recon_loss_fun,dict):
                    loss_v = self.recon_loss_fun[k](v,targets[k].unsqueeze(1))
                else:
                    loss_v = self.recon_loss_fun(v,targets[k].unsqueeze(1))
                sum_dim=tuple(range(2,loss_v.ndim))
                loss_v = loss_v.sum(dim=sum_dim)
                loss_v_detached = loss_v.detach()
                min_flag = (loss_v==loss_v.min(dim=1,keepdim=True)[0])
                nonmin_flag = torch.logical_not(min_flag)
                recon_loss +=(loss_v*min_flag*outputs["q"]).sum(dim=1)+(loss_v_detached*nonmin_flag*outputs["q"]).sum(dim=1)

        KL_loss = self.compute_kl_loss(outputs)
        return recon_loss + gamma*KL_loss


class ECDiscreteCVAE(DiscreteCVAE):
    def sample(self, condition_inputs, n: int,cond_traj = None, decoder_kwargs=None):
        """
        Draw data samples (x') given a batch of condition inputs (x_c) and the VAE prior.

        Args:
            condition_inputs (dict, torch.Tensor): condition inputs (x_c)
            n (int): number of samples to draw
            decoder_kwargs (dict): Extra keyword args for decoder (e.g., dynamics model states)

        Returns:
            dictionary of batched samples (x') of size [B, n, ...]
        """
        assert n<=self.K
        
        if cond_traj is not None:
            condition_inputs["cond_traj"] = cond_traj
        c = self.c_net(condition_inputs)  # [B, ...]
            
        
        bs,Na = c.shape[:2]
        c_joined = TensorUtils.join_dimensions(c,0,2)
        logp = self.p_net(c_joined)["logp"]
        p = torch.exp(logp)
        p = p/p.sum(dim=-1,keepdim=True)
        # z = (-logp).argsort()[...,:n]
        # z = F.one_hot(z,self.K)

        dis_p = Categorical(probs=p)  # [n_sample, batch] -> [batch, n_sample]
        
        z = dis_p.sample((n,)).permute(1, 0)
        z = F.one_hot(z, self.K)

        z_samples = TensorUtils.join_dimensions(z, begin_axis=0, end_axis=2)  # [B * Na * n, ...]
        c_samples = TensorUtils.repeat_by_expand_at(c_joined, repeats=n, dim=0)  # [B * Na * n, ...]

        if decoder_kwargs is None:
            decoder_kwargs = dict()
        else:
            
            if decoder_kwargs["current_states"].ndim==2:
                decoder_kwargs = TensorUtils.unsqueeze_expand_at(decoder_kwargs,Na,1)
            else:
                assert decoder_kwargs["current_states"].ndim==3 and decoder_kwargs["current_states"].shape[1]==Na
            decoder_kwargs = TensorUtils.unsqueeze_expand_at(decoder_kwargs,n,2)
            decoder_kwargs = TensorUtils.join_dimensions(decoder_kwargs,0,3)
        x_out = self.decoder(latents=z_samples, condition_features=c_samples, **decoder_kwargs)
        x_out = TensorUtils.reshape_dimensions(x_out, begin_axis=0, end_axis=1, target_dims=(bs,Na,n))
        
        return x_out

    def predict(self, condition_inputs, cond_traj = None, decoder_kwargs=None):
        """
        Generate a prediction based on latent prior (instead of sample) and condition inputs

        Args:
            condition_inputs (dict, torch.Tensor): condition inputs (x_c)
            decoder_kwargs (dict): Extra keyword args for decoder (e.g., dynamics model states)

        Returns:
            dictionary of batched predictions (x') of size [B, ...]

        """
        if cond_traj is not None:
            condition_inputs["cond_traj"] = cond_traj
        c = self.c_net(condition_inputs)  # [B, ...]
            
        bs,Na = c.shape[:2]
        c_joined = TensorUtils.join_dimensions(c,0,2)
        logp = self.p_net(c_joined)["logp"]
        z = logp.argmax(dim=-1)
        
        if decoder_kwargs is None:
            decoder_kwargs = dict()
        else:
            if decoder_kwargs["current_states"].ndim==2:
                decoder_kwargs = TensorUtils.unsqueeze_expand_at(decoder_kwargs,Na,1)
            else:
                assert decoder_kwargs["current_states"].ndim==3 and decoder_kwargs["current_states"].shape[1]==Na
            decoder_kwargs = TensorUtils.join_dimensions(decoder_kwargs,0,2)
        x_out = self.decoder(latents=F.one_hot(z,self.K), condition_features=c_joined, **decoder_kwargs)
        x_out = TensorUtils.reshape_dimensions(x_out, begin_axis=0, end_axis=1, target_dims=(bs,Na))
        return x_out

    def forward(self, inputs, condition_inputs, cond_traj, decoder_kwargs=None):
        """
        Pass the input through encoder and decoder (using posterior parameters)
        Args:
            inputs (dict, torch.Tensor): encoder inputs (x)
            condition_inputs (dict, torch.Tensor): condition inputs - (x_c)
            n (int): number of samples, if not given, then n=self.K
            decoder_kwargs (dict): Extra keyword args for decoder (e.g., dynamics model states)

        Returns:
            dictionary of batched samples (x')
        """
        condition_inputs["cond_traj"] = cond_traj
        c = self.c_net(condition_inputs)  # [B, ...]
        c_joined = TensorUtils.join_dimensions(c,0,2)
        bs,Na = c.shape[:2]
        logp = TensorUtils.reshape_dimensions(self.p_net(c_joined)["logp"],0,1,(bs,Na))
        if inputs is not None:
            inputs_tiled = TensorUtils.unsqueeze_expand_at(inputs,Na,1)
            inputs_joined = TensorUtils.join_dimensions(inputs_tiled,0,2)
            logq = TensorUtils.reshape_dimensions(self.q_net(inputs=inputs_joined, condition_features=c_joined)["logq"],0,1,(bs,Na))
        else:
            logq = logp
        if self.logpi_clamp is not None:
            logq = logq.clamp(min=self.logpi_clamp,max=2.0)
            logp = logp.clamp(min=self.logpi_clamp,max=2.0)
        
        q = torch.exp(logq)
        p = torch.exp(logp)
        p = p.nan_to_num(nan=0.0, posinf=1.0, neginf=0.0)
        q = q.nan_to_num(nan=0.0, posinf=1.0, neginf=0.0)
        q = q/q.sum(dim=-1,keepdim=True)
        p = p/p.sum(dim=-1,keepdim=True)

        z = torch.arange(self.K).to(q.device).tile(*q.shape[:-1],1)
        z = F.one_hot(z,self.K)

        if decoder_kwargs is None:
            decoder_kwargs = dict()
        else:
            if decoder_kwargs["current_states"].ndim==2:
                decoder_kwargs = TensorUtils.unsqueeze_expand_at(decoder_kwargs,Na,1)
            else:
                assert decoder_kwargs["current_states"].ndim==3 and decoder_kwargs["current_states"].shape[1]==Na
            decoder_kwargs = TensorUtils.unsqueeze_expand_at(decoder_kwargs,self.K,2)
            decoder_kwargs = TensorUtils.join_dimensions(decoder_kwargs,0,3)
        
        c_tiled = c[:,:,None].repeat(1,1,self.K,1)
        x_out = self.decoder(latents=z.reshape(-1,self.K), condition_features=TensorUtils.join_dimensions(c_tiled,0,3), **decoder_kwargs)
        x_out = TensorUtils.reshape_dimensions(x_out,0,1,(bs,Na,self.K))
        return {"x_recons": x_out, "q": q, "p": p, "z": z}

    def compute_kl_loss(self, outputs: dict):
        """
        Compute KL Divergence loss

        Args:
            outputs (dict): outputs of the self.forward() call

        Returns:
            a dictionary of loss values
        """
        p = outputs["p"]
        q = outputs["q"]
        return (p*(torch.log(p)-torch.log(q))).sum(dim=-1).mean()

    def compute_losses(self,outputs,targets,gamma=1):
        recon_loss = 0
        for k,v in outputs['x_recons'].items():
            if k in targets:
                if isinstance(self.recon_loss_fun,dict):
                    loss_v = self.recon_loss_fun[k](v,targets[k].unsqueeze(1))
                else:
                    loss_v = self.recon_loss_fun(v,targets[k].unsqueeze(1))
                sum_dim=tuple(range(2,loss_v.ndim))
                loss_v = loss_v.sum(dim=sum_dim)
                loss_v_detached = loss_v.detach()
                min_flag = (loss_v==loss_v.min(dim=1,keepdim=True)[0])
                nonmin_flag = torch.logical_not(min_flag)
                recon_loss +=(loss_v*min_flag*outputs["q"]).sum(dim=1)+(loss_v_detached*nonmin_flag*outputs["q"]).sum(dim=1)

        KL_loss = self.compute_kl_loss(outputs)
        return recon_loss + gamma*KL_loss


class SceneDiscreteCVAE(DiscreteCVAE):
    def __init__(self,
                 q_net: nn.Module,
                 p_net: nn.Module,
                 c_net: nn.Module,
                 decoder: nn.Module,
                 transformer: nn.Module,
                 K: int,
                 aggregate_func="max",
                 recon_loss_fun=None,
                 logpi_clamp = None,
                 num_latent_sample = None,
                 ):
        super(SceneDiscreteCVAE,self).__init__(q_net, p_net, c_net, decoder, K, recon_loss_fun, logpi_clamp)
        self.transformer = transformer
        self.aggregate_func = aggregate_func
        if num_latent_sample is None:
            num_latent_sample = self.K
        assert num_latent_sample<=self.K
        self.num_latent_sample = num_latent_sample
            
    def sample(self, condition_inputs, mask, pos, n: int,cond_traj = None, decoder_kwargs=None):
        """
        Draw data samples (x') given a batch of condition inputs (x_c) and the VAE prior.

        Args:
            condition_inputs (dict, torch.Tensor): condition inputs (x_c) [B, Na, D]
            mask (torch.Tensor): mask of the agents in the scene [B,Na]
            pos (torch.Tensor): position of the agents in the scene [B,Na,2]
            n (int): number of samples to draw
            decoder_kwargs (dict): Extra keyword args for decoder (e.g., dynamics model states)

        Returns:
            dictionary of batched samples (x') of size [B, n, ...]
        """
        assert n<=self.K
        

        bs,Na = next(iter(condition_inputs.values())).shape[:2]
        condition_inputs = TensorUtils.join_dimensions(condition_inputs,0,2)
        if cond_traj is not None:
            condition_inputs["cond_traj"] = cond_traj #[B,T,3]
        
        c = self.c_net(condition_inputs).reshape(bs,Na,-1)  # [B*Na, ...]
        
        c = self.transformer(c,mask,pos)+c
        if self.aggregate_func == "max":
            c_agg = c.max(1)[0]
        elif self.aggregate_func == "mean":
            c_agg = c.mean(1)
        logp = self.p_net(c_agg)["logp"]
        p = torch.exp(logp)
        p = p/p.sum(dim=-1,keepdim=True)

        dis_p = Categorical(probs=p)  # [n_sample, batch] -> [batch, n_sample]
        
        z = dis_p.sample((n,)).permute(1, 0)
        z = F.one_hot(z, self.K) #[B,n,K]
        z = TensorUtils.repeat_by_expand_at(z,repeats=Na,dim=0)

        z_samples = TensorUtils.join_dimensions(z, begin_axis=0, end_axis=2)  # [B * Na * n, ...]
        c_samples = TensorUtils.repeat_by_expand_at(c, repeats=n, dim=0)  # [B * Na * n, ...]

        if decoder_kwargs is None:
            decoder_kwargs = dict()
        else:
            
            assert decoder_kwargs["current_states"].ndim==3 and decoder_kwargs["current_states"].shape[1]==Na
            decoder_kwargs = TensorUtils.unsqueeze_expand_at(decoder_kwargs,n,2)
            decoder_kwargs = TensorUtils.join_dimensions(decoder_kwargs,0,3)
        x_out = self.decoder(latents=z_samples, condition_features=c_samples, **decoder_kwargs)
        x_out = TensorUtils.reshape_dimensions(x_out, begin_axis=0, end_axis=1, target_dims=(bs,Na,n))
        
        return x_out

    def predict(self, condition_inputs, mask, pos, cond_traj = None, decoder_kwargs=None):
        """
        Generate a prediction based on latent prior (instead of sample) and condition inputs

        Args:
            condition_inputs (dict, torch.Tensor): condition inputs (x_c)
            mask (torch.Tensor): mask of the agents in the scene [B,Na]
            pos (torch.Tensor): position of the agents in the scene [B,Na,2]
            decoder_kwargs (dict): Extra keyword args for decoder (e.g., dynamics model states)

        Returns:
            dictionary of batched predictions (x') of size [B, ...]

        """

        bs,Na = next(iter(condition_inputs.values())).shape[:2]
        condition_inputs = TensorUtils.join_dimensions(condition_inputs,0,2)
        if cond_traj is not None:
            condition_inputs["cond_traj"] = cond_traj #[B,T,3]
        
        c = self.c_net(condition_inputs).reshape(bs,Na,-1)  # [B*Na, ...]
        
        c = self.transformer(c,mask,pos)+c
        if self.aggregate_func == "max":
            c_agg = c.max(1)[0]
        elif self.aggregate_func == "mean":
            c_agg = c.mean(1)
        logp = self.p_net(c_agg)["logp"]
        p = torch.exp(logp)
        p = p/p.sum(dim=-1,keepdim=True)

        
        z = p.argmax(-1)  #[B]
        z = F.one_hot(z, self.K) #[B,K]
        z = TensorUtils.repeat_by_expand_at(z,repeats=Na,dim=0) #[B*Na,K]

        z = TensorUtils.join_dimensions(z, begin_axis=0, end_axis=2)  # [B * Na , ...]


        if decoder_kwargs is None:
            decoder_kwargs = dict()
        else:
            assert decoder_kwargs["current_states"].ndim==3 and decoder_kwargs["current_states"].shape[1]==Na
            decoder_kwargs = TensorUtils.join_dimensions(decoder_kwargs,0,2)
        x_out = self.decoder(latents=z, condition_features=c, **decoder_kwargs)
        x_out = TensorUtils.reshape_dimensions(x_out, begin_axis=0, end_axis=1, target_dims=(bs,Na))
        
        return x_out

    def forward(self, inputs, condition_inputs, mask, pos, cond_traj=None, decoder_kwargs=None):
        """
        Pass the input through encoder and decoder (using posterior parameters)
        Args:
            inputs (dict, torch.Tensor): encoder inputs (x)
            condition_inputs (dict, torch.Tensor): condition inputs - (x_c)
            mask (torch.Tensor): mask of the agents in the scene [B,Na]
            pos (torch.Tensor): position of the agents in the scene [B,Na,2]
            n (int): number of samples, if not given, then n=self.K
            decoder_kwargs (dict): Extra keyword args for decoder (e.g., dynamics model states)

        Returns:
            dictionary of batched samples (x')
        """

        
        bs,Na = next(iter(condition_inputs.values())).shape[:2]
        condition_inputs = TensorUtils.join_dimensions(condition_inputs,0,2)
        if cond_traj is not None:
            condition_inputs["cond_traj"] = TensorUtils.join_dimensions(cond_traj,0,2) #[B*Na,T,3]
        
        c = self.c_net(condition_inputs).reshape(bs,Na,-1)  # [B*Na, ...]
        
        c = self.transformer(c,mask,pos)+c
        if self.aggregate_func == "max":
            c_agg = c.max(1)[0]
        elif self.aggregate_func == "mean":
            c_agg = c.mean(1)
        logp = self.p_net(c_agg)["logp"]
        p = torch.exp(logp)
        p = p/p.sum(dim=-1,keepdim=True)
        if inputs is not None:
            # inputs_joined = TensorUtils.join_dimensions(inputs,0,2)
            logq = self.q_net(inputs,c,mask,pos)["logq"]
        else:
            logq = logp
        if self.logpi_clamp is not None:
            logq = logq.clamp(min=self.logpi_clamp,max=2.0)
            logp = logp.clamp(min=self.logpi_clamp,max=2.0)
        
        q = torch.exp(logq)
        p = torch.exp(logp)
        p = p.nan_to_num(nan=0.0, posinf=1.0, neginf=0.0)
        q = q.nan_to_num(nan=0.0, posinf=1.0, neginf=0.0)
        
        _,z = torch.topk(q,self.num_latent_sample,dim=-1)
        p = p.take_along_dim(z,-1)
        q = q.take_along_dim(z,-1)
        q = q/q.sum(dim=-1,keepdim=True)
        p = p/p.sum(dim=-1,keepdim=True)
        z = F.one_hot(z,self.K)
        

        if decoder_kwargs is None:
            decoder_kwargs = dict()
        else:
            if "current_states" in decoder_kwargs:
                assert decoder_kwargs["current_states"].ndim==3 and decoder_kwargs["current_states"].shape[1]==Na
                    
            decoder_kwargs = TensorUtils.unsqueeze_expand_at(decoder_kwargs,self.num_latent_sample,2)
            decoder_kwargs = TensorUtils.join_dimensions(decoder_kwargs,0,3)
        
        c_tiled = c[:,:,None].repeat(1,1,self.num_latent_sample,1)
        x_out = self.decoder(
                             latents=TensorUtils.unsqueeze_expand_at(z,Na,1).reshape(-1,self.K), 
                             condition_features=TensorUtils.join_dimensions(c_tiled,0,3), 
                             **decoder_kwargs
                            )

        x_out = TensorUtils.reshape_dimensions(x_out,0,1,(bs,Na,self.num_latent_sample))
        return {"x_recons": x_out, "q": q, "p": p, "z": z}

    def compute_kl_loss(self, outputs: dict):
        """
        Compute KL Divergence loss

        Args:
            outputs (dict): outputs of the self.forward() call

        Returns:
            a dictionary of loss values
        """
        p = outputs["p"]
        q = outputs["q"]
        return (p*(torch.log(p)-torch.log(q))).sum(dim=-1).mean()

    def compute_losses(self,outputs,targets,gamma=1):
        recon_loss = 0
        for k,v in outputs['x_recons'].items():
            if k in targets:
                if isinstance(self.recon_loss_fun,dict):
                    loss_v = self.recon_loss_fun[k](v,targets[k].unsqueeze(1))
                else:
                    loss_v = self.recon_loss_fun(v,targets[k].unsqueeze(1))
                sum_dim=tuple(range(2,loss_v.ndim))
                loss_v = loss_v.sum(dim=sum_dim)
                loss_v_detached = loss_v.detach()
                min_flag = (loss_v==loss_v.min(dim=1,keepdim=True)[0])
                nonmin_flag = torch.logical_not(min_flag)
                recon_loss +=(loss_v*min_flag*outputs["q"]).sum(dim=1)+(loss_v_detached*nonmin_flag*outputs["q"]).sum(dim=1)

        KL_loss = self.compute_kl_loss(outputs)
        return recon_loss + gamma*KL_loss


class SceneDiscreteCVAEDiverse(SceneDiscreteCVAE):
    def __init__(self,
                 q_net: nn.Module,
                 p_net: nn.Module,
                 c_net: nn.Module,
                 decoder: nn.Module,
                 transformer: nn.Module,
                 latent_embeding: nn.Module,
                 K: int,
                 aggregate_func="max",
                 recon_loss_fun=None,
                 logpi_clamp = None,
                 num_latent_sample=None):
        super(SceneDiscreteCVAEDiverse,self).__init__(q_net,p_net,c_net, decoder, transformer, K, aggregate_func, recon_loss_fun, logpi_clamp, num_latent_sample)
        self.latent_embeding = latent_embeding
    def sample(self, condition_inputs, mask, pos, n: int,cond_traj = None, decoder_kwargs=None):
        
        """
        Draw data samples (x') given a batch of condition inputs (x_c) and the VAE prior.

        Args:
            condition_inputs (dict, torch.Tensor): condition inputs (x_c) [B, Na, D]
            mask (torch.Tensor): mask of the agents in the scene [B,Na]
            pos (torch.Tensor): position of the agents in the scene [B,Na,2]
            n (int): number of samples to draw
            decoder_kwargs (dict): Extra keyword args for decoder (e.g., dynamics model states)

        Returns:
            dictionary of batched samples (x') of size [B, n, ...]
        """
        assert n<=self.K
        res = self.forward(None, condition_inputs, mask, pos, cond_traj, decoder_kwargs)
        dis_p = Categorical(probs=res["p"])  # [n_sample, batch] -> [batch, n_sample]
        
        z = dis_p.sample((n,)).permute(1, 0)
        x_out = {k:v[z] for k,v in res["x_recons"].items()}
        return x_out

    def predict(self, condition_inputs, mask, pos, cond_traj = None, decoder_kwargs=None):
        """
        Generate a prediction based on latent prior (instead of sample) and condition inputs

        Args:
            condition_inputs (dict, torch.Tensor): condition inputs (x_c)
            mask (torch.Tensor): mask of the agents in the scene [B,Na]
            pos (torch.Tensor): position of the agents in the scene [B,Na,2]
            decoder_kwargs (dict): Extra keyword args for decoder (e.g., dynamics model states)

        Returns:
            dictionary of batched predictions (x') of size [B, ...]

        """

        bs,Na = next(iter(condition_inputs.values())).shape[:2]
        condition_inputs = TensorUtils.join_dimensions(condition_inputs,0,2)
        if cond_traj is not None:
            condition_inputs["cond_traj"] = cond_traj #[B,T,3]
        
        c = self.c_net(condition_inputs).reshape(bs,Na,-1)  # [B*Na, ...]
        z_enum = torch.eye(self.K).repeat(bs,Na,1,1).to(c.device)
        z_emb = self.latent_embeding(z_enum)
        latent_emb_dim = z_emb.shape[-1]
        c_tiled = c.unsqueeze(-2).repeat_interleave(self.K,-2)
        cz_tiled = torch.cat((c_tiled,z_emb),-1)
        
        cz_tiled = TensorUtils.join_dimensions(cz_tiled.transpose(1,2),0,2)
        mask_tiled = mask.repeat_interleave(self.K,0)
        pos_tiled = pos.repeat_interleave(self.K,0)
        
        cz = self.transformer(cz_tiled,mask_tiled,pos_tiled)+cz_tiled
        if self.aggregate_func == "max":
            c_agg = cz.max(1)[0][...,-latent_emb_dim:]
        elif self.aggregate_func == "mean":
            c_agg = cz.mean(1)[...,-latent_emb_dim:]
        logp = self.p_net(c_agg)["logp"]
        logp = logp.reshape(bs,self.K)
        
        
        p = torch.exp(logp)

        if self.logpi_clamp is not None:
            logp = logp.clamp(min=self.logpi_clamp,max=2.0)
        p = torch.exp(logp)
        p = p.nan_to_num(nan=0.0, posinf=1.0, neginf=0.0) 
        p = p/p.sum(dim=-1,keepdim=True)
        z_idx = p.argmax(dim=1)
        z_idx = torch.arange(bs).to(c.device)*self.K+z_idx


        

        if decoder_kwargs is None:
            decoder_kwargs = dict()
        else:
            if "current_states" in decoder_kwargs:
                if decoder_kwargs["current_states"].ndim==2:
                    decoder_kwargs = TensorUtils.unsqueeze_expand_at(decoder_kwargs,self.K,1)
                else:
                    assert decoder_kwargs["current_states"].ndim==3 and decoder_kwargs["current_states"].shape[1]==Na
            decoder_kwargs = TensorUtils.join_dimensions(decoder_kwargs,0,3)
        x_out = self.decoder(TensorUtils.join_dimensions(cz[z_idx],0,2), **decoder_kwargs)


        x_out = TensorUtils.reshape_dimensions(x_out,0,1,(bs, Na))
        
        return x_out

    def forward(self, inputs, condition_inputs, mask, pos, cond_traj=None, decoder_kwargs=None):
        """
        Pass the input through encoder and decoder (using posterior parameters)
        Args:
            inputs (dict, torch.Tensor): encoder inputs (x)
            condition_inputs (dict, torch.Tensor): condition inputs - (x_c)
            mask (torch.Tensor): mask of the agents in the scene [B,Na]
            pos (torch.Tensor): position of the agents in the scene [B,Na,2]
            n (int): number of samples, if not given, then n=self.K
            decoder_kwargs (dict): Extra keyword args for decoder (e.g., dynamics model states)

        Returns:
            dictionary of batched samples (x')
        """

        
        bs,Na = next(iter(condition_inputs.values())).shape[:2]
        condition_inputs = TensorUtils.join_dimensions(condition_inputs,0,2)
        if cond_traj is not None:
            condition_inputs["cond_traj"] = TensorUtils.join_dimensions(cond_traj,0,2) #[B*Na,T,3]
        
        c = self.c_net(condition_inputs).reshape(bs,Na,-1)  # [B*Na, ...]
        z_enum = torch.eye(self.K).repeat(bs,Na,1,1).to(c.device)
        z_emb = self.latent_embeding(z_enum)
        latent_emb_dim = z_emb.shape[-1]
        c_tiled = c.unsqueeze(-2).repeat_interleave(self.K,-2)
        cz_tiled = torch.cat((c_tiled,z_emb),-1)
        
        cz_tiled = TensorUtils.join_dimensions(cz_tiled.transpose(1,2),0,2)
        mask_tiled = mask.repeat_interleave(self.K,0)
        pos_tiled = pos.repeat_interleave(self.K,0)
        
        cz = self.transformer(cz_tiled,mask_tiled,pos_tiled)+cz_tiled
        if self.aggregate_func == "max":
            c_agg = cz.max(1)[0][...,-latent_emb_dim:]
        elif self.aggregate_func == "mean":
            c_agg = cz.mean(1)[...,-latent_emb_dim:]
        logp = self.p_net(c_agg)["logp"]
        logp = logp.reshape(bs,self.K)
        
        
        p = torch.exp(logp)
        if inputs is not None:
            # inputs_joined = TensorUtils.join_dimensions(inputs,0,2)
            inputs_tiled = {k:v.repeat_interleave(self.K,0) for k,v in inputs.items()}
            logq = self.q_net(inputs_tiled,cz[...,-latent_emb_dim:],mask_tiled,pos_tiled)["logq"]
            logq = logq.reshape(bs,self.K)
        else:
            logq = logp
        if self.logpi_clamp is not None:
            logq = logq.clamp(min=self.logpi_clamp,max=3.0)
            logp = logp.clamp(min=self.logpi_clamp,max=3.0)
        
        q = torch.exp(logq)
        p = torch.exp(logp)
        p = p.nan_to_num(nan=0.0, posinf=1.0, neginf=0.0)
        q = q.nan_to_num(nan=0.0, posinf=1.0, neginf=0.0)

        _,z = torch.topk(q,self.num_latent_sample,dim=-1)
        p = p.take_along_dim(z,-1)
        q = q.take_along_dim(z,-1)
        
        cz = cz.reshape(bs,Na,self.K,-1)
        cz = cz.take_along_dim(z[:,None,:,None],2)
        cz = cz.transpose(1,2)
        q = q/q.sum(dim=-1,keepdim=True)
        p = p/p.sum(dim=-1,keepdim=True)


        if decoder_kwargs is None:
            decoder_kwargs = dict()
        else:
            if "current_states" in decoder_kwargs:
                assert decoder_kwargs["current_states"].ndim==3 and decoder_kwargs["current_states"].shape[1]==Na
            decoder_kwargs = TensorUtils.unsqueeze_expand_at(decoder_kwargs,self.num_latent_sample,1)
            decoder_kwargs = TensorUtils.join_dimensions(decoder_kwargs,0,3)
        x_out = self.decoder(TensorUtils.join_dimensions(cz,0,3), **decoder_kwargs)


        x_out = TensorUtils.reshape_dimensions(x_out,0,1,(bs, self.num_latent_sample, Na))
        x_out = {k:v.transpose(1,2) for k,v in x_out.items()}
        return {"x_recons": x_out, "q": q, "p": p, "z": z}




def main():
    import tbsim.models.base_models as l5m

    inputs = OrderedDict(trajectories=torch.randn(10, 50, 3))
    condition_inputs = OrderedDict(image=torch.randn(10, 3, 224, 224))

    condition_dim = 16
    latent_dim = 4

    prior = FixedGaussianPrior(latent_dim=4)

    map_encoder = l5m.RasterizedMapEncoder(
        model_arch="resnet18",
        num_input_channels=3,
        feature_dim=128
    )

    q_encoder = l5m.PosteriorEncoder(
        condition_dim=condition_dim,
        trajectory_shape=(50, 3),
        output_shapes=OrderedDict(mu=(latent_dim,), logvar=(latent_dim,))
    )
    c_encoder = l5m.ConditionEncoder(
        map_encoder=map_encoder,
        trajectory_shape=(50, 3),
        condition_dim=condition_dim
    )
    decoder = l5m.ConditionDecoder(
        condition_dim=condition_dim,
        latent_dim=latent_dim,
        output_shapes=OrderedDict(trajectories=(50, 3))
    )

    model = CVAE(
        q_net=q_encoder,
        c_net=c_encoder,
        decoder=decoder,
        prior=prior,
        target_criterion=nn.MSELoss(reduction="none")
    )


    outputs = model(inputs=inputs, condition_inputs=condition_inputs)
    losses = model.compute_losses(outputs=outputs, targets=inputs)
    samples = model.sample(condition_inputs=condition_inputs, n=10)
    print()

    traj_encoder = l5m.RNNTrajectoryEncoder(
        trajectory_dim=3,
        rnn_hidden_size=100
    )

    c_net = l5m.ConditionNet(
        condition_input_shapes=OrderedDict(
            map_feature=(map_encoder.output_shape()[-1],)
        ),
        condition_dim=condition_dim,
    )

    q_net = l5m.PosteriorNet(
        input_shapes=OrderedDict(
            traj_feature=(traj_encoder.output_shape()[-1],)
        ),
        condition_dim=condition_dim,
        param_shapes=prior.posterior_param_shapes,
    )

    lean_model = CVAE(
        q_net=q_net,
        c_net=c_net,
        decoder=decoder,
        prior=prior,
        target_criterion=nn.MSELoss(reduction="none")
    )

    map_feats = map_encoder(condition_inputs["image"])
    traj_feats = traj_encoder(inputs["trajectories"])
    input_feats = dict(traj_feature=traj_feats)
    condition_feats = dict(map_feature=map_feats)

    outputs = lean_model(inputs=input_feats, condition_inputs=condition_feats)
    losses = lean_model.compute_losses(outputs=outputs, targets=inputs)
    samples = lean_model.sample(condition_inputs=condition_feats, n=10)
    print()


def main_discrete():
    import tbsim.models.base_models as l5m

    inputs = OrderedDict(trajectories=torch.randn(10, 50, 3))
    condition_inputs = OrderedDict(image=torch.randn(10, 3, 224, 224))

    condition_dim = 16
    latent_dim = 20

    map_encoder = l5m.RasterizedMapEncoder(
        model_arch="resnet18",
        feature_dim=128
    )

    q_encoder = l5m.PosteriorEncoder(
        condition_dim=condition_dim,
        trajectory_shape=(50, 3),
        output_shapes=OrderedDict(logq=(latent_dim,))
    )
    p_encoder = l5m.SplitMLP(
                input_dim=condition_dim,
                layer_dims=(128,128),
                output_shapes=OrderedDict(logp=(latent_dim,))
            )
    c_encoder = l5m.ConditionEncoder(
        map_encoder=map_encoder,
        trajectory_shape=(50, 3),
        condition_dim=condition_dim
    )
    decoder_MLP = l5m.SplitMLP(
        input_dim=condition_dim+latent_dim,
        output_shapes=OrderedDict(trajectories=(50, 3)),
        layer_dims=(128,128),
        output_activation=nn.ReLU,
    )
    decoder = l5m.ConditionDecoder(decoder_model=decoder_MLP)

    model = DiscreteCVAE(
        q_net=q_encoder,
        p_net=p_encoder,
        c_net=c_encoder,
        decoder=decoder,
        K=latent_dim,
    )


    outputs = model(inputs=inputs, condition_inputs=condition_inputs)
    losses = model.compute_losses(outputs=outputs, targets = inputs)
    samples = model.sample(condition_inputs=condition_inputs, n=10)
    KL_loss = model.compute_kl_loss(outputs)


    # traj_encoder = l5m.RNNTrajectoryEncoder(
    #     trajectory_dim=3,
    #     rnn_hidden_size=100
    # )

    # c_net = l5m.ConditionNet(
    #     condition_input_shapes=OrderedDict(
    #         map_feature=(map_encoder.output_shape()[-1],)
    #     ),
    #     condition_dim=condition_dim,
    # )

    # q_net = l5m.PosteriorNet(
    #     input_shapes=OrderedDict(
    #         traj_feature=(traj_encoder.output_shape()[-1],)
    #     ),
    #     condition_dim=condition_dim,
    #     param_shapes=prior.posterior_param_shapes,
    # )

    # lean_model = CVAE(
    #     q_net=q_net,
    #     c_net=c_net,
    #     decoder=decoder,
    #     prior=prior,
    #     target_criterion=nn.MSELoss(reduction="none")
    # )

    # map_feats = map_encoder(condition_inputs["image"])
    # traj_feats = traj_encoder(inputs["trajectories"])
    # input_feats = dict(traj_feature=traj_feats)
    # condition_feats = dict(map_feature=map_feats)

    # outputs = lean_model(inputs=input_feats, condition_inputs=condition_feats)
    # losses = lean_model.compute_losses(outputs=outputs, targets=inputs)
    # samples = lean_model.sample(condition_inputs=condition_feats, n=10)
    # print()

if __name__ == "__main__":
    main_discrete()