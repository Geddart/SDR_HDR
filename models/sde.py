"""
IRSDE (Image Restoration SDE) — extracted from Refusion-HDR.
Inference-only: forward noise injection + reverse posterior sampling.
"""
import math
import torch
from tqdm import tqdm


class IRSDE:
    def __init__(self, max_sigma: float = 50.0, T: int = 100,
                 schedule: str = "cosine", eps: float = 0.005,
                 device: torch.device = None):
        self.T = T
        self.device = device or torch.device("cpu")
        self.max_sigma = max_sigma / 255 if max_sigma >= 1 else max_sigma
        self._initialize(self.max_sigma, T, schedule, eps)
        self.mu = None
        self.model = None

    def _initialize(self, max_sigma, T, schedule, eps):
        if schedule == "cosine":
            thetas = self._cosine_schedule(T)
        elif schedule == "linear":
            thetas = self._linear_schedule(T)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        sigmas = torch.sqrt(max_sigma ** 2 * 2 * thetas)
        thetas_cumsum = torch.cumsum(thetas, dim=0) - thetas[0]
        dt = -1 / thetas_cumsum[-1] * math.log(eps)
        sigma_bars = torch.sqrt(max_sigma ** 2 * (1 - torch.exp(-2 * thetas_cumsum * dt)))

        self.dt = dt.item()
        self.thetas = thetas.to(self.device)
        self.sigmas = sigmas.to(self.device)
        self.thetas_cumsum = thetas_cumsum.to(self.device)
        self.sigma_bars = sigma_bars.to(self.device)

    @staticmethod
    def _cosine_schedule(T, s=0.008):
        steps = T + 3
        x = torch.linspace(0, T + 2, steps, dtype=torch.float32)
        alphas_cumprod = torch.cos(((x / (T + 2)) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        return 1 - alphas_cumprod[1:-1]

    @staticmethod
    def _linear_schedule(T):
        timesteps = T + 1
        scale = 1000 / timesteps
        return torch.linspace(scale * 0.0001, scale * 0.02, timesteps, dtype=torch.float32)

    def set_mu(self, mu: torch.Tensor):
        """Set the mean (LDR condition image)."""
        self.mu = mu

    def set_model(self, model: torch.nn.Module):
        """Set the denoiser network."""
        self.model = model

    def noise_state(self, tensor: torch.Tensor) -> torch.Tensor:
        """Add initial noise to start reverse diffusion."""
        return tensor + torch.randn_like(tensor) * self.max_sigma

    def reverse_posterior(self, xt: torch.Tensor, save_states: bool = False) -> torch.Tensor:
        """
        Run T-step reverse posterior sampling.

        Args:
            xt: Noisy input tensor (1, 3, H, W) on device.

        Returns:
            Denoised output tensor (1, 3, H, W).
        """
        x = xt.clone()
        for t in tqdm(reversed(range(1, self.T + 1)), total=self.T, desc="Diffusion"):
            noise = self.model(x, self.mu, t)
            x = self._reverse_posterior_step(x, noise, t)
        return x

    def _reverse_posterior_step(self, xt, noise, t):
        x0 = self._get_init_state_from_noise(xt, noise, t)
        mean = self._reverse_optimum_step(xt, x0, t)
        std = self._reverse_optimum_std(t)
        return mean + std * torch.randn_like(xt)

    def _get_init_state_from_noise(self, xt, noise, t):
        A = torch.exp(self.thetas_cumsum[t] * self.dt)
        return (xt - self.mu - self.sigma_bars[t] * noise) * A + self.mu

    def _reverse_optimum_step(self, xt, x0, t):
        A = torch.exp(-self.thetas[t] * self.dt)
        B = torch.exp(-self.thetas_cumsum[t] * self.dt)
        C = torch.exp(-self.thetas_cumsum[t - 1] * self.dt)

        term1 = A * (1 - C ** 2) / (1 - B ** 2)
        term2 = C * (1 - A ** 2) / (1 - B ** 2)

        return term1 * (xt - self.mu) + term2 * (x0 - self.mu) + self.mu

    def _reverse_optimum_std(self, t):
        A = torch.exp(-2 * self.thetas[t] * self.dt)
        B = torch.exp(-2 * self.thetas_cumsum[t] * self.dt)
        C = torch.exp(-2 * self.thetas_cumsum[t - 1] * self.dt)

        posterior_var = (1 - A) * (1 - C) / (1 - B)
        min_value = 1e-20 * self.dt
        log_posterior_var = torch.log(torch.clamp(posterior_var, min=min_value))
        return (0.5 * log_posterior_var).exp() * self.max_sigma
