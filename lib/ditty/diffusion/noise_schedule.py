import torch
from torch import Tensor


class Scheduler:
    def __init__(self, timesteps=1):
        self.timesteps = timesteps
        self.alphas: Tensor
        self.betas: Tensor
        self.alphas_cumprod: Tensor

    def _cosine_beta_schedule(self, s=0.008, max_beta=0.999):
        """
        Cosine schedule as proposed in Improved DDPM paper.
        Matches HuggingFace diffusers betas_for_alpha_bar implementation.
        """
        import math

        def alpha_bar_fn(t):
            return math.cos((t + s) / (1 + s) * math.pi / 2) ** 2

        betas = []
        for i in range(self.timesteps):
            t1 = i / self.timesteps
            t2 = (i + 1) / self.timesteps
            betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))

        return torch.tensor(betas, dtype=torch.float32)

    def _linear_beta_schedule(self, beta_start=0.0001, beta_end=0.02):
        """
        Linear schedule as used in original DDPM paper
        """
        return torch.linspace(beta_start, beta_end, self.timesteps)

    def create_noise_schedule(self, schedule_type="cosine", F=1.0, beta_start=0.0001, beta_end=0.02):
        """
        Create noise schedule. Matches HuggingFace diffusers implementation.

        Args:
            schedule_type: "cosine" or "linear"
            F: Rescale factor for cosine schedule (betas scaled by F², default 1.0)
            beta_start: Starting beta for linear schedule
            beta_end: Ending beta for linear schedule
        """
        if schedule_type == "linear":
            betas = self._linear_beta_schedule(beta_start, beta_end)
        elif schedule_type == "cosine":
            betas = self._cosine_beta_schedule()
            betas = betas * (F**2)
        else:
            raise ValueError(f"Unknown schedule_type: {schedule_type}")

        betas = torch.clip(betas, 0.0001, 0.9999)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.alphas = alphas
        self.betas = betas
        self.alphas_cumprod = alphas_cumprod
