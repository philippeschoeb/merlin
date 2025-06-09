"""
Quantum measurement sampling utilities.
"""

import torch


class SamplingProcess:
    """Handles quantum measurement sampling with different methods."""

    def __init__(self):
        self.gradient_method = "exact"  # Always use exact for gradients

    def pcvl_sampler(self, distribution: torch.Tensor, shots: int,
                     method: str = 'multinomial') -> torch.Tensor:
        """Apply sampling noise to a probability distribution."""
        if shots <= 0:
            return distribution

        # Validate method
        valid_methods = ['multinomial', 'binomial', 'gaussian']
        if method not in valid_methods:
            raise ValueError(f"Invalid sampling method: {method}. Valid options are: {valid_methods}")

        if method == 'multinomial':
            if distribution.dim() == 1:
                sampled_counts = torch.multinomial(
                    distribution, num_samples=shots, replacement=True
                )
                noisy_dist = torch.zeros_like(distribution)
                for idx in sampled_counts:
                    noisy_dist[idx] += 1
                return noisy_dist / shots
            else:
                batch_size = distribution.shape[0]
                noisy_dists = []
                for i in range(batch_size):
                    sampled_counts = torch.multinomial(
                        distribution[i], num_samples=shots, replacement=True
                    )
                    noisy_dist = torch.zeros_like(distribution[i])
                    for idx in sampled_counts:
                        noisy_dist[idx] += 1
                    noisy_dists.append(noisy_dist / shots)
                return torch.stack(noisy_dists)

        elif method == 'binomial':
            return torch.distributions.Binomial(shots, distribution).sample() / shots

        elif method == 'gaussian':
            std_dev = torch.sqrt(distribution * (1 - distribution) / shots)
            noise = torch.randn_like(distribution) * std_dev
            noisy_dist = distribution + noise
            noisy_dist = torch.clamp(noisy_dist, 0, 1)
            noisy_dist = noisy_dist / noisy_dist.sum(dim=-1, keepdim=True)
            return noisy_dist