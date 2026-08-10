"""Order-invariant relation rewards."""

import torch

from .relations import alternative_results


def relation_progress(env) -> torch.Tensor:
    return alternative_results(env)[1]


def success_bonus(env) -> torch.Tensor:
    return alternative_results(env)[0].float()


def action_l2(env) -> torch.Tensor:
    return torch.sum(torch.square(env.action_manager.action), dim=1)

