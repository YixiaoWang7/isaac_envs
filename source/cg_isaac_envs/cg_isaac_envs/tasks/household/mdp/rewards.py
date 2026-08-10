"""Order-invariant household task rewards."""

import torch
from .relations import alternative_results


def relation_progress_reward(env):
    return alternative_results(env)[1]


def success_bonus_reward(env):
    return alternative_results(env)[0].float()


def action_l2(env):
    return torch.sum(torch.square(env.action_manager.action), dim=1)
