"""Multi-armed bandit (MAB) policies.

The bandit policies in River have a generic API. This allows them to be used in a variety of
situations. For instance, they can be used for model selection
(see `model_selection.BanditRegressor`).

"""

from . import base, envs
from .epsilon_greedy import EpsilonGreedy
from .evaluate import evaluate, evaluate_offline
from .thompson import ThompsonSampling
from .ucb import UCB

__all__ = [
    "base",
    "envs",
    "evaluate",
    "evaluate_offline",
    "EpsilonGreedy",
    "ThompsonSampling",
    "UCB",
]
