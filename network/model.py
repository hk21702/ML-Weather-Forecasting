from dataclasses import dataclass

import torch
import torch.nn as nn
import pandas as pd


@dataclass
class ModelConfig:
    """Configuration for the model

    Attributes:
        channels (int): Number of channels in the input
        target_dim (int): Squared dim of the target area
        context_dim (int): Squared dim of the context area
        target_delta (int): Number of time steps to predict into the future
        context_delta (int): Number of time steps to look back in the context
        feature_variables (list[str]): List of features to use (short variable names)
        target_variable (str): Target variable to predict (short variable name)
    """
    channels: int
    target_dim: int
    context_dim: int
    target_delta: int
    context_delta: int
    feature_variables: list[str]
    target_variable: str
