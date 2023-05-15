import torch
from torch import nn
import torch.nn.utils.prune as prune





def get_parameters_to_prune(model):
  parameters_to_prune = []
  for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
      parameters_to_prune.append((module, 'weight'))
  return tuple(parameters_to_prune)
