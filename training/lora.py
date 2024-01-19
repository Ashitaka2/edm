import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Callable, Dict, List, Optional, Set, Tuple, Type, Union
import math

import os
import time
import copy
import json
import pickle
import psutil
import numpy as np
import torch
import dnnlib
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc
from torch_utils import persistence

def weight_init(shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform': return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':  return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == 'kaiming_uniform': return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':  return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')

@persistence.persistent_class
class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_mode='kaiming_normal', init_weight=1, init_bias=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(weight_init([out_features, in_features], **init_kwargs) * init_weight)
        self.bias = torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x

@persistence.persistent_class
class GroupNorm(torch.nn.Module):
    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        # self.num_groups = num_groups #channel들에 divide되어야 하는데, 지금 18(t), 9(a) 쓰고 있으니...
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        x = torch.nn.functional.group_norm(x, num_groups=self.num_groups, weight=self.weight.to(x.dtype),
                                           bias=self.bias.to(x.dtype), eps=self.eps)
        return x

def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


class LoraInjectedConv2d(nn.Module): #Conv up/down not compatible   
    def __init__(
            self, in_channels, out_channels, embed_dim, kernel=1, bias=False, scale=1.0, r_lora=None, num_basis=None,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_dim = embed_dim
        self.kernel = kernel
        self.scale = scale
        self.r_lora = r_lora
        self.num_basis = num_basis

        self.conv2d = conv_nd(2, in_channels, out_channels, kernel, padding=kernel//2 if kernel>1 else 0, bias=bias)
        self.lora_down = conv_nd(2, in_channels, r_lora * num_basis, kernel, padding=kernel//2 if kernel>1 else 0, bias=False)
        self.lora_up = conv_nd(2, r_lora * num_basis, out_channels, kernel, padding=kernel//2 if kernel>1 else 0,  bias=False)
        nn.init.normal_(self.lora_down.weight, std=1 / r_lora)
        nn.init.zeros_(self.lora_up.weight)
        self.bias = nn.Parameter(torch.zeros((self.r_lora * self.num_basis, self.out_channels)))

        self.comp_weights = nn.Sequential(
            Linear(in_features=embed_dim, out_features=128),
            GroupNorm(num_channels=128, eps=1e-6),
            torch.nn.SiLU(),
            Linear(in_features=128, out_features=64),
            GroupNorm(num_channels=64, eps=1e-6),
            torch.nn.SiLU(),
            Linear(in_features=64, out_features=num_basis),
        )

        self.bypass = False

    def set_bypass(self, bypass):
        self.bypass = bypass
    
    def forward(self, input, emb):
        if self.bypass:
            return self.conv2d(input)
        
        else:
            mask = self.comp_weights(emb)
            mask = torch.repeat_interleave(mask, self.r_lora, dim=1).to(device=self.conv2d.weight.device, dtype=self.conv2d.weight.dtype)
            
            out_conv = self.conv2d(input)
            out_lora = self.lora_up(mask.unsqueeze(-1).unsqueeze(-1) * self.lora_down(input)) + torch.matmul(mask, self.bias).unsqueeze(-1).unsqueeze(-1)
            return out_conv + self.scale * out_lora


def _find_children(
    model,
    search_class: List[Type[nn.Module]] = [nn.Conv1d],
):
    for parent in model.modules():
        for name, module in parent.named_children():
            if any([isinstance(module, _class) for _class in search_class]):
                yield parent, name, module

def _find_modules(
    model,
    ancestor_class: Optional[Set[str]] = None,
    search_class: List[Type[nn.Module]] = [nn.Conv2d],
    exclude_children_of: Optional[List[Type[nn.Module]]] = [
        LoraInjectedConv2d,
    ],
):
    if ancestor_class is not None:
        ancestors = (
            module
            for module in model.modules()
            if module.__class__.__name__ in ancestor_class
        )
    else:
        # this, incase you want to naively iterate over all modules.
        ancestors = [module for module in model.modules()]

    # For each target find every linear_class module that isn't a child of a LoraInjectedLinear
    for ancestor in ancestors:
        for fullname, module in ancestor.named_modules():
            if any([isinstance(module, _class) for _class in search_class]):
                # Find the direct parent if this is a descendant, not a child, of target
                *path, name = fullname.split(".")
                parent = ancestor
                while path:
                    parent = parent.get_submodule(path.pop(0))
                # Skip this linear if it's a child of a LoraInjectedLinear
                if exclude_children_of and any(
                    [isinstance(parent, _class) for _class in exclude_children_of]
                ):
                    continue
                # Otherwise, yield it
                yield parent, name, module


DEFAULT_TARGET_REPLACE = {"UNetBlock"}
def set_bypass_for_lora(
    model: nn.Module,
    bypass: bool,
    target_replace_module: Set[str] = DEFAULT_TARGET_REPLACE,
    verbose: bool = True,
):
    for _module, name, _child_module in _find_modules(
        model, target_replace_module, search_class=[LoraInjectedConv2d], exclude_children_of=[]
    ):
        if name in ["qkv", "proj_out"]:
            _child_module.set_bypass(bypass)
            if verbose:
                print(f"Setting bypass as {bypass} for {name}")
    return


def inject_trainable_lora(
    model: nn.Module,
    target_replace_module: Set[str] = DEFAULT_TARGET_REPLACE,
    r_lora: int = None,
    loras=None,  # path to lora .pt
    num_basis: int = None,
    conditioning: list = None, # ['res', 'attn']
    scale: float = 1.0,
    verbose: bool = False,
):
    """
    inject lora into model, and returns lora parameter groups.
    """

    require_grad_params = []
    names = []
    if loras != None:
        loras = torch.load(loras)
    
    targets = []
    if 'res' in conditioning:
        targets.append("conv1")
    if 'attn' in conditioning:
        targets.append("qkv")
        targets.append("proj")
    
    from .networks import Conv2d
    for _module, name, _child_module in _find_modules(
        model, target_replace_module , search_class=[Conv2d]
    ):
        # if name in ["qkv", "proj"]:
        if name in targets:
            weight = _child_module.weight
            bias = _child_module.bias
            if verbose:
                print("LoRA Injection : injecting lora into ", name)
                print("LoRA Injection : weight shape", weight.shape) 
                
            _tmp = LoraInjectedConv2d(
                _child_module.in_channels,
                _child_module.out_channels,
                _module.emb_channels,
                _child_module.kernel,
                _child_module.bias is not None,
                scale=scale,
                r_lora=r_lora,
                num_basis=num_basis,
            )
            _tmp.conv2d.weight = weight
            if bias is not None:
                _tmp.conv2d.bias = bias

            # switch the module
            _tmp.to(_child_module.weight.device).to(_child_module.weight.dtype)
            _module._modules[name] = _tmp

            # require_grad_params.append(_module._modules[name].lora_up.parameters())
            # require_grad_params.append(_module._modules[name].lora_down.parameters())
            # if interpolate == "train":
            #     require_grad_params.append(_module._modules[name].embedding.parameters())

            if loras != None:
                _module._modules[name].lora_up.weight = loras.pop(0)
                _module._modules[name].lora_down.weight = loras.pop(0)
            # _module._modules[name].lora_up.weight.requires_grad = True
            # _module._modules[name].lora_down.weight.requires_grad = True
            names.append(name)

    return require_grad_params, names

